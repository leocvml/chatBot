import collections
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wname", help="name of weighting", type=str)
parser.add_argument("--epoch", help="number of epochs", type=int)
parser.add_argument("--dataset", help="name of dataset", type=str)
parser.add_argument("--retrain", help="load weighting and continue training", type=int)

args = parser.parse_args()

PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'

w_name = args.wname
epochs = args.epoch
dataset = args.dataset
retrain = args.retrain

epoch_period = 10
lr = 0.005
batch_size = 128
# max seq length
max_seq_len = 20
# max test output length
max_test_output_len = 20
encoder_num_layers = 1
decoder_num_layers = 1
encoder_drop_prob = 0.1
decoder_drop_prob = 0.1
encoder_num_hiddens = 256
decoder_num_hiddens = 256
alignment_size = 10
ctx = mx.gpu(0)


def read_data(dataset, max_seq_len):
    input_tokens = []
    output_tokens = []
    input_seqs = []
    output_seqs = []
    with open(dataset, 'r', encoding = 'utf8') as f:
        lines = f.readlines()
        for line in lines:
            input_seq, output_seq = line.rstrip().split('@')
            cur_input_tokens = input_seq.split(' ')
            cur_output_tokens = output_seq.split(' ')
            if len(cur_input_tokens) < max_seq_len and \
                    len(cur_output_tokens) < max_seq_len:
                input_tokens.extend(cur_input_tokens)
                # add eos in end of seq
                cur_input_tokens.append(EOS)
                # use pad for each seq make them have same length
                while len(cur_input_tokens) < max_seq_len:
                    cur_input_tokens.append(PAD)
                input_seqs.append(cur_input_tokens)
                output_tokens.extend(cur_output_tokens)
                cur_output_tokens.append(EOS)
                while len(cur_output_tokens) < max_seq_len:
                    cur_output_tokens.append(PAD)
                output_seqs.append(cur_output_tokens)
        fr_vocab = text.vocab.Vocabulary(collections.Counter(input_tokens),
                                         reserved_tokens=[PAD, BOS, EOS])
        en_vocab = text.vocab.Vocabulary(collections.Counter(output_tokens),
                                         reserved_tokens=[PAD, BOS, EOS])
    return fr_vocab, en_vocab, input_seqs, output_seqs

class Encoder(nn.Block):
    """ encoder"""

    def __init__(self, num_inputs, num_hiddens, num_layers, drop_prob,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding = nn.Embedding(num_inputs, num_hiddens)
            self.dropout = nn.Dropout(drop_prob)
            self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob,
                               input_size=num_hiddens)

    def forward(self, inputs, state):
        embedding = self.embedding(inputs).swapaxes(0, 1)
        embedding = self.dropout(embedding)
        output, state = self.rnn(embedding, state)
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

class Decoder(nn.Block):
    # decoder with attention
    def __init__(self, num_hiddens, num_outputs, num_layers, max_seq_len,
                 drop_prob, alignment_size, encoder_num_hiddens, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.encoder_num_hiddens = encoder_num_hiddens
        self.hidden_size = num_hiddens
        self.num_layers = num_layers
        with self.name_scope():
            self.embedding = nn.Embedding(num_outputs, num_hiddens)
            self.dropout = nn.Dropout(drop_prob)
            # attention model

            self.attention = nn.Sequential()
            with self.attention.name_scope():
                self.attention.add(
                    nn.Dense(alignment_size,
                             in_units=num_hiddens + encoder_num_hiddens,
                             activation="tanh", flatten=False))
                self.attention.add(nn.Dense(1, in_units=alignment_size,
                                            flatten=False))

            self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob,
                               input_size=num_hiddens)
            self.out = nn.Dense(num_outputs, in_units=num_hiddens,
                                flatten=False)
            self.rnn_concat_input = nn.Dense(
                num_hiddens, in_units=num_hiddens + encoder_num_hiddens,
                flatten=False)

    def forward(self, cur_input, state, encoder_outputs):
        # get the layer whitch is close output
        single_layer_state = [state[0][-1].expand_dims(0)]
        encoder_outputs = encoder_outputs.reshape((self.max_seq_len, -1,
                                                   self.encoder_num_hiddens))
        hidden_broadcast = nd.broadcast_axis(single_layer_state[0], axis=0,
                                             size=self.max_seq_len)
        encoder_outputs_and_hiddens = nd.concat(encoder_outputs,
                                                hidden_broadcast, dim=2)
        energy = self.attention(encoder_outputs_and_hiddens)
        batch_attention = nd.softmax(energy, axis=0).transpose((1, 2, 0))
        batch_encoder_outputs = encoder_outputs.swapaxes(0, 1)
        decoder_context = nd.batch_dot(batch_attention, batch_encoder_outputs)
        input_and_context = nd.concat(
            nd.expand_dims(self.embedding(cur_input), axis=1),
            decoder_context, dim=2)
        concat_input = self.rnn_concat_input(input_and_context).reshape(
            (1, -1, 0))
        concat_input = self.dropout(concat_input)
        state = [nd.broadcast_axis(single_layer_state[0], axis=0,
                                   size=self.num_layers)]
        output, state = self.rnn(concat_input, state)
        output = self.dropout(output)
        output = self.out(output).reshape((-3, -1))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

class DecoderInitState(nn.Block):

    def __init__(self, encoder_num_hiddens, decoder_num_hiddens, **kwargs):
        super(DecoderInitState, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(decoder_num_hiddens,
                                  in_units=encoder_num_hiddens,
                                  activation="tanh", flatten=False)

    def forward(self, encoder_state):
        return [self.dense(encoder_state)]

def translate(encoder, decoder, decoder_init_state, QA_pair, ctx, max_seq_len):
    random_matrix = np.arange(len(QA_pair))
    np.random.shuffle(random_matrix)
    #print(random_matrix)
    for idx in random_matrix[:5]:
        print('[input] ', QA_pair[idx][0])
        input_tokens = QA_pair[idx][0].split(' ') + [EOS]
        # use padding =max_seq_len

        while len(input_tokens) < max_seq_len:
            input_tokens.append(PAD)
        inputs = nd.array(input_vocab.to_indices(input_tokens), ctx=ctx)
        encoder_state = encoder.begin_state(func=nd.zeros, batch_size=1,
                                            ctx=ctx)
        encoder_outputs, encoder_state = encoder(inputs.expand_dims(0),
                                                 encoder_state)
        encoder_outputs = encoder_outputs.flatten()
        # use bos as first input on decoder

        decoder_input = nd.array([output_vocab.token_to_idx[BOS]], ctx=ctx)
        decoder_state = decoder_init_state(encoder_state[0])
        output_tokens = []

        for _ in range(max_test_output_len):
            decoder_output, decoder_state = decoder(
                decoder_input, decoder_state, encoder_outputs)
            pred_i = int(decoder_output.argmax(axis=1).asnumpy()[0])
            # when out is eos , finish seq

            if pred_i == output_vocab.token_to_idx[EOS]:
                break
            else:
                output_tokens.append(output_vocab.idx_to_token[pred_i])
            decoder_input = nd.array([pred_i], ctx=ctx)
        print('[output]', ' '.join(output_tokens))
        print('[expect]', QA_pair[idx][1], '\n')

import time
def train(encoder, decoder, decoder_init_state, max_seq_len, ctx, retrain, test):
    encoder.initialize(init.Xavier(), ctx=ctx)
    decoder.initialize(init.Xavier(), ctx=ctx)
    decoder_init_state.initialize(init.Xavier(), ctx=ctx)
    we = w_name + '_encoder.params'
    wd = w_name + '_decoder.params'
    wd_init = w_name + '_decoderinit.params'
    if retrain == 1:
        encoder.load_params(we)
        decoder.load_params(wd)
        decoder_init_state.load_params(wd_init)
        print('load params!!!!!!')
    encoder_optimizer = gluon.Trainer(encoder.collect_params(), 'adam',
                                      {'learning_rate': lr})
    decoder_optimizer = gluon.Trainer(decoder.collect_params(), 'adam',
                                      {'learning_rate': lr})
    decoder_init_state_optimizer = gluon.Trainer(
        decoder_init_state.collect_params(), 'adam', {'learning_rate': lr})

    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    l_sum = 0

    for epoch in range(1, epochs + 1):
        tic = time.time()
        for x, y in data_iter:
            cur_batch_size = x.shape[0]
            with autograd.record():
                l = nd.array([0], ctx=ctx)
                valid_length = nd.array([0], ctx=ctx)
                encoder_state = encoder.begin_state(
                    func=nd.zeros, batch_size=cur_batch_size, ctx=ctx)
                encoder_outputs, encoder_state = encoder(x, encoder_state)
                encoder_outputs = encoder_outputs.flatten()
                # use bos as init on decoder

                decoder_input = nd.array(
                    [output_vocab.token_to_idx[BOS]] * cur_batch_size,
                    ctx=ctx)
                mask = nd.ones(shape=(cur_batch_size,), ctx=ctx)
                decoder_state = decoder_init_state(encoder_state[0])
                for i in range(max_seq_len):
                    decoder_output, decoder_state = decoder(
                        decoder_input, decoder_state, encoder_outputs)
                    # use decoder current  predict  for next  state input

                    decoder_input = decoder_output.argmax(axis=1)
                    valid_length = valid_length + mask.sum()
                    l = l + (mask * loss(decoder_output, y[:, i])).sum()
                    mask = mask * (y[:, i] != eos_id)
                l = l / valid_length
            l.backward()
            encoder_optimizer.step(1)
            decoder_optimizer.step(1)
            decoder_init_state_optimizer.step(1)
            l_sum += l.asscalar() / max_seq_len

        if epoch % epoch_period == 0 or epoch == 1:
            if epoch == 1:
                print('epoch %d, loss %f, ' % (epoch, l_sum / len(data_iter)))
            else:
                print('epoch %d, loss %f'
                      % (epoch, l_sum / epoch_period / len(data_iter)))
                translate(encoder, decoder, decoder_init_state, test, ctx, max_seq_len)
            if epoch != 1:
                l_sum = 0


        encoder.save_params(we)
        decoder.save_params(wd)
        decoder_init_state.save_params(wd_init)


input_vocab, output_vocab, input_seqs, output_seqs = read_data(dataset, max_seq_len)
Q = nd.zeros((len(input_seqs), max_seq_len), ctx=ctx)
A = nd.zeros((len(output_seqs), max_seq_len), ctx=ctx)
for i in range(len(input_seqs)):
    Q[i] = nd.array(input_vocab.to_indices(input_seqs[i]), ctx=ctx)
    A[i] = nd.array(output_vocab.to_indices(output_seqs[i]), ctx=ctx)

dataset = gdata.ArrayDataset(Q, A)
loss = gloss.SoftmaxCrossEntropyLoss()
eos_id = output_vocab.token_to_idx[EOS]

encoder = Encoder(len(input_vocab), encoder_num_hiddens, encoder_num_layers,
                   encoder_drop_prob)
decoder = Decoder(decoder_num_hiddens, len(output_vocab),
                   decoder_num_layers, max_seq_len, decoder_drop_prob,
                   alignment_size, encoder_num_hiddens)
decoder_init_state = DecoderInitState(encoder_num_hiddens,
                                       decoder_num_hiddens)

QA_pair = []
with open(args.dataset, 'r', encoding = 'utf8') as f:
    pair = f.readlines()

    for i in pair:
        q, a = i.split('@')
        QA_pair.append([q, a[:-1]])
# #print(QA_pair)
#
train(encoder, decoder, decoder_init_state, max_seq_len, ctx, retrain, QA_pair)
#
#
#
# # with open('another.txt', 'r',encoding = 'utf8') as f:
# #     lines = f.readlines()
# #
# # print(lines[:3])
