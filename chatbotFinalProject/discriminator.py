import sys
sys.path.append('..')
import collections
import mxnet as mx
from mxnet import autograd, gluon, init, metric, nd
from mxnet.gluon import loss as gloss, nn, rnn
from mxnet.contrib import text
import os
import random
import zipfile

demo = True
if demo:
    with zipfile.ZipFile('data/aclImdb_tiny.zip', 'r') as zin:
        zin.extractall('data/')


def readIMDB(dir_url, seg='train'):
    pos_or_neg = ['pos', 'neg']
    data = []
    for label in pos_or_neg:
        files = os.listdir(
            'data/' + dir_url + '/' + seg + '/' + label + '/')
        for file in files:
            with open('data/' + dir_url + '/' + seg + '/' + label + '/'
                      + file, 'r', encoding='utf8') as rf:
                review = rf.read().replace('\n', '')
                if label == 'pos':
                    data.append([review, 1])
                elif label == 'neg':
                    data.append([review, 0])
    return data

if demo:
    train_data = readIMDB('aclImdb_tiny/', 'train')
    test_data = readIMDB('aclImdb_tiny/', 'test')
else:
    train_data = readIMDB('aclImdb/', 'train')
    test_data = readIMDB('aclImdb/', 'test')

random.shuffle(train_data)
random.shuffle(test_data)


def tokenizer(text):
    return [tok.lower() for tok in text.split(' ')]

train_tokenized = []
for review, score in train_data:
    train_tokenized.append(tokenizer(review))
test_tokenized = []
for review, score in test_data:
    test_tokenized.append(tokenizer(review))


token_counter = collections.Counter()
def count_token(train_tokenized):
    for sample in train_tokenized:
        for token in sample:
            if token not in token_counter:
                token_counter[token] = 1
            else:
                token_counter[token] += 1

count_token(train_tokenized)
vocab = text.vocab.Vocabulary(token_counter, unknown_token='<unk>',
                              reserved_tokens=None)

def encode_samples(tokenized_samples, vocab):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in vocab.token_to_idx:
                feature.append(vocab.token_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features

def pad_samples(features, maxlen=500, padding=0):
    padded_features = []
    for feature in features:
        if len(feature) > maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            # 添加 PAD 符号使每个序列等长（长度为 maxlen ）。
            while len(padded_feature) < maxlen:
                padded_feature.append(padding)
        padded_features.append(padded_feature)
    return padded_features

ctx = mx.gpu()
train_features = encode_samples(train_tokenized, vocab)
test_features = encode_samples(test_tokenized, vocab)
train_features = nd.array(pad_samples(train_features, 500, 0), ctx=ctx)
test_features = nd.array(pad_samples(test_features, 500, 0), ctx=ctx)
train_labels = nd.array([score for _, score in train_data], ctx=ctx)
test_labels = nd.array([score for _, score in test_data], ctx=ctx)

# print(train_features[0])
# print(train_labels[0])




glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)

class SentimentNet(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers,
                 bidirectional, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding = nn.Embedding(len(vocab), embed_size)
            self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                    bidirectional=bidirectional,
                                    input_size=embed_size)
            self.decoder = nn.Dense(num_outputs, flatten=False)

    def forward(self, inputs, begin_state=None):
        embeddings = self.embedding(inputs)
        states = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态。
        encoding = nd.concat(states[0], states[-1])
        outputs = self.decoder(encoding)
        return outputs


num_outputs = 2
lr = 0.1
num_epochs = 100
batch_size = 10
embed_size = 100
num_hiddens = 100
num_layers = 2
bidirectional = True

net = SentimentNet(vocab, embed_size, num_hiddens, num_layers, bidirectional)
net.initialize(init.Xavier(), ctx=ctx)
# 设置 embedding 层的 weight 为预训练的词向量。
net.embedding.weight.set_data(glove_embedding.idx_to_vec.as_in_context(ctx))
# 训练中不迭代词向量（net.embedding中的模型参数）。
net.embedding.collect_params().setattr('grad_req', 'null')
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()

# def eval_model(features, labels):
#     l_sum = 0
#     l_n = 0
#     accuracy = metric.Accuracy()
#     for i in range(features.shape[0] // batch_size):
#         X = features[i*batch_size : (i+1)*batch_size].as_in_context(ctx).T
#         y = labels[i*batch_size :(i+1)*batch_size].as_in_context(ctx).T
#         output = net(X)
#         l = loss(output, y)
#         l_sum += l.sum().asscalar()
#         l_n += l.size
#         accuracy.update(preds=nd.argmax(output, axis=1), labels=y)
#     return l_sum / l_n, accuracy.get()[1]

# net.load_params('classfication.params')
# for epoch in range(1, num_epochs + 1):
#     for i in range(train_features.shape[0] // batch_size):
#         X = train_features[i*batch_size : (i+1)*batch_size].as_in_context(
#             ctx).T
#         y = train_labels[i*batch_size : (i+1)*batch_size].as_in_context(
#             ctx).T
#         with autograd.record():
#             l = loss(net(X), y)
#         l.backward()
#         trainer.step(batch_size)
#     train_loss, train_acc = eval_model(train_features, train_labels)
#     test_loss, test_acc = eval_model(test_features, test_labels)
#     print('epoch %d, train loss %.6f, acc %.2f; test loss %.6f, acc %.2f'
#           % (epoch, train_loss, train_acc, test_loss, test_acc))
# net.save_params('classfication.params')


# def inference(set):
#         feature = []
#         for token in set:
#             if token in vocab.token_to_idx:
#                 feature.append(vocab.token_to_idx[token])
#             else:
#                 feature.append(0)
#         return feature
#
#
# review = ['i', 'think', 'it', 'is','low']
# inf_data = inference(review)
#
#
# print(nd.argmax(net(nd.reshape(
#     nd.array( inf_data, ctx=ctx),
#     shape=(-1, 1))), axis=1).asscalar())