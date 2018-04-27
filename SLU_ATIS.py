import speech_recognition as sr

import numpy as np
import data.load
from keras.models import load_model

train_set, valid_set, dicts = data.load.atisfull()
w2idx, labels2idx = dicts['words2idx'], dicts['labels2idx']

train_x, _, train_label = train_set
val_x, _, val_label = valid_set

# Create index to word/label dicts
idx2w = {w2idx[k]: k for k in w2idx}

idx2la = {labels2idx[k]: k for k in labels2idx}

words_train = [list(map(lambda x: idx2w[x], w)) for w in train_x]
print(train_x[0])
labels_train = [list(map(lambda x: idx2la[x], y)) for y in train_label]
words_val = [list(map(lambda x: idx2w[x], w)) for w in val_x]
labels_val = [list(map(lambda x: idx2la[x], y)) for y in val_label]

n_classes = len(idx2la)
print(n_classes)
n_vocab = len(idx2w)
print(n_vocab)

print("Example sentence : {}".format(words_train[0]))
print("Encoded form: {}".format(train_x[0]))
print()
print("It's label : {}".format(labels_train[0]))
print("Encoded form: {}".format(train_label[0]))

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D

model = Sequential()
model.add(Embedding(n_vocab, 100))
model.add(Dropout(0.25))
model.add(SimpleRNN(100, return_sequences=True))
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
model.compile('rmsprop', 'categorical_crossentropy')
model = load_model('my_model.h5')

#obtain audio from the microphone
r=sr.Recognizer()
with sr.Microphone() as source:
    print("Please wait. Calibrating microphone...")
    #listen for 5 seconds and create the ambient noise energy level
    r.adjust_for_ambient_noise(source, duration=5)
    print("Say something!")
    audio=r.listen(source)

# recognize speech using Google Speech Recognition
try:
    print("Google Speech Recognition thinks you said:")
    print(r.recognize_google(audio, language="zh-TW"))
    string = r.recognize_google(audio, language="zh-TW")
    from googletrans import Translator
    translator = Translator()
    print(translator.translate(string).text)
    sentence = translator.translate(string).text
    sentence =sentence.lower()

    print(sentence)
    a = sentence.split(' ')
    print(a)
    encode = []
    for i in a:
        if i in w2idx:
            encode.append(w2idx[i])
        else:
            encode.append(126)
    labels_pred_val = []
    test = []
    encode = np.array(encode)
    encode = encode[np.newaxis, :]

    for n_batch, sent in enumerate(encode[:]):
        label = val_label[n_batch]
        label = np.eye(n_classes)[label][np.newaxis, :]
        sent = sent[np.newaxis, :]

        pred = model.predict_on_batch(sent)
        pred = np.argmax(pred, -1)[0]
        labels_pred_val.extend(pred)



    result = []
    for i in labels_pred_val:
        if i in idx2la:
            result.append(idx2la[i])
        else:
            result.append('o')

    print(result)

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("No response from Google Speech Recognition service: {0}".format(e))
