# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '12/1/18'

from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Model
from keras.layers import *

try:

    from src.attention_keras import Position_Embedding
    from src.attention_keras import Attention
except:
    from attention.src.attention_keras import Attention
    from attention.src.attention_keras import Position_Embedding

maxfeature = 20000
batch_size = 32
maxlen = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=maxfeature)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

s_inputs = Input(shape=(None, ), dtype='int32')
embeddings = Embedding(maxfeature, 128)(s_inputs)

O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
# O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = GlobalMaxPool1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
# O_seq = MyLayer(10)(O_seq)

outputs = Dense(1, activation='sigmoid')(O_seq)

model = Model(s_inputs, outputs)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))