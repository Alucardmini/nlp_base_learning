#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: bi_demo.py
@time: 12/7/18 3:36 PM
"""

from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, LSTM, Input, Embedding
from keras.preprocessing import sequence

max_features = 10000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)



s_inputs = Input(shape=(max_len, ), dtype='float32')
x = Dense(64, activation='relu')(s_inputs)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
out_put = Dense(1, activation='sigmoid')(x)
model = Model(s_inputs, outputs=out_put)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=512, epochs=3000)