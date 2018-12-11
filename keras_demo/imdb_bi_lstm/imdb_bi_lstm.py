#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: imdb_bi_lstm.py
@time: 12/11/18 3:27 PM
"""

from keras.layers import LSTM, Dense, Dropout, Activation, Embedding, Bidirectional
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.datasets import imdb
import numpy as np

max_features = 20000
max_len = 200
batch_size = 32

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, max_len)
    x_test = sequence.pad_sequences(x_test, max_len)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = Sequential()
    # model.add()
    model.add(Embedding(input_dim=max_features, output_dim=256))
    # model.add(LSTM(256))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=4,
              validation_data=[x_test, y_test])

