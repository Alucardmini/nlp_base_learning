#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: embedding_demo.py
@time: 12/7/18 10:07 AM
"""

from keras.layers import Embedding, Dense, Dropout, Flatten, Conv2D, LSTM,Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.datasets import imdb
import numpy as np

max_features = 25000
max_size = 200
(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=max_size)
x_test = sequence.pad_sequences(x_test, maxlen=max_size)

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=512)