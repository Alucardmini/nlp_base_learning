#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: bi_category.py
@time: 12/6/18 5:16 PM
"""

from keras.datasets import imdb
from keras.layers import Dense, Activation, Input
from keras.preprocessing import sequence
from keras.models import Sequential
import numpy as np

batch_size = 256
max_len = 512

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=8000)

x_train = sequence.pad_sequences(sequences=x_train, maxlen=max_len)
x_test = sequence.pad_sequences(sequences=x_test, maxlen=max_len)

# https://blog.csdn.net/bqw18744018044/article/details/82598131
def vectorize_sequences(sequences, dimension=max_len):
# 生成25000*8000的二维Numpy数组
    results = np.zeros((len(sequences),dimension))
    # one-hot编码
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
        return results
x_train = vectorize_sequences(x_train)
x_test = vectorize_sequences(x_test)



model = Sequential()


model.add(Dense(input_shape=(len(x_train), ), units=256))
model.add(Activation('relu'))
model.add(Dense(units=128))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size)

score = model.evaluate(x_test, y_test, batch_size=batch_size)
print(score)



