#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: mlp_mnist.py
@time: 12/10/18 4:44 PM
"""

from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
from keras.optimizers import RMSprop
from keras.utils import np_utils

batch_size = 128
num_class = 10
epoch = 30

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(len(x_train), 28*28)
x_test = x_test.reshape(len(x_test), 28*28)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, num_class)
y_test = np_utils.to_categorical(y_test, num_class)

model = Sequential()
model.add(Dense(input_shape=(28*28,), units=512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_class, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epoch)




