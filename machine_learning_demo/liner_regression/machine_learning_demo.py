#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: machine_learning_demo.py
@time: 12/6/18 4:44 PM
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

x = np.linspace(-2, 3, 1000)
y = 2*x + 0.2 + np.random.normal(0, 0.05) #生成Y并添加噪声
batch_size = 32
train_size = 800
x_train = x[0:train_size]
y_train = y[0:train_size]
x_test = x[train_size:]
y_test = y[train_size:]

model = Sequential()
model.add(Dense(units=1, input_shape=(1,), use_bias=True),)
model.compile(optimizer='sgd',
              loss='mse',
              )
model.fit(x_train, y_train, batch_size=batch_size, epochs=40, initial_epoch=0)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

print(y_test)


res = model.predict(x_test, batch_size=len(x_test))

print('=====')

print(res)