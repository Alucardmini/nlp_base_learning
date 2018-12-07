# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 23:52:46 2018
序贯模型实例
@author: BruceWong
"""
#MLP的二分类：
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import imdb

import keras.models as models
import keras.layers as layers

(x_train, y_train), (x_test, y_test) =imdb.load_data(num_words=10000)
print(x_train.shape)


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

from keras import optimizers
# 配置优化器
model.compile(optimizer=optimizers.adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))