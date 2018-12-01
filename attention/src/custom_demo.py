# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '12/1/18'


from keras.models import Model
from keras.models import Sequential
from keras.layers import Embedding
import numpy as np

model = Sequential()
model.add(Embedding(32, 3, input_length=3))
input_array = np.random.randint(10, size=(32, 3))
print(input_array)
print('---')
model.compile(optimizer='rmsprop', loss='mse')
output_array = model.predict(input_array)
print(output_array)