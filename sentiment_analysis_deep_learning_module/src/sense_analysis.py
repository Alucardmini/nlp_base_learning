#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: sense_analysis.py
@time: 12/3/18 10:10 AM
"""

from keras.layers import Embedding, LSTM, Dense, Dropout,Activation
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adagrad
import pandas as pd
import numpy as np

batch_size=256
min_count=5
max_len=200

pos = pd.read_excel('../data/pos.xls', header=None)
neg = pd.read_excel('../data/neg.xls', header=None)
pos['label'] = 1
neg['label'] = 0
alldata = pos.append(neg, ignore_index=True)

content = ''.join(alldata[0])
vocab_series = pd.Series(list(content)).value_counts()
vocab_series = vocab_series[vocab_series > min_count]
vocab_series[:] = list(range(1, len(vocab_series) + 1))
# 空字符串补全
vocab_series[''] = 0

word_set = set(vocab_series.index)
# print(word_set)

def doc2num(s):
    s = [x for x in s if x in word_set]
    s = s[:max_len] + [''] * max(0, (max_len - len(s)))
    return list(vocab_series[s])

alldata['doc2num'] = alldata[0].apply(lambda s: doc2num(s))
x = np.array(list(alldata['doc2num']))
y = np.array(list(alldata['label'])).reshape((-1, 1))

train_size = 15000
train_x = x[:train_size]
train_y = y[:train_size]
test_x = x[train_size:]
test_y = y[train_size:]

model = Sequential()
model.add(Embedding(len(vocab_series), 256, input_length=max_len))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=batch_size, epochs=5)
score = model.evaluate(test_x, test_y,batch_size=batch_size)
print(score)