# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '11/21/18'

import pandas as pd
import numpy as np
import jieba
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
# from __future__ import absolute_import
# from __future__ import print_function

neg = pd.read_excel('../data/neg.xls', header=None, index_col=None)
pos = pd.read_excel('../data/pos.xls', header=None, index_col=None)
pos['mark'] = 1
neg['mark'] = 0

pn = pd.concat([pos, neg], ignore_index=True)
neglen = len(neg)
poslen = len(pos)

cw = lambda x: list(jieba.cut(x))
pn['words'] = pn[0].apply(cw)

comment = pd.read_excel('../data/sum.xls')
comment = comment[comment['rateContent'].notnull()]
comment['words'] = comment['rateContent'].apply(cw)

d2v_train = pd.concat([pn['words'], comment['words']], ignore_index=True)

w = []
for i in d2v_train:
    w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts())
del w, d2v_train
dict['id'] = list(range(1, len(dict) + 1))
get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent)

maxlen = 50

print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

x = np.array(list(pn['sent']))[::2]
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2]
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent']))
ya = np.array(list(pn['mark']))

print('build model ...')
model = Sequential()
model.add(Embedding(len(dict) + 1), 256)
model.add(LSTM(256, 128))
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              classmethod='binary')

model.fit(x, y, batch_size=16, np_epoch=10)
classes = model.predict_classes(xt)
# acc = np_utils.accuracy(classes, yt)
