# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '11/26/18'

import numpy as np
import pandas as pd
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
import sys
import os
sys.setrecursionlimit(10000) #增大堆栈最大深度(递归深度)，据说默认为1000，报错


class SenseAnalysis(object):
    def __init__(self, pos_path, neg_path, maxlen=200, min_count=20, batch_size=1024, train_num=15000):
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.maxlen = maxlen
        self.min_count = min_count
        self.batch_size = batch_size
        self.train_num = train_num
        self.pos, self.neg, self.all_ = self.loadData()

    def loadData(self):
        pos = pd.read_excel(self.pos_path, header=None)
        pos['label'] = 1
        neg = pd.read_excel(self.neg_path, header=None)
        neg['label'] = 1
        all_ = pos.append(neg, ignore_index=True)
        return pos, neg, all_

    def preProcess(self):
        content = ''.join(self.all_[0])
        abc = pd.Series(list(content)).value_counts()
        abc = abc[abc >= self.min_count]
        abc[:] = list(range(len(abc)))
        word_set = set(abc.index)
        self.all_['doc2num'] = self.all_[0].apply(lambda s: self.doc2num(src_data=abc, s=s,
                                                                         maxlen=self.maxlen,
                                                                         word_set=word_set))
        idx = list(range(len(self.all_)))
        np.random.shuffle(idx)
        self.all_ = self.all_.loc[idx]
        x = np.array(list(self.all_['doc2num']))
        y = np.array(list(self.all_['label']))
        y = y.reshape((-1, 1))
        return x, y

    def doc2num(self, src_data, s, maxlen, word_set):
        s = [i for i in s if i in word_set]
        s = s[:maxlen]
        return list(src_data[s])

    def buildModel(self):
        model = Sequential
        model.add(Embedding(len(), 256, input_length=self.maxlen))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def fitModel(self):
        model = self.buildModel()
        x, y = self.preProcess()
        model.fit(x[:self.train_num], y[:self.train_num],
                  batch_size=self.batch_size,
                  nb_epoch=30)

        model.evaluate(x[self.train_num:], y[self.train_num:], batch_size=self.batch_size)





if __name__ == "__main__":
    pass