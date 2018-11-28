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


class Sentiment_analysis(object):

    def __init__(self, pos_path, neg_path, maxlen=200, min_count=20, batch_size=128, train_num=1000):
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.maxlen = maxlen
        self.min_count = min_count
        self.batch_size = batch_size
        self.train_num = train_num
        self.pos, self.neg, self.all_ = self.load_data()

    def load_data(self):
        pos = pd.read_excel(self.pos_path, header=None)
        pos['label'] = 1
        neg = pd.read_excel(self.neg_path, header=None)
        neg['label'] = 0
        all_ = pos.append(neg, ignore_index=True)
        return pos, neg, all_

    def build_lstm_data(self):
        model = Sequential()
        model.add(Embedding(), 256)
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def open_fire(self):
        model = self.buildLstmModel()
        model.fit()