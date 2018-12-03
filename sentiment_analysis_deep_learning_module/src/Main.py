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

    def __init__(self, pos_path, neg_path, maxlen=200, min_count=5, batch_size=128, train_num=1000):
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.maxlen = maxlen
        self.min_count = min_count
        self.batch_size = batch_size
        self.train_num = train_num
        self.pos, self.neg, self.all_ = self.load_data()
        self.vocabulary_series = None

    def load_data(self):
        pos = pd.read_excel(self.pos_path, header=None)
        pos['label'] = 1
        neg = pd.read_excel(self.neg_path, header=None)
        neg['label'] = 0
        all_ = pos.append(neg, ignore_index=True)
        return pos, neg, all_

    def pre_process(self):
        # 拼接语料
        content = ''.join(self.all_[0])
        # 按个数序列化全文
        vocabulary_series = pd.Series(list(content)).value_counts()
        vocabulary_series = vocabulary_series[vocabulary_series > self.min_count]
        vocabulary_series[:] = list(range(1, len(vocabulary_series) + 1))
        # 空字符串补全
        vocabulary_series[''] = 0
        self.vocabulary_series = vocabulary_series
        word_set = set(vocabulary_series.index)

        def doc2num(s, maxlen):
            s = [i for i in s if i in word_set]
            s = s[:maxlen] + ['']*max(0, maxlen - len(s))
            return list(vocabulary_series[s])

        self.all_['doc2num'] = self.all_[0].apply(lambda s: doc2num(s, self.maxlen))
        # shuffle data 打乱数据
        idx = list(range(len(self.all_)))
        np.random.shuffle(idx)
        self.all_ = self.all_.loc[idx]
        x_train = np.array(list(self.all_['doc2num']))
        y_train = np.array(list(self.all_['label'])).reshape((-1, 1))
        return x_train, y_train

    def build_lstm_data(self):
        model = Sequential()
        model.add(Embedding(len(self.vocabulary_series), 256))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def build_one_embedding_model(self):
        model = Sequential()
        model.add(Embedding(len(self.vocabulary_series), 256, input_length=self.maxlen))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def open_fire(self):
        x_train, y_train = self.pre_process()
        # model = self.build_lstm_data()
        model = self.build_one_embedding_model()
        train_num = 15000
        model.fit(x_train[:train_num], y_train[:train_num], batch_size=self.batch_size, epochs=5)
        score = model.evaluate(x_train[train_num:], y_train[train_num:], batch_size=self.batch_size)

        print(model.metrics_names)
        print(score)


if __name__ == '__main__':
    sense_analysis_app = Sentiment_analysis(pos_path='../data/pos.xls',
                                             neg_path='../data/pos.xls')
    sense_analysis_app.open_fire()
