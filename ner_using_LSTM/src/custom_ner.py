#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: custom_ner.py
@time: 12/4/18 3:24 PM
"""

import pandas as pd
import numpy as np
import re
import jieba
from tqdm import tqdm

from keras.layers import Dense, LSTM, Lambda, TimeDistributed, Input, Masking, Bidirectional
from keras.models import Model
from keras.utils import np_utils
from keras.regularizers import l1 #通过L1正则项，使得输出更加稀疏

word_size = 128 #词向量维度
maxlen = 80 #句子截断长度

json_data = pd.read_json('../data/data.json')
not_cut = re.compile(u'([A-Za-z\d]+)|《(.*)》|"(.{1,9})"')
need_replace = re.compile(u'[^\u4e00-\u9fa50-9a-zA-Z《》\(\)（）“”·\.]')
def custom_cut(s):
    result = []
    j = 0
    for i in not_cut.finditer(s):
        tmp = s[j:i.start()]
        tmp = need_replace.sub(' ', tmp)
        result.extend(jieba.lcut(tmp, HMM=False))
        if s[i.start()] in [u'《', u'“', u'"']:
            result.append(s[i.start()+1:i.end()-1])
        else:
            result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(jieba.lcut(s[j:], HMM=False))
    return result

json_data['words'] = json_data['content'].apply(custom_cut)
# print(json_data['word'])

def label(k): #将输出结果转换为标签序列
    s = json_data['words'][k]
    r = ['0']*len(s)
    for i in range(len(s)):
        for j in json_data['core_entity'][k]:
            if s[i] in j:
                r[i] = '1'
                break
    s = ''.join(r)
    r = [0]*len(s)
    for i in re.finditer('1+', s):
        if i.end() - i.start() > 1:
            r[i.start()] = 2
            r[i.end()-1] = 4
            for j in range(i.start()+1, i.end()-1):
                r[j] = 3
        else:
            r[i.start()] = 1
    return r

json_data['label'] = map(label, tqdm(iter(json_data.index))) #输出tags

idx = np.arange(len(json_data))
json_data.index = idx
json_data.loc[3] #这一行无意义 但是去掉以后没有json_data 没有被打乱
np.random.shuffle(idx)
json_data = json_data.loc[idx]
json_data.reset_index(drop=True, inplace=True)

test_data = open('../data/opendata_20w').read().split('\n')
test_data = pd.DataFrame([test_data]).T
test_data.columns = ['content']
test_data = test_data[:-1]
test_data['words'] = test_data['content'].apply(custom_cut)
# print(test_data)

import gensim
word2vec = gensim.models.Word2Vec.load('../data/word2vec_words_final.model')

sequence = Input(shape=(maxlen, word_size))
mask = Masking(make_value=0.)(sequence)
blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(mask)
blstm = Bidirectional(LSTM(32, return_sequences=True), merge_mode='sum')(blstm)
output = TimeDistributed(Dense(5, activation='softmax', activity_regularizer=l1(0.01)))(blstm)
model = Model(inputs=sequence, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

gen_matrix = lambda z: np.vstack((word2vec[z[:maxlen]], np.zeros((maxlen-len(z[:maxlen]), word_size))))
gen_target = lambda z: np_utils.to_categorical(np.array(z[:maxlen] + [0]*(maxlen-len(z[:maxlen]))), 5)
