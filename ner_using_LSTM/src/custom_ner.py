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

def data_generator(data, targets, batch_size):
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size * i, min(len((data), batch_size*(i+1))))] for i in range(len(data)/batch_size + 1)]
    while True:
        for i in batches:
            xx, yy = np.array(map(gen_matrix, data[i])), np.array(map(gen_target, targets[i]))
            yield (xx, yy)

batch_size = 1024
history = model.fit_generator(data_generator(json_data['words'], json_data['label'], batch_size), samples_per_epoch=len(json_data), nb_epoch=200)
model.save_weights('../data/words_seq2seq_final_1.model')

def predict_data(data, batch_size):
    batches = [range(batch_size*i, min(len(data), batch_size*(i+1))) for i in range(len(data)/batch_size+1)]
    p = model.predict(np.array(map(gen_matrix, data[batches[0]])), verbose=1)
    for i in batches[1:]:
        print(min(i), 'done')
        p = np.vstack((p, model.predict(np.array(map(gen_matrix, data[i])), verbose=1)))
    return p
json_data['predict'] = list(predict_data(json_data['words'], batch_size))
test_data['predict'] = list(predict_data(test_data['words'], batch_size))

'''
动态规划部分：
1、zy是转移矩阵，用了对数概率；概率的数值是大概估计的，事实上，这个数值的精确意义不是很大。
2、viterbi是动态规划算法。
'''
zy = {'00':0.15,
      '01':0.15,
      '02':0.7,
      '10':1.0,
      '23':0.5,
      '24':0.5,
      '33':0.5,
      '34':0.5,
      '40':1.0
     }
zy = {i: np.log(zy[i]) for i in zy.keys()}

def viterbi(nodes):
    paths = nodes[0]
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
    for i in nodes[l].keys():
        nows = {}
        for j in paths_.keys():
            if j[-1]+i in zy.keys():
                nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
        k = np.argmax(nows.values())
        paths[nows.keys()[k]] = nows.values()[k]
    return paths.keys()[np.argmax(paths.values())]


def predict(i):
    nodes = [dict(zip(['0','1','2','3','4'], k)) for k in np.log(dd['predict'][i][:len(dd['words'][i])])]