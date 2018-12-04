#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: Main.py
@time: 12/3/18 6:00 PM
"""

import numpy as np
import pandas as pd
import jieba
from tqdm import tqdm
import re
import logging
import gensim

d = pd.read_json('../data/data.json')
d.index = range(len(d))

word_size = 128
maxlen = 80

not_cuts = re.compile(u'([\da-zA-Z \.]+)|《(.*?)》|“(.{1,10})”')
re_replace = re.compile(u'[^\u4e00-\u9fa50-9a-zA-Z《》\(\)（）“”·\.]')

def mycut(s):
    result = []
    j = 0
    s = re_replace.sub(' ', s)
    for i in not_cuts.finditer(s):
        result.extend(jieba.lcut(s[j:i.start()], HMM=False))
        if s[i.start()] in [u'《', u'“']:
            result.extend([s[i.start()], s[i.start()+1:i.end()-1], s[i.end()-1]])
        else:
            result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(jieba.lcut(s[j:], HMM=False))
    return result

# print(d['content'])
d['words'] = d['content'].apply(mycut)
print(d['words'])

def label(k): #将输出结果转换为标签序列
    s = d['words'][k]
    r = ['0']*len(s)
    for i in range(len(s)):
        for j in d['core_entity'][k]:
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

d['label'] = map(label, tqdm(iter(d.index))) #输出tags
# print(d['label'])


#随机打乱数据
idx = np.arange(len(d))
d.index = idx
np.random.shuffle(idx)
d = d.loc[idx]
d.index = range(len(d))
print(d)

#读入测试数据并进行分词
# dd = open('../data/opendata_20w').read().split('\n')
# dd = pd.DataFrame([dd]).T
# dd.columns = ['content']
# dd = dd[:-1]
# print(u'测试语料分词中......')
# dd['words'] = dd['content'].apply(mycut)
# print(dd['words'])

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# word2vec = gensim.models.Word2Vec(dd['words'].append(d['words']),
#                                   min_count=1,
#                                   size=word_size,
#                                   workers=20,
#                                   iter=20,
#                                   window=8,
#                                   negative=8,
#                                   sg=1)
#
# word2vec.save('../data/word2vec_words_final.model')
# word2vec.init_sims(replace=True) #预先归一化，使得词向量不受尺度影响

# word2vec = gensim.models.Word2Vec.load('../data/word2vec_words_final.model')
#
# from keras.layers import Dense, LSTM, Lambda, TimeDistributed, Input, Masking, Bidirectional
# from keras.models import Model
# from keras.utils import np_utils
# from keras.regularizers import l1
#
# sequence = Input(shape=(maxlen, word_size))
# mask = Masking(mask_value=0.)(sequence)
#
# blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(mask)
# blstm = Bidirectional(LSTM(32, return_sequences=True), merge_mode='sum')(blstm)
# output = TimeDistributed(Dense(5, activation='softmax', activity_regularizer=l1(0.01)))(blstm)
# model = Model(input=sequence, output=output)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# gen_matrix = lambda z: np.vstack((word2vec[z[:maxlen]], np.zeros((maxlen-len(z[:maxlen]), word_size))))
# gen_target = lambda z: np_utils.to_categorical(np.array(z[:maxlen] + [0]*(maxlen-len(z[:maxlen]))), 5)
#
#
# def data_generator(data, targets, batch_size):
#     idx = np.arange(len(data))
#     np.random.shuffle(idx)
#     batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)/batch_size+1)]
#
#     while True:
#         for i in batches:
#             xx, yy = np.array(map(gen_matrix, data[i])), np.array(map(gen_target, targets[i]))
#             yield xx, yy
#
# batch_size = 1024
# history = model.fit_generator(data_generator(d['words'], d['label'], batch_size), samples_per_epoch=len(d), nb_epoch=200)
# model.save_weights('../data/words_seq2seq_final_1.model')
#
# def predict_data(data, batch_size):
#     batches = [range(batch_size*i, min(len(data), batch_size*(i+1))) for i in range(len(data)/batch_size+1)]
#     p = model.predict(np.array(map(gen_matrix, data[batches[0]])), verbose=1)
#
#     for i in batches[1:]:
#         print(min(i), 'done')
#         p = np.vstack((p, model.predict(np.array(map(gen_matrix, data[i])), verbose=1)))
#     return p
#
# d['predict'] = list(predict_data(d['word'], batch_size))
# dd['predict'] = list(predict_data(dd['words'], batch_size))
#
#
# '''
# 动态规划部分：
# 1、zy是转移矩阵，用了对数概率；概率的数值是大概估计的，事实上，这个数值的精确意义不是很大。
# 2、viterbi是动态规划算法。
# '''
# zy = {'00':0.15,
#       '01':0.15,
#       '02':0.7,
#       '10':1.0,
#       '23':0.5,
#       '24':0.5,
#       '33':0.5,
#       '34':0.5,
#       '40':1.0
#      }
#
# zy = {i: np.log(zy[i]) for i in zy.keys()}
#
# def viterbi(nodes):
#     paths = nodes[0]
#     for l in range(1,len(nodes)):
#         paths_ = paths.copy()
#         paths = {}
#         for i in nodes[l].keys():
#             nows = {}
#             for j in paths_.keys():
#                 if j[-1]+i in zy.keys():
#                     nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
#             k = np.argmax(nows.values())
#             paths[nows.keys()[k]] = nows.values()[k]
#     return paths.keys()[np.argmax(paths.values())]
#
# def predict(i):
#     nodes = [dict(zip(['0','1','2','3','4'], k)) for k in np.log(dd['predict'][i][:len(dd['words'][i])])]
#     r = viterbi(nodes)
#     result = []
#     words = dd['words'][i]
#     for j in re.finditer('2.*?4|1', r):
#         result.append((''.join(words[j.start():j.end()]), np.mean([nodes[k][r[k]] for k in range(j.start(),j.end())])))
#     if result:
#         result = pd.DataFrame(result)
#         return [result[0][result[1].argmax()]]
#     else:
#         return result
#
# dd['core_entity'] = map(predict, tqdm(iter(dd.index), desc=u'第一次预测'))
#
# '''
# 导出提交的JSON格式
# '''
# gen = lambda i:'[{"content": "'+dd.iloc[i]['content']+'", "core_entity": ["'+''.join(dd.iloc[i]['core_entity_2'])+'"]}]'
#
# ssss = map(gen, tqdm(range(len(dd))))
# result='\n'.join(ssss)
# import codecs
# f=codecs.open('../data/result2.txt', 'w', encoding='utf-8')
# f.write(result)
# f.close()
# import os
# os.system('rm result1.zip')
# os.system('zip result1.zip result1.txt')
#
# print(u'正在进行第一次迁移学习......')
#
# def label(k): #将输出结果转换为标签序列
#     s = dd['words'][k]
#     r = ['0']*len(s)
#     for i in range(len(s)):
#         for j in dd['core_entity'][k]:
#             if s[i] in j:
#                 r[i] = '1'
#                 break
#     s = ''.join(r)
#     r = [0]*len(s)
#     for i in re.finditer('1+', s):
#         if i.end() - i.start() > 1:
#             r[i.start()] = 2
#             r[i.end()-1] = 4
#             for j in range(i.start()+1, i.end()-1):
#                 r[j] = 3
#         else:
#             r[i.start()] = 1
#     return r
#
# dd['label'] = map(label, tqdm(iter(dd.index))) #输出tags
#
#
# '''
# 将测试集和训练集一起放到模型中训练，
# 其中测试集的样本权重设置为1，训练集为10
# '''
# w = np.array([1]*len(dd) + [10]*len(d))
# def data_generator(data, targets, batch_size):
#     idx = np.arange(len(data))
#     np.random.shuffle(idx)
#     batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)/batch_size+1)]
#     while True:
#         for i in batches:
#             xx, yy = np.array(map(gen_matrix, data[i])), np.array(map(gen_target, targets[i]))
#             yield (xx, yy, w[i])
#
# history = model.fit_generator(data_generator(
#                                     dd[['words']].append(d[['words']], ignore_index=True)['words'],
#                                     dd[['label']].append(d[['label']], ignore_index=True)['label'],
#                                     batch_size),
#                               samples_per_epoch=len(dd)+len(d),
#                               nb_epoch=20)
#
# model.save_weights('words_seq2seq_final_2.model')
# d['predict'] = list(predict_data(d['words'], batch_size))
# dd['predict'] = list(predict_data(dd['words'], batch_size))
# dd['core_entity_2'] = map(predict, tqdm(iter(dd.index), desc=u'第一次迁移学习预测'))
#
# '''
# 导出提交的JSON格式
# '''
# gen = lambda i:'[{"content": "'+dd.iloc[i]['content']+'", "core_entity": ["'+''.join(dd.iloc[i]['core_entity_2'])+'"]}]'
# ssss = map(gen, tqdm(range(len(dd))))
# result='\n'.join(ssss)
# import codecs
# f=codecs.open('result2.txt', 'w', encoding='utf-8')
# f.write(result)
# f.close()
# import os
# os.system('rm result2.zip')
# os.system('zip result2.zip result2.txt')
#
# print(u'正在进行第二次迁移学习......')


