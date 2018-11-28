#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: test_demo.py
@time: 11/28/18 10:53 AM
"""

import pandas as pd
import numpy as np


pos = pd.read_excel('../data/pos.xls', header=None)
neg = pd.read_excel('../data/neg.xls', header=None)
pos['label'] = 1
neg['label'] = 0
all_ = pos.append(neg)

content = ''.join(all_[0])
abc = list(content)
abc = pd.Series(abc).value_counts()
abc = abc[abc > 2]
print(abc)
abc[''] = 0
# print(abc)
word_set = set(abc.index)
print(word_set)

def doc2num(doc, maxlen):
    doc = [i for i in doc if i in word_set]
    doc = doc[:maxlen] + ['']*max(0, maxlen - len(doc))
    return list(abc[doc])

all_['doc2num'] = all_[0].apply(lambda s : doc2num(s, 300))

x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
y = y.reshape((-1, 1))