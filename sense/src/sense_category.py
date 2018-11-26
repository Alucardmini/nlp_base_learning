# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '11/21/18'

import numpy as np
import pandas as pd
import jieba

def pre_process(csv_path, m):
    wenjian = pd.read_csv(csv_path, delimiter='     xxx     ', encoding='utf-8', \
                          header=None, names=['comment'])  # 导入文本
    wenjian = wenjian['comment'].str.replace('(<.*?>.*?<.*?>)', '').str.replace('(<.*?>)', '') \
        .str.replace('(@.*?[ :])', ' ')  # 替换无用字符
    wenjian = pd.DataFrame({'comment': wenjian[wenjian != '']})
    wenjian.to_csv('out_' + csv_path, header=False, index=False)
    wenjian['mark'] = m  # 样本标记
    return wenjian.reset_index()

neg = pre_process('data_neg.txt', -1)
pos = pre_process('data_pos.txt', 1)

mydata = pd.concat([neg, pos], ignore_index=True)[['comment','mark']] #结果文件

print("---")

negdict = []; posdict = []; nodict = []; plusdict = [];

sl = pd.read_csv('dict/neg.txt', header=None, encoding='utf-8')
for i in range(len(sl[0])):
    negdict.append(sl[0][i])

sl = pd.read_csv('dict/pos.txt', header=None, encoding='utf-8')
for i in range(len(sl[0])):
    posdict.append(sl[0][i])
sl = pd.read_csv('dict/no.txt', header=None, encoding='utf-8')
for i in range(len(sl[0])):
    nodict.append(sl[0][i])
sl = pd.read_csv('dict/plus.txt', header=None, encoding='utf-8')
for i in range(len(sl[0])):
    plusdict.append(sl[0][i])
