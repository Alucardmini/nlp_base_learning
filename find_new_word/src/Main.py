#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: Main.py
@time: 11/28/18 3:59 PM
"""
import pandas as pd
import numpy as np
import jieba
import codecs
from numpy import log,min
import re
class Findnewwords(object):

    def __init__(self, data_path, min_count=10, max_step=4, min_s=3, min_support=30):
        self.max_step = max_step
        self.min_count = min_count
        self.min_s = min_s
        self.min_support = min_support
        self.src_content_list = self.load_data_from_path(data_path)

    def load_data_from_path(self, data_path):
        src_content = open(data_path, 'r').read()
        src_content = src_content.replace(u'\u3000', u'').replace(u'：', u'')
        data = []
        with open('../data/chineseStopWords.txt', 'r', encoding='utf-8')as f:
            lines = f.readlines()
            stop_list = [x.strip() for x in lines]
        src_content_list = [x.strip() for x in list(src_content) if x not in stop_list]
        data.append(pd.Series(src_content_list).value_counts())

        vocabulary_sum = data[0].sum()

        rt = []
        # 为了方便调用，自定义了一个正则表达式的词典
        myre = {2: '(..)', 3: '(...)', 4: '(....)', 5: '(.....)', 6: '(......)', 7: '(.......)'}

        for m in range(2, self.max_step + 1):
            print(u'生成 %s 字词 ...' % m)
            data.append([])
            for i in range(m):
                data[m - 1] = data[m-1] + re.findall(myre[m], src_content[i:])

            data[m-1] = pd.Series(data[m-1]).value_counts()
            data[m-1] = data[m-1][data[m-1] > self.min_count]

            process_data = data[m-1][:]
            for k in range(m - 1):

                qq = np.array(list(
                    map(lambda ms: vocabulary_sum * data[m - 1][ms] / data[m - 2 - k][ms[:m - 1 - k]] / data[k][ms[m - 1 - k:]],
                        process_data.index))) > self.min_support  # 最小支持度筛选。

                process_data = process_data[qq]
            rt.append(process_data.index)

        for i in range(2, self.max_step + 1):
            print(u'处理 %d 字词的最大熵筛选 (%s)...' % (i, rt[i - 1]))
            lr_info = []
            for j in range(i + 2):
                lr_info = lr_info + re.findall('(.)%s(.)' % myre[i], src_content[j:])

            lr_info = pd.DataFrame(lr_info).set_index(1).sort_index()
            index = np.sort(np.intersect1d(rt[i-2], lr_info.index))
            index = index[np.array(list(map(lambda s: self.cal_S(pd.Series(lr_info[0][s]).value_counts()), index))) > self.min_s]
            rt[i - 2] = index[np.array(list(map(lambda s: self.cal_S(pd.Series(lr_info[2][s]).value_counts()), index))) > self.min_s]

        for i in range(len(rt)):
            data[i + 1] = data[i + 1][rt[i]]
            data[i + 1].sort(ascending=False)

        pd.DataFrame(pd.concat(data[1:])).to_csv('../data/result.txt', header=False)

        return data

    def cal_S(self, sl):  # 信息熵计算函数
        return -((sl / sl.sum()).apply(log) * sl / sl.sum()).sum()

    def pre_process(self):
        pass


if __name__ == '__main__':
    path = '../data/tlbb.txt'
    app = Findnewwords(path)
    rt = app.load_data_from_path(path)

