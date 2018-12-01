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

min_count = 10  # 录取词语最小出现次数
min_support = 30  # 录取词语最低支持度，1代表着随机组合
min_s = 3  # 录取词语最低信息熵，越大说明越有可能独立成词
max_sep = 4  # 候选词语的最大字数

# 为了方便调用，自定义了一个正则表达式的词典
myre = {2: '(..)', 3: '(...)', 4: '(....)', 5: '(.....)', 6: '(......)', 7: '(.......)'}

class Findnewwords(object):

    def __init__(self, data_path, min_count=10, max_step=4, min_s=3, min_support=30):
        self.max_step = max_step
        self.min_count = min_count
        self.min_s = min_s
        self.min_support = min_support

    @staticmethod
    def load_data_from_path(data_path):
        f = open(data_path)
        content = f.read()

        # 定义要去掉的标点字
        drop_dict = [u'，', u'\n', u'。', u'、', u'：', u'(', u')', u'[', u']', u'.', u',', u' ', u'\u3000', u'”', u'“',
                     u'？', u'?',
                     u'！', u'‘', u'’', u'…']

        for i in drop_dict:
            content = content.replace(i, '')
        data = []
        data.append(pd.Series(list(content)).value_counts())
        return data, content

    @staticmethod
    def generate_n_gram_word(content, input_data, max_count):
        vocab_sum = input_data[0].sum()
        n_gram_result = []

        for i in range(2, max_count+1):
            print("正在生成%d字词" % i)
            input_data.append([])
            for j in range(i):
                input_data[i - 1] = input_data[i - 1] + re.findall(myre[i], content[j:])

            input_data[i - 1] = pd.Series(input_data[i-1]).value_counts()
            # 筛选次数
            input_data[i - 1] = input_data[i - 1][input_data[i - 1] > min_count]

            tt = input_data[i - 1][:]
            for k in range(i - 1):
                filter_mutual_info_res = np.array(list(map(lambda word: vocab_sum * input_data[i-1][word] / \
                                                      (input_data[i - 2 - k][word[:i - 1 - k]] * input_data[k][word[i - 1 - k:]])
                                             ,tt.index))) > min_support
                tt = tt[filter_mutual_info_res]
            n_gram_result.append(tt)
        return n_gram_result

    @staticmethod
    def filter_entropy(content, input_data, max_step):
        for i in range(2, max_step + 1):
            print(u'正在进行 %d 字词的最大熵筛选 (%s) ... '%(i, input_data[i - 2].sum()))
            pp = [] # 左右邻结果
            for j in range(i + 2):
                pp += re.findall('(.)%s(.)' % myre[i], content[j:])
            pp = pd.DataFrame(pp).set_index(1).sort_index()  # 先排序，这个很重要，可以加快检索速度


if __name__ == '__main__':
    path = '../data/tlbb.txt'

    # # 预处理
    # data, content = Findnewwords.load_data_from_path(path)
    # # 生成n-gram 词并经过互信息过滤
    # data = Findnewwords.generate_n_gram_word(content, data, 4)

    content = '挖掘新词的传统方法是，先对文本进行分词，然后猜测未能成功匹配的剩余片段就是新词。这似乎陷入了一个怪圈：分词的准确性本身就依赖于词库的完整性，如果词库中根本没有新词，我们又怎么能信任分词结果呢？此时，一种大胆的想法是，首先不依赖于任何已有的词库，仅仅根据词的共同特征，将一段大规模语料中可能成词的文本片段全部提取出来，不管它是新词还是旧词。然后，再把所有抽出来的词和已有词库进行比较，不就能找出新词了吗？有了抽词算法后，我们还能以词为单位做更多有趣的数据挖掘工作。这里，我所选用的语料是人人网 2011 年 12 月前半个月部分用户的状态。非常感谢人人网提供这份极具价值的网络语料。'

    for i in range(2, 5):
        pp = []
        for j in range(i + 2):
            pp += re.findall('(.)%s(.)'%myre[i], content[j:])
        pp = pd.DataFrame(pp)
        print(pp)

        pp = pp.set_index(1)
        print('***')
        print(pp)
        pp = pp.sort_index()
        print('===')
        print(pp)




