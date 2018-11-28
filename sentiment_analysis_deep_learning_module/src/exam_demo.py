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

if __name__ == '__main__':
    content = '你好呀李银河， 李银河呀李银河'
    # print(list(content))
    # abc = pd.Series(list(content))
    # print(abc)
    # abc = abc.value_counts()
    # print(abc)

    abc = pd.Series(list(content)).value_counts()
    abc = abc[abc > 2]
    print(abc)
    # 贴序号
    abc[:] = list(range(1, len(abc) + 1))
    print(abc)
    abc[''] = 0
    word_set = set(abc.index)

    def doc2num(s, maxlen):
        s = [i for i in s if i in word_set]
        s = s[:maxlen] + ['']*max(0, maxlen - len(s))
        return list(abc[s])

    print(doc2num(word_set, 20))
