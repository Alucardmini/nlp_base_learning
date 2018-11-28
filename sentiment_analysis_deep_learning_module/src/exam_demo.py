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
    abc = pd.Series(list(content)).value_counts()
    abc = abc[abc >= 2]
    # print(abc)
    # print(list(range(len(abc))))
    # abc = list(range(len(abc)))
    # print(abc)
    abc[:] = list(range(len(abc)))
    # print(abc)

    # print(set(abc.index))
    y = np.array(list(range(100)))
    print(y)
    y = y.reshape((-1, 1))  # 调整标签形状
    print(y)

