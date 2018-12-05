#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: ner_demo.py
@time: 12/4/18 3:08 PM
"""
import os
import time
import sys
# from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter
def main():

    s = {'b': Counter()}

    print(s)
    s['b']['a'] += 20
    s['b']['d'] = 30
    print(s)
    print(max(s['b'].values()))
    # print(s['b'].fromkeys( max(s['b'].values())))

    print(get_key(s['b'], max(s['b'].values())))

    # print( s['b'].values().index(max(s['b'].values())))
    # student = {'小萌': '1001', '小智': '1002', '小强': '1003', '小明': '1004'}
    #
    # print( list (student.keys()) [list (student.values()).index ('1004')])


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]

if __name__ == "__main__":
    main()
