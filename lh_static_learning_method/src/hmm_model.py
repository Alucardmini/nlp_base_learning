#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: hmm_model.py
@time: 12/5/18 11:47 AM
"""
from collections import Counter

hmm_model = {i: Counter() for i in 'smbe'}
print(hmm_model)

with open('../data/dict.txt', 'r')as f:

    lines = f.readlines()
    for line in lines:
        line_list = line.split(' ')
        if len(line_list) > 0:
            word = line_list[0]
            if len(word) == 1:
                # hmm_model[]
                pass



