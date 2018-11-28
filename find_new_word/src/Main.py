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

class Findnewwords(object):

    def __init__(self, data_path, min_count=10, max_step=4):
        self.max_step = max_step
        self.min_count = min_count
        self.src_content_list = self.load_data_from_path(data_path)

    @staticmethod
    def load_data_from_path(data_path):
        src_content = open(data_path, 'r').read()

        with open('../data/chineseStopWords.txt', 'r', encoding='utf-8')as f:
            lines = f.readlines()
            stop_list = [x.strip() for x in lines]
        src_content_list = [x.strip() for x in list(src_content) if x not in stop_list]
        data = pd.Series(src_content_list).value_counts()
        return data

    def pre_process(self):
        pass


if __name__ == '__main__':
    print(Findnewwords.load_data_from_path('../data/tlbb.txt'))