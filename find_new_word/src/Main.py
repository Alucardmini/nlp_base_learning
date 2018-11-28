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

class Findnewwords(object):

    def __init__(self, data_path, min_count=10, max_step=4):
        self.max_step = max_step
        self.min_count = min_count
        self.src_content_list = self.loaddatafromPath(data_path)

    def loaddatafromPath(self, data_path):
        with open(data_path, 'r')as f:
            src_content = ''.join(f.readlines())
        src_content_list = list(src_content)

        src_count_list = pd.Series(src_content_list).value_counts()
        return src_content_list


if __name__ == '__main__':
    pass