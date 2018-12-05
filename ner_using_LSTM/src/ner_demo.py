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
from tqdm import tqdm
import numpy as np
import pandas as pd
def main():

    a = [1, 2, 3]
    b = [4, 5, 6]

    a = [[1], [2], [3]]
    b = [[4], [5], [6]]

    c = np.vstack((a, b))
    print(c)



if __name__ == "__main__":
    main()
