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

    pd_data = pd.DataFrame(np.random.rand(7, 5), index=list('ABCDEFG'))
    print(pd_data)

    vocab_index = np.array(list('ABCDEFG'))
    np.random.shuffle(vocab_index)
    print(vocab_index)
    data = pd_data.loc[vocab_index]
    data.index = list('HIJKLMN')
    data.index = list('ABCDEFG')
    print(data)





if __name__ == "__main__":
    main()
