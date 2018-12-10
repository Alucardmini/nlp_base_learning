#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: Main.py
@time: 12/10/18 2:39 PM
"""


from word2vec.word2vec_model import *

if __name__ == '__main__':

    texts = ['']
    wv = Word2Vec(texts, model='cbow', nb_negative=16, shared_softmax=True, epochs=2)  # 建立并训练模型

