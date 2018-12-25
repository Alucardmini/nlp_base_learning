#!/usr/bin/python
#coding:utf-8

from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Model
from keras.layers import *

maxfeature = 200
batch_size = 32
maxlen = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=maxfeature)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

s_inputs = Input(shape=(None, ), dtype='int32')