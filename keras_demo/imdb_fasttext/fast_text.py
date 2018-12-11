#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: fast_text.py
@time: 12/11/18 5:22 PM
"""

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb


ngram_range = 1
max_features = 20000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 5


def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

def calc_sum(*number):
    res = 0
    for _ in number:
        res += _
    return res

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        max_features = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
        print('Average train sequence length: {}'.format(
            np.mean(list(map(len, x_train)), dtype=int)))
        print('Average test sequence length: {}'.format(
            np.mean(list(map(len, x_test)), dtype=int)))

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    model = Sequential()

    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))


