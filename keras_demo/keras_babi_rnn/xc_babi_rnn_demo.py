#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: xc_babi_rnn_demo.py
@time: 12/11/18 1:59 PM
"""

from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np
from keras.utils.data_utils import  get_file
from keras.layers import Dense, LSTM, Embedding, Input, concatenate
from keras.preprocessing.sequence import pad_sequences


EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 20

def tokenize(sent):
    return [x.strip() for x in re.split(r'(\W)?', sent) if x.strip()]

def parse_stories(lines, only_supporting=False):
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if 1 == nid:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            if only_supporting:
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')

        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    # 没看明白这步操作
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, a) for story, q, a in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return (pad_sequences(xs, maxlen=story_maxlen),
            pad_sequences(xqs, maxlen=query_maxlen), np.array(ys))

if __name__ == '__main__':
    path = get_file('babi-tasks-v1-2.tar.gz',
                    origin='https://s3.amazonaws.com/text-datasets/'
                           'babi_tasks_1-20_v1-2.tar.gz')

    challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
    with tarfile.open(path) as tar:
        train = get_stories(tar.extractfile(challenge.format('train')))
        test = get_stories(tar.extractfile(challenge.format('test')))

    vocab = set()
    for story, q, answer in train + test:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)

    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

    print('build model')
    sentence = Input(shape=(story_maxlen, ), dtype='int32')
    encode_sentence = Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
    lstm_sentence = LSTM(SENT_HIDDEN_SIZE)(encode_sentence)

    question = Input(shape=(query_maxlen, ), dtype='int32')
    encode_question = Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
    lstm_question = LSTM(SENT_HIDDEN_SIZE)(encode_question)

    merged = concatenate([lstm_question, lstm_sentence])
    preds = Dense(vocab_size, activation='softmax')(merged)
    from keras.models import Model
    model = Model([sentence, question], preds)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit([x, xq], y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.05)

    print('Evaluation ... ')
    loss, acc = model.evaluate([tx, txq], ty,
                               batch_size=BATCH_SIZE)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))