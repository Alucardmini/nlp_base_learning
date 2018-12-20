#!/usr/bin/python
#coding:utf-8

import tensorflow as tf
import numpy as np

from tensorflow_demo.practical_seq2seq.datasets.twitter import data
import tensorflow_demo.practical_seq2seq.data_utils as data_utils
import tensorflow_demo.practical_seq2seq.seq2seq_wrapper as seq2seq_wrapper
import importlib


metadata, idx_q, idx_a = data.load_data(PATH='datasets/twitter/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024

importlib.reload(seq2seq_wrapper)
model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/twitter/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )
val_batch_gen = data_utils.rand_batch_gen(validX, validY, 256)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, 256)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)
sess = model.train(train_batch_gen, val_batch_gen)
sess = model.restore_last_session()

input_ = test_batch_gen.__next__()[0]
output = model.predict(sess, input_)
print(output.shape)

replies = []
for ii, oi in zip(input_.T, output):
    q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
    decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
    if decoded.count('unk') == 0:
        if decoded not in replies:
            print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
            replies.append(decoded)