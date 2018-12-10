#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@version: 1.0
@Author by Lizhongxun ubuntu
@Mail: lizhongxun@bjgoodwill.com
@File: word2vec.py
@time: 2018-12-06 16:02
"""
__author__ = 'lizhongxun'
import os
import re
import math
import random
from collections import Counter, deque
import tensorflow as tf
import numpy as np
from utils import build_dataset

# 生成一个batch的训练数据，用于训练
def generate_batch(data, batch_size, num_skips, skip_window, end_id):
    '''
    生成一个batch的训练数据，用于训练
    :param data: build_dataset的结果
    :param batch_size: batch大小
    :param num_skips: 对每个单词生成的样本数
    :param skip_window: 滑窗大小
    :param end_id: 结束标志'END'的id
    :return:
    '''
    global data_index
    # batch_size必须是num_skips的整数倍,保证每个batch包含了一个词汇对应的所有样本
    assert batch_size % num_skips == 0
    # 样本数小于2倍的滑窗大小
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # 定义span为对某个单词创建相关样本时会使用到的单词数量,包括目标单词本身和它前后的单词 [ skip_window target skip_window ]
    span = 2 * skip_window + 1
    # 创建一个最大容量为span的deque(双向队列,在对deque使用append方法添加数据时,只会保留最后插入的span变量)
    buffer = deque(maxlen=span)    # 缓冲器
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):  # //除法取整
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]  # 用于过滤已使用的单词
        for j in range(num_skips):  # 对一个单词生成num_skips个样本
            while target in targets_to_avoid:  # 随机出一个满足整数(顺序不定但不重复)
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)  # 单词已经使用了,过滤掉
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])  # 读入下一个单词,会自动抛弃一个单词
        data_index = (data_index + 1) % len(data)
        # 保证'END'除了在最后个不在别的地方出现
        while True:
            try:
                buffer.index(end_id, 0, len(buffer)-1)
            except ValueError:
                break
            else:
                buffer.append(data[data_index])  # 读入下一个单词,会自动抛弃一个单词
                data_index = (data_index + 1) % len(data)
    return batch, labels

# 训练时batch_size为128
batch_size = 128
# embedding_size即将单词转为稠密向量的维度,一般取50~1000这个范围内的值
embedding_size = 128
# 窗口大小，左或者右多少词
skip_window = 1
# 每一个字产生label的次数
num_skips = 2

# 验证的单词数
valid_size = 16
# 验证单词只从频率最高的100个字中抽取
valid_window = 100
# 从valid_window中随机取valid_size个数据, replace有无放回,p是抽取概率,如果没有p,就是均匀的取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# 训练时用来做负样本的噪声单词的数量
num_sampled = 64

data, count, dictionary, reverse_dictionary, vocabulary_size = build_dataset()
data_index = 0
batch, labels = generate_batch(data, batch_size=30, num_skips=2, skip_window=1, end_id=dictionary['END'])

graph = tf.Graph()
with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # 计算NCE loss(计算学习出的词向量embedding在训练数据上的loss,并使用tf.reduce_mean汇总)
    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each # time we evaluate the loss.
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))
    # 使用SGD优化器,学习率为1
    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    # 计算嵌入向量embeddings的L2范数
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    # 将embeddings除以其L2范数得到标准化后的normalized_embeddings
    normalized_embeddings = embeddings/norm
    # 使用embedding_lookup查询验证单词的嵌入向量,并计算验证单词的嵌入向量与词汇表中所有单词的相似性
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

num_steps = 100001
with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")
    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips, skip_window, end_id=dictionary['END'])
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val= session.run([optimizer, loss], feed_dict=feed_dict)

        average_loss += loss_val
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
        # 每2000次计算一下平均的loss并显示
        # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # 每10000次计算一次验证单词与全部单词的相似度,将最相似的8个单词打印出来
        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8    # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
    np.save('model/word2vec.npy', final_embeddings)

if __name__ == '__main__':
    pass