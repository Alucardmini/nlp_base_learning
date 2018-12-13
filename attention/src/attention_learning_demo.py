#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: attention_learning_demo.py
@time: 12/13/18 11:17 AM
"""

import keras.backend as K
import numpy as np
import tensorflow as tf

def print_tensor(tf_input_data):

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        print(tf_input_data.get_shape())
        print(tf_input_data.eval())

# data = np.array(range(120))
# data = data.reshape((-1, 4, 5))
# print(data)
#
# tf_data = tf.Variable(data, dtype=tf.float32)
#
# # 转换维度
# # tf_data = K.permute_dimensions(tf_data, (2, 1, 0))
#
#
# data_weight = np.array(range(5 * 3)).reshape(data.shape[-1], 3)
# # print(data_weight)
#
# tf_data_weight = tf.Variable(data_weight, dtype=tf.float32)
#
# print_tensor(tf_data_weight)
# # data.shape() = (x, ... ,y, z) weight.shape = (z, k) => output.shape = (x,..., y, k)
# # res = K.dot(tf_data, tf_data_weight)
#
# # print('- res -')
# #
# # print_tensor(res)
# #
# # print('- soft max -')
# #
# # print_tensor(K.softmax(res))
#
# tf_data_2 = tf_data
#
# res = K.batch_dot(tf_data, tf_data_2, axes=[1, 1])
# print('-- batch_dot --')
# print_tensor(res)
#
#
# res = K.batch_dot(tf_data, tf_data_2, axes=[2, 2])
# print('-- batch_dot _2 --')
# print_tensor(res)

# x = tf.Variable(np.array(range(24)).reshape(-1, 2, 3), dtype=tf.float32)
# y = tf.Variable(np.array(range(48)).reshape(-1, 3, 4), dtype=tf.float32)
# print_tensor(x)
# print_tensor(y)
# # res = K.batch_dot(x, y, [0, 0])
# # print(' -- 0 --')
# # print_tensor(res)
# # res = K.batch_dot(x, y, [1, 1])
# # print(' -- 1 --')
# # print_tensor(res)
# # res = K.batch_dot(x, y, [2, 2])
# # print(' -- 3 --')
# # print_tensor(res)
# res = K.batch_dot(x, y, axes=[2, 1])
# print(res)

x = tf.Variable(np.array(range(6)).reshape(2, 3), dtype=tf.float32)
y = tf.Variable(np.array(range(3)).reshape(3, 1), dtype=tf.float32)
print_tensor(y)
res = K.batch_dot(x, y, axes=[1, 0])
print_tensor(res)

