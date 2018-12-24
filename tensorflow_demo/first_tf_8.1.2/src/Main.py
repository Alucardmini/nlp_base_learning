# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '12/15/18'

import tensorflow as tf
import numpy as np

x_data = np.linspace(-1, 1, 300)
x_data = x_data[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


def add_layer(inputs, in_size, out_size, activation_function=None):

    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.random_normal([1, out_size]))
    wx_plus_b = tf.add(tf.matmul(inputs, Weight), bias)

    if activation_function is None:
        return wx_plus_b
    else:
        return activation_function(wx_plus_b)


h1 = add_layer(xs, 1, 20, activation_function=tf.nn.relu)
prediction = add_layer(h1, 20, 1)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step  =tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))




