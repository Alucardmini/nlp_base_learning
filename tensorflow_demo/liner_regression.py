#!/usr/bin/python
#coding:utf-8

import tensorflow as tf
import numpy as np

inputs_data = np.linspace(-2, 2, 100, dtype=float)
target_value = 2.0*inputs_data + 0.4

print(inputs_data)
print(target_value)

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

y = W*inputs_data + b

loss = tf.reduce_mean(tf.square(y - target_value))
optimizor = tf.train.GradientDescentOptimizer(0.5)
train = optimizor.minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(201):
        sess.run(train)
        if i % 20 == 0:
            print (i, sess.run(W), sess.run(b),  tf.reduce_mean(tf.square(y - target_value)))

