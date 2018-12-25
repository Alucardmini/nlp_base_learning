#!/usr/bin/python
#coding:utf-8

import tensorflow as tf
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_size = 200

batch_size = 100
np_epoches = 101

x_train = x_train[0:train_size]
y_train = y_train[0:train_size]
x_test = x_test[0:train_size]
y_test = y_test[0:train_size]

print(x_train.shape)

width = x_train.shape[1]

x_train = x_train.reshape(-1, width*width)
x_test = x_test.reshape(-1, width*width)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

nb_classes = 10
y_test = np_utils.to_categorical(y_test, nb_classes)
y_train = np_utils.to_categorical(y_train, nb_classes)


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_pred = tf.nn.softmax(tf.matmul(x, W) + b)
# cross_entropy = -tf.reduce_sum(y*tf.log(y_pred))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred)))

cross_entropy = -tf.reduce_sum(y*tf.log(y_pred))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# # 创建一个隐藏层，输入数据：x_data， 输出10个神经元，激励函数使用softmax
# prediction = tf.layers.dense(x_train, 10, tf.nn.softmax)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(np_epoches):


        sess.run(train_step, feed_dict={x: x_train, y: y_train})
        if i % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: x_test, y: y_test}))








