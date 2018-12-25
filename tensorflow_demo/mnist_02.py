# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '12/24/18'

import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
imgWidth = x_train.shape[1]

train_size = -1
test_size = 10
batch_size = 50
np_classes = 10
nb_epochs = 10001

x_train = x_train[0:train_size]
y_train = y_train[0:train_size]

x_train = x_train.reshape([-1, imgWidth*imgWidth])
x_test = x_test.reshape([-1, imgWidth*imgWidth])
x_train = np.array(x_train).astype(np.float32) / 255.
x_test = np.array(x_test).astype(np.float32) / 255.

y_train = np_utils.to_categorical(y_train, np_classes)
y_test = np_utils.to_categorical(y_test, np_classes)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])


def add_layer(inputs, in_size, out_size, activation_function=None):

    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.random_normal([1, out_size]))
    wx_plus_b = tf.add(tf.matmul(inputs, Weight), bias)

    if activation_function is None:
        return wx_plus_b
    else:
        return activation_function(wx_plus_b)

# h1 = add_layer(x, 784, 128, activation_function=tf.nn.relu)
# y_pred = add_layer(h1, 128, 10, activation_function=tf.nn.softmax)

y_pred = add_layer(x, 784, 10, activation_function=tf.nn.softmax)

# W_1 = tf.Variable(tf.zeros([784, 64]), trainable=True)
# b_1 = tf.Variable(tf.zeros([64]), trainable=True)
# # y_1 = tf.matmul(x, W_1) + b_1
# W = tf.Variable(tf.zeros([64, 10]))
# b = tf.Variable(tf.zeros([10]))
# y_pred = tf.nn.softmax(tf.matmul(tf.matmul(x, W_1) + b_1, W) + b)

# cross_entropy = -tf.reduce_sum(y*tf.log(y_pred))

cross_entropy = -tf.reduce_sum(y*tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(nb_epochs):
        for batch_index in range(1, int(len(x_train)/batch_size + 1)):
            # print(batch_index)
            max_index = min(batch_index * batch_size, len(x_train))
            batch_x = x_train[(batch_index-1)* batch_size: max_index]
            batch_y = y_train[(batch_index-1)* batch_size: max_index]
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})

        if i % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: x_test, y: y_test}))



