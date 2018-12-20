#!/usr/bin/python
#coding:utf-8

import tensorflow as tf

weight = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name='weights')
biases = tf.Variable(tf.zeros([200]), name='biases')