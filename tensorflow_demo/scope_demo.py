#!/usr/bin/python
#coding:utf-8

import tensorflow as tf
import numpy as np

with tf.variable_scope("V1"):
    a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a2 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')

with tf.variable_scope("v2"):
    a3 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a4 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(a1.name)
    print(a2.name)
    print(a3.name)
    print(a4.name)

print(' === ')

with tf.name_scope("V1"):
    a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a2 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')

with tf.name_scope("v2"):
    a3 = tf.get_variable(name='a3', shape=[1], initializer=tf.constant_initializer(1))
    a4 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a4')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(a1.name)
    print(a2.name)
    print(a3.name)
    print(a4.name)