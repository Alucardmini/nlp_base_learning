#!/usr/bin/python
#coding:utf-8

import numpy as np

import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
import chinese_ner.rnncell as rnn


class Model(object):
    def __init__(self, config):
        self.config = config
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")
        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """

        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :param name: name of this layer
        :return: [1, num_steps, embedding_size]
        """
        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                name="char_embedding",
                shape=[self.num_chars, self.char_dim],
                initializer=self.initializer
            )
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer
                    )
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """

        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :param lstm_dim: [batch_size, num_steps, 2*lstm_dim]
        :param lengths:
        :param name:
        :return:
        """

        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs:
        :param name:
        :return:
        """




















