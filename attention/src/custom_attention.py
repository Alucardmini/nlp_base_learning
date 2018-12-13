# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '12/1/18'

from keras.layers import Embedding
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer


class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = self.nb_head * self.size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.WQ = self.add_weight(name='wq',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  trainable=True,
                                  initializer='glorot_uniform')
        self.WK = self.add_weight(name='wk',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  trainable=True,
                                  initializer='glorot_uniform')
        self.WV = self.add_weight(name='wv',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  trainable=True,
                                  initializer='glorot_uniform')

        super(Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if len(inputs) == 3:
            q_seq, v_seq, k_seq = inputs

        q_seq = K.dot(q_seq, self.WQ)  # shape = (batchz_size, m, 128)
        k_seq = K.dot(k_seq, self.WK)
        v_seq = K.dot(v_seq, self.WV)  # shape = (batchz_size, m, 128)

        q_seq = K.reshape(q_seq, (-1, K.shape(q_seq)[1], self.nb_head, self.size_per_head))
        k_seq = K.reshape(k_seq, (-1, K.shape(k_seq)[1], self.nb_head, self.size_per_head))
        v_seq = K.reshape(v_seq, (-1, K.shape(v_seq)[1], self.nb_head, self.size_per_head))  # (? ? 8 16)

        q_seq = K.permute_dimensions(q_seq, (0, 2, 1, 3))
        k_seq = K.permute_dimensions(k_seq, (0, 2, 1, 3))
        v_seq = K.permute_dimensions(v_seq, (0, 2, 1, 3))  # (?, 8, ?, 16)


        QK = K.batch_dot(k_seq, q_seq, axes=(3, 3)) / self.size_per_head**0.5  # shape = (?, 8, ?, ?)
        A = K.softmax(QK)

        # 输出并mask
        O_seq = K.batch_dot(A, v_seq, axes=[3, 2])  # softmax(Q * K ,mask ) * V
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))

        return O_seq




    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

