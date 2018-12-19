#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@software: PyCharm Community Edition
@file: vae_gen_poem.py
@time: 12/19/18 11:34 AM
"""

import keras.backend as K
from keras.layers import Dense, Lambda, Conv1D, Embedding, Input, GlobalAveragePooling1D, Reshape
from keras.losses import mse, categorical_crossentropy
import numpy as np
import re
import codecs

n = 7  # 只抽取五言诗
latent_dim = 64  # 隐变量维度
hidden_dim = 64  # 隐层节点数

s = codecs.open('data/shi.txt', encoding='utf-8').read()

# 通过正则表达式找出所有的五言诗
s = re.findall(u'　　(.{%s}，.{%s}。.*?)\r\n'%(n,n), s)
shi = []
for i in s:
    for j in i.split(u'。'): # 按句切分
        if j:
            shi.append(j)

shi = [i[:n] + i[n+1:] for i in shi if len(i) == 2*n+1]

# 构建字与id的相互映射
id2char = dict(enumerate(set(''.join(shi))))
char2id = {j:i for i,j in id2char.items()}

# 诗歌id化
shi2id = [[char2id[j] for j in i] for i in shi]
shi2id = np.array(shi2id)

from keras.engine.topology import Layer

class GCNN(Layer): # 定义GCNN层，结合残差
    def __init__(self, output_dim=None, residual=False, **kwargs):
        super(GCNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.residual = residual
    def build(self, input_shape):
        if self.output_dim == None:
            self.output_dim = input_shape[-1]
        self.kernel = self.add_weight(name='gcnn_kernel',
                                     shape=(3, input_shape[-1],
                                            self.output_dim * 2),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def call(self, x):
        _ = K.conv1d(x, self.kernel, padding='same')
        _ = _[:,:,:self.output_dim] * K.sigmoid(_[:,:,self.output_dim:])
        if self.residual:
            return _ + x
        else:
            return _


inputs = Input(shape=(2*n, ), dtype='int32')
embedding = Embedding(len(char2id), hidden_dim)(inputs)

h = GCNN(residual=True)(embedding)
h = GCNN(residual=True)(h)
h = GlobalAveragePooling1D()(h)

z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
    return z_mean + K.exp(z_log_var/2)*epsilon
z = Lambda(sampling)([z_mean, z_log_var])

decoder_hidden = Dense(hidden_dim*(2*n))
decoder_cnn = GCNN(residual=True)
decoder_Dense = Dense(len(char2id), activation='softmax')

h = decoder_hidden(z)
h = Reshape((2*n, hidden_dim))(h)
h = decoder_cnn(h)
output = decoder_Dense(h)

from keras.models import Model
vae = Model(inputs, output)

xent_loss = K.sum(K.sparse_categorical_crossentropy(inputs, output), 1)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# 重用解码层，构建单独的生成模型
decoder_input = Input(shape=(latent_dim,))
_ = decoder_hidden(decoder_input)
_ = Reshape((2*n, hidden_dim))(_)
_ = decoder_cnn(_)
_output = decoder_Dense(_)
generator = Model(decoder_input, _output)

# 利用生成模型随机生成一首诗
def gen():
    r = generator.predict(np.random.randn(1, latent_dim))[0]
    r = r.argmax(axis=1)
    return ''.join([id2char[i] for i in r[:n]])\
           + u'，'\
           + ''.join([id2char[i] for i in r[n:]])



vae.fit(shi2id,
        shuffle=True,
        epochs=20,
        batch_size=64)

vae.save_weights('shi.model')

for i in range(20):
    print(gen())


