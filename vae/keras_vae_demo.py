#!/usr/bin/python
#coding:utf-8

from keras.layers import Dense, Input, Lambda
from keras.datasets import mnist
from keras.models import Model
from keras.losses import mse, binary_crossentropy

import numpy as np
import keras.backend as K

(x_train, _), (x_test, _) = mnist.load_data()

print(x_train.shape)
epochs = 20
image_size = x_train.shape[1]
origin_dim = image_size * image_size
batch_size = 128
latent_dim = 2

x_train = np.reshape(x_train, [-1, origin_dim])
x_test = np.reshape(x_test, [-1, origin_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))  # 标准正态分布  z_mean = 均值 z_log_var 方差

    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# encoder
inputs = Input(shape=(origin_dim,), name='encoder_inputs')
x = Dense(512, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling, output_shape=(latent_dim, ), name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

latent_input = Input(shape=(latent_dim, ), name='z_sampling')
x = Dense(512, activation='relu')(latent_input)
outputs = Dense(origin_dim, activation='sigmoid')(x)
decoder = Model(latent_input, outputs, name='decoder')

outputs = decoder(encoder(inputs)[2])
# outputs = decoder(encoder(inputs))
vae = Model(inputs, outputs, name='vae_mlp')


reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= origin_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

vae.compile(optimizer='adam')
vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))

vae.save_weights('vae_mlp_mnist.h5')



