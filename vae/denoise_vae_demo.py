#!/usr/bin/python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

np.random.seed(1337)

(x_train, _), (x_test, _) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test = np.clip(x_train_noisy, 0., 1.)

input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
laten_dim = 16
layer_filters = [32, 64]

inputs = Input(shape=input_shape, name='ecoder_input')
x = inputs

for filtes in layer_filters:
    x = Conv2D(filters=filtes,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)

shape = K.int_shape(x)

x = Flatten()(x)
latent = Dense(laten_dim, name='latent_layer')(x)
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

latent_inputs = Input(shape=(laten_dim, ), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for filtes in layer_filters[::-1]:
    x = Conv2DTranspose(filtes=filtes,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

x = Conv2DTranspose(filtes=1, kernel_size=kernel_size, padding='same')
outputs = Activation('sigmoid', name='decoder_output')(x)

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

autoencoder.compile(loss='mse', optimizer='adam')