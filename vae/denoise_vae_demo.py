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

