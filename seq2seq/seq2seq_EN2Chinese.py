#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@software: PyCharm Community Edition
@file: seq2seq_EN2Chinese.py
@time: 12/17/18 11:26 AM
"""

from  __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 32
epochs = 100
latent_dim = 256
num_samples = 10000
data_path = 'cmn-eng/cmn.txt'

input_texts = []
target_texts = []
input_characters = set()
tartget_characters = set()

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')

    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        input_characters.add(char)
    for char in target_text:
        tartget_characters.add(char)

input_characters = sorted(list(input_characters))
tartget_characters = sorted(list(tartget_characters))

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(tartget_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(tartget_characters)])


encoder_input_data = np.zeros((len(input_texts),
                               max_encoder_seq_length,
                               num_encoder_tokens),
                              dtype='float32')
decoder_input_data = np.zeros((len(input_texts),
                               max_decoder_seq_length,
                               num_decoder_tokens),
                              dtype='float32')
decoder_target_data = np.zeros((len(input_texts),
                                max_decoder_seq_length,
                                num_decoder_tokens),
                               dtype='float32')

for t, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for c, char in enumerate(input_text):
        encoder_input_data[t, c, input_token_index[char]] = 1.
    for c, char in enumerate(target_text):
        decoder_input_data[t, c, target_token_index[char]] = 1.
        if c > 0:
            decoder_target_data[t, c - 1, target_token_index[char]] = 1.

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_outputs, _, __ = LSTM(latent_dim, return_state=True, return_sequences=True)(decoder_inputs, initial_state=encoder_states)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

model.save_weights('e2c.h5')