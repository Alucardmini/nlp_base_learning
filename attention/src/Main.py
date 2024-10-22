# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '12/1/18'

from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Model
from keras.layers import *

try:

    from src.attention_keras import Position_Embedding
    from src.attention_keras import Attention
except:
    # from attention.src.attention_keras import Attention
    from attention.src.attention_keras import Position_Embedding
    from attention.src.custom_attention import Attention
maxfeature = 200
batch_size = 32
maxlen = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=maxfeature)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

s_inputs = Input(shape=(None, ), dtype='int32')
embeddings = Embedding(maxfeature, 128, name='embedding')(s_inputs)

O_seq = Attention(8, 16, name='attention')([embeddings, embeddings, embeddings])
# O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = GlobalMaxPool1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
# O_seq = MyLayer(10)(O_seq)

outputs = Dense(1, activation='sigmoid')(O_seq)

model = Model(s_inputs, outputs)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size,
          epochs=1,
          validation_data=(x_test, y_test))

# embed_name = 'embedding'
# embed_layer_model = Model(inputs=model.input, outputs=model.get_layer(embed_name).output)
# embed_output = embed_layer_model.predict(x_train[:10])
# 10 (size) 200 * 128
#
# attention_layer_name = 'attention'
# attention_layer_model = Model(inputs=embed_output, outputs=model.get_layer(attention_layer_name).output)
# attention_output = attention_layer_model.predict(embed_output[:10])


