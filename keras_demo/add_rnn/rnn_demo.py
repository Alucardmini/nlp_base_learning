#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: rnn_demo.py
@time: 12/11/18 9:36 AM
"""

import numpy as np

# Parameters for the model and dataset.
TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = True

MAXLEN = DIGITS + 1 + DIGITS
chars = '0123456789+ '
seen = set()

questions = []
expected = []
def generateQuestions():
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(1, DIGITS + 1)))
    while len(questions) < TRAINING_SIZE:
        a, b = f(), f()
        key = tuple(sorted((a, b)))
        if (a, b) in seen:
            continue
        seen.add(key)
        q = '{}+{}'.format(a, b)
        query = q + ' ' * (MAXLEN - len(q))
        ans = str(a + b)
        ans += ' ' * (DIGITS + 1 - len(ans))

        if REVERSE:
            # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
            # space used for padding.)
            query = query[::-1]
        questions.append(query)
        expected.append(ans)


class CharacterTable(object):

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

if __name__ == '__main__':
    generateQuestions()

    x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(expected), DIGITS+1, len(chars)), dtype=np.bool)
    ctable = CharacterTable(chars)
    for i, q in enumerate(questions):
        x[i] = ctable.encode(q, MAXLEN)
    for i, a in enumerate(expected):
        y[i] = ctable.encode(a, DIGITS + 1)

    split_at = len(x) - len(x)//10

    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
    from keras.optimizers import rmsprop, sgd, adam

    HIDDEN_SIZE = 128
    BATCH_SIZE = 128
    LAYERS = 1

    model = Sequential()
    model.add(LSTM(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
    model.add(RepeatVector(DIGITS + 1))
    model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
    model.add(TimeDistributed(Dense(len(chars), activation='softmax')))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()

    for iteration in range(1, 200):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=1,
                  validation_data=(x_val, y_val))
        # Select 10 samples from the validation set at random so we can visualize
        # errors.
        for i in range(10):
            ind = np.random.randint(0, len(x_val))
            rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
            preds = model.predict_classes(rowx, verbose=0)
            q = ctable.decode(rowx[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            print('Q', q[::-1] if REVERSE else q, end=' ')
            print('T', correct, end=' ')
            if correct == guess:
                print(colors.ok + '☑' + colors.close, end=' ')
            else:
                print(colors.fail + '☒' + colors.close, end=' ')
            print(guess)