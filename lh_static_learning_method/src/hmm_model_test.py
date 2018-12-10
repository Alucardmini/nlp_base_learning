#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: hmm_model_test.py
@time: 12/10/18 10:59 AM
"""

import numpy as np
from collections import Counter
import os
import pickle
from math import log


# hmm_model = {i: Counter() for i in 'sbem'};
# # print(hmm_model)
# with open('../data/dict.txt') as f:
#     lines = f.readlines()
#
#     for line in lines:
#         line = line.strip()
#         line_list = line.split(' ')
#         if len(line_list) != 3:
#             continue
#         word = line_list[0]
#         count = int(line_list[1])
#
#         if len(word) == 1:
#             hmm_model['s'][word] += count
#         else:
#             hmm_model['b'][word[0]] += count
#             hmm_model['e'][word[-1]] += count
#             for w in range(1, len(word)-1):
#                 hmm_model['m'][w] += count
#
# print(hmm_model)


model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    'data/hmm_test.model')

# with open(model_path, 'wb') as f:
#     pickle.dump(hmm_model, f)
with open(model_path, 'rb')as f:
    hmm_model = pickle.load(f)

trans = {
    'ss': 0.3,
    'sb': 0.7,
    'be': 0.7,
    'bm': 0.3,
    'me': 0.7,
    'mm': 0.3,
    'eb': 0.3,
    'es': 0.7

}

trans = {i: log(float(trans[i])) for i in trans}
log_total = {i: log(sum(hmm_model[i].values())) for i in 'sbme'}
# print(log_total)

def viterbi(nodes):
    current_path = nodes[0]

    for n in range(1, len(nodes)):
        tmp_path = {}
        for i in nodes[n]:
            now = {}
            for j in current_path:
                if j[-1] + i in trans:
                    now[j + i] = nodes[n][i] + current_path[j] + trans[j[-1] + i]

            max_v = get_key(now, max(now.values()))
            tmp_path[max_v[0]] = max(now.values())
        current_path = tmp_path

    return get_key(current_path, max(current_path.values()))


def get_key(src_dict, value):
    return [k for k, v in src_dict.items() if v == value]


def hmm_cut(s):
    nodes = [{i:log(hmm_model[i][t] + 1) for i in hmm_model} for t in s]
    tags = viterbi(nodes)
    print(tags)
    if len(tags) > 0:
        tags = tags[0]
    word = []
    for i in range(len(tags)):
        if tags[i] == 's' or tags[i] == 'b':
            word.append(s[i])
        else:
            word[-1] += s[i]
    print(word)


if __name__ == '__main__':
    src_content = '赵刚是个好同志'
    hmm_cut(src_content)


