#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: hmm_model.py
@time: 12/5/18 11:47 AM
"""
from collections import Counter
import os
import sys
from math import log
import pickle

# hmm_model = {i: Counter() for i in 'smbe'}
#
# path = '/home/wuxikun/nlp_base/nlp_base_learning/lh_static_learning_method/data/dict.txt'
#
# with open(path, 'r')as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.strip().split(' ')
#         if len(line_list) > 0:
#             word = line_list[0]
#             if len(word)==0 or word == '\n' or len(line_list)!=3:
#                 continue
#
#             if len(word) == 1:
#                 hmm_model['s'][word] += int(line_list[1])
#
#             else:
#                 hmm_model['b'][word[0]] += int(line_list[1])
#                 hmm_model['e'][word[-1]] += int(line_list[1])
#                 for i in word[1:-1]:
#                     hmm_model['m'][i] += int(line_list[1])

model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    'data/hmm.model')
print(model_path)

# with open(model_path, 'wb') as f:
#     pickle.dump(hmm_model, f)
with open(model_path, 'rb')as f:
    hmm_model = pickle.load(f)

log_total = {i: log(sum(hmm_model[i].values())) for i in 'sbme'}
print(log_total)

trans = {'ss':0.3,
    'sb':0.7,
    'bm':0.3,
    'be':0.7,
    'mm':0.3,
    'me':0.7,
    'es':0.3,
    'eb':0.7
 }

trans = {i: log(float(trans[i])) for i in trans}

def viterbi(nodes):
    paths = nodes[0]
    for l in range(1, len(nodes)):
        paths_ = paths
        paths = {}
        for i in nodes[l]:
            # i 当前节点的 "sbme"
            nows = {}
            for j in paths_:
                # j 上一个节点的 "sbme"
                if j[-1] + i in trans:
                    nows[j + i] = paths_[j] + nodes[l][i] + trans[j[-1] + i]
            now_key = get_key(nows, max(nows.values()))
            paths[now_key[0]] = max(nows.values())
    return get_key(paths, max(paths.values()))


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def hmm_cut(s):
    nodes = [{i:log(hmm_model[i][t]+1)-log_total[i] for i in hmm_model} for t in s]
    tags = viterbi(nodes)
    if len(tags) >0:
        tags = tags[0]
    else:
        return
    word = ['']

    for i in range(len(tags)):
        if tags[i] in ['b', 's']:
            word.append(s[i])
        else:
            word[-1] += s[i]

    return word



if __name__ == '__main__':
    test_str = u'李想是一个好孩子'
    print(' '.join(hmm_cut(test_str)))


