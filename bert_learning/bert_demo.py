#!/usr/bin/python
#coding:utf-8

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# text = "Who was Jim henson? Jim Henson was a puppeteer"
# tokenized_text = tokenizer.tokenize(text)
# print(tokenized_text)

text = "我爱你中国，　改革开放多久了? 四十周年啦"
tokenized_text = tokenizer.tokenize(text)

print(tokenized_text)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print(indexed_tokens)
