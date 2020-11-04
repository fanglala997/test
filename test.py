import jieba
import numpy as np
import matplotlib
import fileDispose
from glove import glove
from glove import corpus


word2id = fileDispose.getFile('small_word2id.json')
total_list = fileDispose.getFile('total_list.json')
count = 0
for i in range(5000,10000,2):
    flag = 0
    for word in total_list[i]:
        if word not in word2id:
            flag = 1
            break
    if flag == 0:
        print(total_list[i])
        count += 1

    if count > 10:
        break





