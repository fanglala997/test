import matplotlib
import fileDispose
import numpy as np

def get_bench(text,n):
    """
    get_bench(list,int)
    共现矩阵窗口循环的范围，对联选择上下联两句做循环范围
    text：文本列表
    n：几句话合并

    """
    bench = []
    lable = 0
    for i in range(len(text)):
        if(i % n == 0):
            lable = i
            continue
        else:
            text[lable].extend(text[i])
        bench.append(text[lable])
    return bench





def co_occurrence_matrix_for_word(window,text,word_listindex):
    re_matrix =np.zeros((len(word_listindex),len(word_listindex)),dtype=int)
    bench = get_bench(text,2)
    for sentence in bench:
        for i in range(len(sentence)):
            for j in range(1,window+1):
                if(i-j >= 0):
                    n = int(word_listindex[sentence[i-j]])
                    re_matrix[i,n] += 1
                if(i+j <= len(sentence)):
                    n = int(word_listindex[sentence[i+j-1]])
                    re_matrix[i,n] += 1
                else:
                    continue
    return re_matrix




total_list = fileDispose.getFile('total_list.json')
word_listindex = fileDispose.getFile('allcut_word_listindex.json')


co_occurrence = co_occurrence_matrix_for_word(2,total_list,word_listindex)

# fileDispose.writeToFile(co_occurrence.tolist(),'./Data/train/co_occurrence.json')
np.savetxt('./Data/train/co_occurrence.txt',co_occurrence)


# print(co_occurrence[:,1])

# bench = get_bench(total_list,2)
# print(bench[:10])

# danzi = 0
# shuangzi = 0
# qita = 0
# for k in word_listindex:
#     if (len(word_listindex[k]) == 1):
#         danzi += 1
#     elif (len(word_listindex[k]) == 2):
#         shuangzi += 1
#     else:
#         qita += 1
# print('单个字词个数：',danzi)
# print('双字词个数：',shuangzi)
# print('其他字词个数：',qita)

# for i, (k, v) in enumerate(word_listindex.items()):
#     if i in range(0, 10):
#         print(k, v)