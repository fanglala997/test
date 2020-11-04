import json
import os

work_path = os.getcwd()
save_path = work_path.replace('\\','/') + '/Data/'
# print(save_path)
def writeToFile(item,filename):
    """
    数据写入文件，json格式
    :param item: 写入文件对象
    :param filename: 文件位置，默认存在Data里
    :return:
    """
    # 将数据写入到文件中
    file = open(filename,'w')
    str = json.JSONEncoder().encode(item)
    file.write(str)
    file.close()

def getFile(filename):
    """
    读取文件
    :param filename: 文件路径，默认从Data文件夹开始
    :return:
    """
    file = open(save_path + filename, 'r')
    indexStr = file.read()
    index = json.JSONDecoder().decode(indexStr)
    file.close()
    return index

def get_dic(sen_list):
    """
    将此列表转化为dic
    如{'晚风'：0,'摇':1....}以及{0：'晚风',1:'摇'....}
    list:序列列表形如：[['晚风','摇','树'....]['晨露','润',...]....]
    :return: word2id字典以及id2word
    """
    word2id = {}
    id2word = {}
    for i in range(1,len(sen_list)):
        sen_list[0].extend(sen_list[i])
    dic = set(sen_list[0])
    list_dic = list(dic)
    for i in range(len(list_dic)):
        word2id.setdefault(list_dic[i],i)
        id2word.setdefault(i,list_dic[i])
    return word2id,id2word



def get_small_corpus(num=10000):
    """
    获取小型文本库，用于调试网络模型
    :param num: 文本库前n/2条对联
    :return: 默认返回前500条对联（1000句话）的list
    """
    list = getFile('/total_list.json')
    return list[:num]

# def fit_corpus(total_list,test_falg=True):
#     corpus_shang = []
#     corpus_xia = []
#     new_list = []
#     if test_falg:
#         for sentence in total_list:
#             start = ['<start>']
#             end = ['<end>']
#             for i in range(len(sentence)):
#                 start.append(sentence[i])
#             start.extend(end)
#             new_list.append(start)
#         for i in range (0,len(new_list),2):
#             corpus_shang.append(new_list[i])
#         for i in range (1,len(new_list),2):
#             corpus_xia.append(new_list[i])
#     else:
#         print('mei xie hao')
#
#
#     return corpus_shang,corpus_xia


def fit_corpus(total_list,test_flag=True):
    corpus_shang = []
    corpus_xia = []
    new_list = []
    if test_flag:
        for sentence in total_list:
            start = ['<start>']
            end = ['<end>']
            for i in range(len(sentence)):
                start.append(sentence[i])
            start.extend(end)
            new_list.append(start)
        for i in range(0, len(new_list), 2):
            corpus_shang.append(new_list[i])
        for i in range (1,len(new_list),2):
            corpus_xia.append(new_list[i])

    else:
        print('mei xie hao')


    return corpus_shang,corpus_xia





def embedding_fit(total_list,test_flag=True):
    new_list = []
    if test_flag:
        for sentence in total_list:
            start = ['<start>']
            end = ['<end>']
            for i in range(len(sentence)):
                start.append(sentence[i])
            start.extend(end)
            new_list.append(start)
    else:
        print('mei xie hao')


    return new_list
