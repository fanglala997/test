from __future__ import print_function
import numpy as np
import fileDispose
from glove import glove
from glove import corpus


def get_glove_word_vec(corpus_text,dictionary,window=5,no_components=100,
                       learning_rate=0.005,no_threads=10,epochs=50,save_flag=False):
    """
    获取glove方法的文本向量化模型
    :param corpus_text: 输入文本集list形式[['word1','word2',...],['wordt',....].....]
    :param dictionary: 输入词典的word2id形式，字典索引范围为0-len（dictionary）-1
    :param window: glove词窗大小默认为10，左右10词，共21词
    :param no_components: 生成词向量维度
    :param learning_rate: 学习率
    :param no_threads: 设置线程数
    :param epochs: 训练次数
    :param save_flag: 是否保存文件，默认./Data/train/corpus.model
    :return: glove模型
    """
    corpus_model = corpus.Corpus(dictionary)
    corpus_model.fit(corpus_text, window=window)
    if save_flag:
        corpus_model.save('./Data/train/corpus.model')
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)

    co_occurrence_matrix = corpus_model.matrix
    glove_model = glove.Glove(no_components=no_components, learning_rate=learning_rate)
    glove_model.fit(co_occurrence_matrix, no_threads=no_threads, epochs=epochs)
    glove_model.add_dictionary(dictionary)
    return glove_model



def get_glove_gra_vec(glove_model,corpus_text,dictionary,save_flag=False,no_components=100):
    """
    使用glove方法生成句向量（没有使用TFIDF方法直接求词向量均值）
    :param glove_model: 生成的glove模型
    :param corpus_text: 文本集
    :param dictionary: word2id形式字典
    :param save_flag: 是否保存，默认./Data/train/gl_paragraph_vectors.txt
    :param no_components: 句向量维度默认100，要与词向量一样
    :return: 句向量矩阵list
    """
    word_vec = glove_model.word_vectors
    gra_vec = []
    for sentence in corpus_text:
        a = np.zeros(no_components, dtype=float)
        for i in range(len(sentence)):
            a = np.row_stack((a, word_vec[int(dictionary[sentence[i]])]))

        one_gra = np.mean(a, axis=0)
        gra_vec.append(one_gra)
    if save_flag:
        fileDispose.writeToFile(gra_vec, './Data/train/gl_paragraph_vectors.txt')
    return gra_vec


def get_most_similar(glove_model,str,topn=10,re_list=False):
    """
    求词典中某次最相似topn
    :param glove_model: 生成的glove模型
    :param str: 输入词
    :param topn: topn？
    :return: 包含相似词和相似度的列表
    """
    list = glove_model.most_similar(str, topn)
    if re_list:
        return list
    print(list)






if __name__ =='__main__':
    # list1 = fileDispose.getFile('total_list.json')
    # glove_list = fileDispose.embedding_fit(list1)
    # word2id,id2word = fileDispose.get_dic(glove_list)
    # teststr = id2word[0]
    # # for i,(k,v) in enumerate(word2id.items()):
    # #     if i in range(0,20):
    # #         print(k,v)
    #
    # print(word2id['<start>'])
    # print(id2word[100])
    # fileDispose.writeToFile(word2id,'./Data/word2id.json')
    # fileDispose.writeToFile(id2word, './Data/id2word.json')
    # print('word2id done')
    # print('id2word done')
    # glove_model = get_glove_word_vec(glove_list,word2id,no_components=500)
    # np.savetxt('./Data/train/glove_vec.txt',glove_model.word_vectors)


    list1 = fileDispose.get_small_corpus(num=5000)
    glove_list = fileDispose.embedding_fit(list1)
    small_word2id, small_id2word = fileDispose.get_dic(glove_list)
    fileDispose.writeToFile(small_word2id, './Data/small_word2id.json')
    fileDispose.writeToFile(small_id2word, './Data/small_id2word.json')
    glove_model = get_glove_word_vec(glove_list, small_word2id, no_components=400)
    np.savetxt('./Data/train/small_glove_vec.txt', glove_model.word_vectors)













