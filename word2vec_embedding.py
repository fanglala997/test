from gensim.models.word2vec import Word2Vec
import fileDispose
import numpy as np


def get_word2vec(corpus,size=500,window=3,sg=1,epochs=50,save_flag=False):
    """
    获取word2vec词向量
    :param corpus: 输入词库，如glove
    :param size: 生成词向量维度
    :param window: 词窗大小如glove
    :param sg: 是否使用skip-grams模式，为0时使用CBOW模式
    :param epochs: 训练次数
    :param save_flag: 是否保存
    :return: word2vec模型
    """
    model = Word2Vec(size=size, workers=window, sg=sg, iter=epochs,min_count=0,alpha=0.005)  # 生成词向量为500维，考虑上下30个单词共11个单词，采用sg=1的方法也就是skip-gram
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
    if save_flag:
        model.save('./Data/train/word2vec_model')
    return model


def get_most_similar(model,word,topn=10,re_flag=False):
    """
    获取相似单词
    :param model: 生成的word2vec模型
    :param word:相似度输入单词
    :param topn:前n个单词
    :param 是否有返回列表
    :return:相似词列表
    """
    sim_words = model.most_similar(positive=[word], topn=topn)
    for word, similarity in sim_words:
        print(word, similarity)
    if re_flag:
        return sim_words


def fit_vec(word2vec_model,dictionary):
    """
    整理word2vec生成向量矩阵
    :param word2vec_model:已生成word2vec模型
    :param dictionary: 自己生成的word2id dictionary
    :return: word_vec
    [[word1_vec][word2_vec][word3_vec]...[wordn_vec]]
    """
    word_vec = []
    for word in dictionary:
        word_vec.append(word2vec_model[word])
    return word_vec


if __name__ == '__main__':
    word2id = fileDispose.getFile('word2id.json')
    list1 = fileDispose.getFile('total_list.json')
    w2v_list = fileDispose.embedding_fit(list1)
    print(w2v_list[:10])
    w2v_model = get_word2vec(w2v_list)
    wordvec = fit_vec(w2v_model,word2id)
    print(wordvec[:2])
    np.savetxt('./Data/train/word2vec_vec.txt',wordvec)






