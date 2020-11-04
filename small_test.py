import fileDispose
import glove_embedding
import word2vec_embedding


small_list = fileDispose.get_small_corpus()
shang,xia = fileDispose.fit_corpus(small_list)
print(small_list)
print(shang[:10])
# new_list = fileDispose.fit_corpus(small_list)
# print(new_list[:10])
# small_dic = fileDispose.get_dic(small_list)
# print('词库长度：',len(small_dic))
# # glove_model = glove_embedding.get_glove_word_vec(small_list,small_dic)
# # print(glove_model.word_vectors[:10])
#
# # print('-----------下面为word2vec---------')
# # str = '晚风'
# word2vec_model = word2vec_embedding.get_word2vec(small_list)
# print(word2vec_model['晚风'])
# # word2vec_embedding.get_most_similar(word2vec_model,str)