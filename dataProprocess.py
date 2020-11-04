import pickle
import os
import jieba
import fileDispose

work_path = os.getcwd()
data_path = work_path.replace('\\','/') + '/'+'Data/data.pkl'
# print(data_path)
fr = open(data_path,'rb')
information = pickle.load(fr)
# print(information[:10])


# 句子合并
total_shanglian = []
total_xialian = []
for i in range(len(information)):
    str_shang =''
    str_xia =''
    temp_tuple =information[i]
    shanglian = temp_tuple[0]
    xialian = temp_tuple[1]
    for i in range(1,len(shanglian)-1):
        str_shang += str(shanglian[i])
    total_shanglian.append(str_shang)

    for i in range(1,len(xialian)-1):
        str_xia += str(xialian[i])
    total_xialian.append(str_xia)




shanglian_cut = []
xialian_cut = []
sum = 0
error_list = []
if(len(total_xialian)==len(total_shanglian)):

    for i in range(len(total_shanglian)):
        sentence = total_shanglian[i]
        text_shang = sentence.replace('，','')
        text_shang = text_shang.replace('。', '')
        text_shang = text_shang.replace('！', '')
        text_shang = text_shang.replace('？', '')
        text_shang = text_shang.replace('：', '')
        text_shang = text_shang.replace('；', '')
        text_shang = text_shang.replace('、', '')
        text_shang = text_shang.replace('——', '')
        # wordlist = jieba.lcut(sentence,HMM=True)
        wordlist = jieba.lcut(sentence,cut_all=True,HMM=True)
        while('' in wordlist):
            wordlist.remove('')
        shanglian_cut.append(wordlist)

        tempstr = total_xialian[i]
        templist = []
        # text =tempstr
        text = tempstr.replace('，','')
        text = text.replace('！', '')
        text = text.replace('。', '')
        text = text.replace('、', '')
        text = text.replace('——', '')
        text = text.replace('：', '')
        text = text.replace('；', '')
        text = text.replace('？', '')
        for j in range(len(wordlist)):
            word = wordlist[j]
            if (len(text)< len(word)):
                # print('列表序号： ',i)
                sum += 1
                templist = []
                error_list.append(i)
                break
            else:
                str = text[:len(word)]
                templist.append(str)
                text = text[len(word):]
        if(len(templist) != 0):
            xialian_cut.append(templist)

    print('分词后损失文本集：',sum)
    print('--------------------------------------')

else:
    print('上下联数不等')

for i in range(len(error_list)):
    shanglian_cut[error_list[i]] = []


new_shanglian_cut =[]
for i in range(len(shanglian_cut)):
    if(shanglian_cut[i] != []):
        new_shanglian_cut.append(shanglian_cut[i])

new_xialian_cut = xialian_cut
# print(new_shanglian_cut[:100])
# print(new_xialian_cut[:100])




fileDispose.writeToFile(new_shanglian_cut,work_path+'/Data/shanglian_all_cut.json')
fileDispose.writeToFile(new_xialian_cut,work_path+'/Data/xialian_all_cut.json')


shanglian_all_cut = fileDispose.getFile('shanglian_all_cut.json')
xialian_all_cut = fileDispose.getFile('xialian_all_cut.json')
total_list = []
for i in range(len(shanglian_all_cut)):
    total_list.append(shanglian_all_cut[i])
    total_list.append(xialian_all_cut[i])

fileDispose.writeToFile(total_list,'./Data/total_list.json')




# total_word = []
# for i in range(len(new_shanglian_cut)):
#     templist = new_shanglian_cut[i]
#     for word in templist:
#         total_word.append(word)
#
# for i in range(len(new_xialian_cut)):
#     templist = new_xialian_cut[i]
#     for word in templist:
#         total_word.append(word)
#
# word_list = set(total_word)
# # print(word_list)
# new_word_list = list(word_list)
# word_listindex = {}
#
# for i in range(len(new_word_list)):
#     word_listindex.setdefault(new_word_list[i],i)
#
# for i,(k,v) in enumerate(word_listindex.items()):
#     if i in range(0,10):
#         print(k,v)
#
# print('词库长度：',len(new_word_list))
# print('--------------------------------------')
# print('文本训练集',len(new_shanglian_cut),len(new_xialian_cut))
#
# fileDispose.writeToFile(word_listindex,work_path+'/Data/allcut_word_listindex.json')












