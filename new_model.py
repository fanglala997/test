import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import fileDispose
import numpy as np
import os
import time

from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']


word2id = fileDispose.getFile('small_word2id.json')
id2word = fileDispose.getFile('small_id2word.json')


def max_length(tensor):
    return max(len(t) for t in tensor)


def fit_index(sen_list):
    new_w2id = {}
    new_id2w = {}
    for i in range(1,len(sen_list)):
        sen_list[0].extend(sen_list[i])
    dic = set(sen_list[0])
    list_dic = list(dic)
    for word in list_dic:
        new_w2id.setdefault(word,word2id[word])
        new_id2w.setdefault(word2id[word],word)

    return new_w2id,new_id2w




def load_dataset(shanglian,xialian,max_len):
    # 创建清理过的输入输出对
    input_tensor = tokenize(shanglian,max_len)
    target_tensor = tokenize(xialian,max_len)

    return input_tensor, target_tensor



def tokenize(tensor,max_len):
    new_tensor = []
    for text in tensor:
        templist = []
        for word in text:
            templist.append(word2id[word])
        new_tensor.append(templist)

    new_tensor = fit_sequence(new_tensor,max_len)

    return new_tensor



def fit_sequence(tensor,max_len):
    length = max_len
    print(length)
    for seq in tensor:
        for i in range(len(seq),length):
            seq.append(0)
    return tensor



def convert(tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, id2word[str(t)]))


class Encoder(tf.keras.Model):

  def __init__(self, vocab_size, embedding_dim,embedding_matrix, enc_units, batch_sz,max_len):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,weights=[embedding_matrix],input_length=max_len,trainable=False)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))




class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # 隐藏层的形状 == （批大小，隐藏层大小）
    # hidden_with_time_axis 的形状 == （批大小，1，隐藏层大小）
    # 这样做是为了执行加法以计算分数
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # 分数的形状 == （批大小，最大长度，1）
    # 我们在最后一个轴上得到 1， 因为我们把分数应用于 self.V
    # 在应用 self.V 之前，张量的形状是（批大小，最大长度，单位）
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # 注意力权重 （attention_weights） 的形状 == （批大小，最大长度，1）
    attention_weights = tf.nn.softmax(score, axis=1)

    # 上下文向量 （context_vector） 求和之后的形状 == （批大小，隐藏层大小）
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights





class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, batch_sz,max_len):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,weights=[embedding_matrix],input_length=max_len,trainable=False)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # 用于注意力
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # 编码器输出 （enc_output） 的形状 == （批大小，最大长度，隐藏层大小）
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
    x = self.embedding(x)

    # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # 将合并后的向量传送到 GRU
    output, state = self.gru(x)

    # 输出的形状 == （批大小 * 1，隐藏层大小）
    output = tf.reshape(output, (-1, output.shape[2]))

    # 输出的形状 == （批大小，vocab）
    x = self.fc(output)

    return x, state, attention_weights





def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)




@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_w2id['<start>']] * BATCH_SIZE, 1)

    # 教师强制 - 将目标词作为下一个输入
    for t in range(1, targ.shape[1]):
      # 将编码器输出 （enc_output） 传送至解码器
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # 使用教师强制
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss





def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    # sentence = fileDispose.fit_corpus(sentence)#输入格式[['word1','word2'...],....]

    inputs = tokenize(sentence,max_len)
    inputs = np.asarray(inputs)
    inputs = tf.convert_to_tensor(inputs)
    print(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_w2id['<start>']], 0)
    print('de_input=',dec_input)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # 存储注意力权重以便后面制图
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        print('预测id：',predicted_id)

        result += targ_id2w[predicted_id] + ' '

        if targ_id2w[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # 预测的 ID 被输送回模型
        dec_input = tf.expand_dims([predicted_id], 0)


    return result, sentence, attention_plot

# 注意力权重制图函数
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()




def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence[0])]
    plot_attention(attention_plot, sentence[0], result.split(' '))







small_list = fileDispose.get_small_corpus(num=5000)
shanglian,xialian = fileDispose.fit_corpus(small_list)
max_len = max_length(shanglian)
# print(shanglian[:20])
# print(xialian[:20])
inp_lang = shanglian
targ_lang = xialian
input_tensor, target_tensor = load_dataset(shanglian,xialian,max_len)
# print(input_tensor[:5])
# print(target_tensor[:5])

inp_w2id,inp_id2w = fit_index(shanglian)
targ_w2id,targ_id2w = fit_index(xialian)

max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
# 采用 80 - 20 的比例切分训练集和验证集
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val =\
                                                            train_test_split(input_tensor, target_tensor, test_size=0.2)
# 显示长度
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
print(input_tensor_train[0])
convert(input_tensor_train[0])
convert(target_tensor_train[0])

# vocab_inp_size = len(inp_w2id)+1
# vocab_tar_size = len(targ_w2id)+1
vocab_size = len(word2id)

embedding = np.loadtxt('./Data/train/small_glove_vec.txt')
# embedding = np.asarray(embd)


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = len(embedding[0])
units = 1024


dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape, example_target_batch.shape)



encoder = Encoder(vocab_size,embedding_dim,embedding,units, BATCH_SIZE,max_len)


# 样本输入
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

decoder = Decoder(vocab_size, embedding_dim,embedding, units, BATCH_SIZE,max_len)

sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),#tf.random.uniform((batch数，1))
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

# optimizer = tf.keras.optimizers.SGD()
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


EPOCHS = 70

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    if batch % 100 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                     batch,
                                                     batch_loss.numpy()))
  # 每 2 个周期（epoch），保存（检查点）一次模型
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

# 恢复检查点目录 （checkpoint_dir） 中最新的检查点
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


testlist1 =[['<start>','晚风','摇','树','树','还','挺','<end>']]
translate(testlist1)
testlist2 =[['<start>','蝉', '参禅', '唱', '知', '了', '自知', '了','<end>']]
translate(testlist2)
testlist3 =[['<start>','堤', '是', '土', '夯', '就','<end>']]
translate(testlist3)
testlist4 =[['<start>','晨露', '润', '花', '红','<end>']]
translate(testlist4)
