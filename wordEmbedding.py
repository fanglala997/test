import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
import matplotlib.pyplot as plt
import os
import fileDispose

sentence ='晚风摇树树还挺，晨露润花花更红。原景天成无墨迹，万方乐奏有余阗。丹枫江冷人初去，绿柳堤新燕复来。'
shanglian_cut =fileDispose.getFile('shanglian_all_cut.json')
xialian_cut =fileDispose.getFile('xialian_all_cut.json')
test_shang = []
test_xia = []
test_shang.append('，')
test_xia.append('。')
for i in range(3):
    templist1 = shanglian_cut[i]
    templist2 = xialian_cut[i]
    for j in range(len(templist1)):
        test_shang.append(templist1[j])
        test_xia.append(templist2[j])
print(test_shang)

word2id ={}
for i in range(len(test_shang)+len(test_xia)):
    if(i<len(test_shang)):
        word2id.setdefault(test_shang[i],i)
    else:
        word2id.setdefault(test_xia[i-len(test_shang)],i)

training_set_scaled = []
for i in range(len(word2id)):
    training_set_scaled.append(i)

print(len(training_set_scaled))

x_train = []
y_train = []

for i in range(1,len(training_set_scaled)):
    x_train.append(training_set_scaled[i-1])
    y_train.append(training_set_scaled[i])

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

# 使x_train符合Embedding输入要求：[送入样本数， 循环核时间展开步数] ，
# 此处整个数据集送入所以送入，送入样本数为len(x_train)；输入4个字母出结果，循环核时间展开步数为4。
x_train = np.reshape(x_train, (len(x_train), 1))
y_train = np.array(y_train)

model = tf.keras.Sequential([
    Embedding(26, 2),
    SimpleRNN(10),
    Dense(26, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./Data/wordEmbedding.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='loss')

history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[cp_callback])

model.summary()

file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

################# predict ##################

preNum = int(input("input the number of test alphabet:"))
for i in range(preNum):
    alphabet1 = input("input test alphabet:")
    alphabet = word2id[alphabet1]
    # alphabet = [word2id[a] for a in alphabet1]
    # 使alphabet符合Embedding输入要求：[送入样本数， 时间展开步数]。
    # 此处验证效果送入了1个样本，送入样本数为1；输入4个字母出结果，循环核时间展开步数为4。
    alphabet = np.reshape(alphabet, (1, 1))
    result = model.predict([alphabet])
    pred = tf.argmax(result, axis=1)
    pred = int(pred)
    tf.print(alphabet1 + '->' + sentence[pred])







