from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import plot_model
import tensorflow.keras
from linear import Linear

# 导入数据
np.random.seed(10)
train_filename = 'train.csv'
train_df = pd.read_csv(train_filename)
label_df = train_df['label']
label = label_df.values
del train_df['label']
del train_df['id']
X_train = train_df.values.reshape(8001,)
Y_train = to_categorical(label, 10)

vocab_size = 1500  # 词表大小
embedding_dim = 256  # 词向量维度
max_len = 200  # 样本单词长度

X_train = np.loadtxt(open("X_train.csv"), delimiter=',', skiprows=1)
# 搭建网络
inputs = Input(shape=(max_len,), dtype='float64')
embed = layers.Embedding(input_dim=vocab_size+1, output_dim=embedding_dim, input_length=max_len)
emb = embed(inputs)
cnn1 = layers.Conv1D(256, 5, activation='relu')(emb)
cnn1 = layers.MaxPooling1D(pool_size=max_len-5+1)(cnn1)
cnn2 = layers.Conv1D(256, 4, activation='relu')(emb)
cnn2 = layers.MaxPooling1D(pool_size=max_len-4+1)(cnn2)
cnn3 = layers.Conv1D(256, 3, activation='relu')(emb)
cnn3 = layers.MaxPooling1D(pool_size=max_len-3+1)(cnn3)
cnn = layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
flat = layers.Flatten()(cnn)
drop = layers.Dropout(0.4)(flat)
outputs = Linear(10, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.01))(drop)
model = tensorflow.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
model.fit(X_train, Y_train, batch_size=32, epochs=1, verbose=1)

# 预测结果
test = np.loadtxt(open("X_test.csv"), delimiter=',', skiprows=1)
predict = model.predict(test)
predict = np.argmax(predict, axis=1)
ids = np.arange(0, 20001)
submission = pd.DataFrame({'id': ids, 'label': predict})
submission.to_csv("submission.csv", index=0)
