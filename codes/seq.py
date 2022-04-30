import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from subword import *

num_words = 700
train_filename = 'C:\\Users\\86138\\Desktop\\文本分类\\train.csv'
train_df = pd.read_csv(train_filename)
del train_df['id']
del train_df['label']
X_train = train_df.values.reshape(8001,)
a = Subword(X_train, num_words, 8001)
X_str = a.get_str()
print(X_str[0])
a.get_tokens()
a.get_fina_token()
ls = a.get_ls()
print(ls)
X_train = get_sequence(X_str, 8001, ls)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=150)
df = pd.DataFrame(X_train)
print(len(a.tokens))
df.to_csv("X_train.csv", index=False)

train_filename = 'C:\\Users\\86138\\Desktop\\文本分类\\test.csv'
train_df = pd.read_csv(train_filename)
del train_df['id']
X_test = train_df.values.reshape(20001,)
b = Subword(X_test, num_words, 20001)
X_t_str = b.get_str()
X_test = get_sequence(X_t_str, 20001, ls)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=150)
df = pd.DataFrame(X_test)
df.to_csv("X_test.csv", index=False)
