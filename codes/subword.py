import collections
import re
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


class Subword:
    """字节对编码"""

    def __init__(self, array, num_words, size):
        """初始化"""
        self.array = array
        self.num_words = num_words
        self.str = []
        self.size = size
        self.vocab = collections.defaultdict(int)
        self.tokens = collections.defaultdict(int)

    def get_str(self):
        """数据清洗"""
        stopwords = [' for ', ' the ', ' that ', ' is ', ' there ', ' what ', ' and ', ' do ', ' i ', ' you ',
                     ' she ', ' he ', ' they ', ' them ', ' these ', ' those ', ' are ', ' was ', ' were ', ' who ',
                     ' why ', ' to ', ' on ', ' in ', ' with ', ' which ', ' your ', ' her ', ' his ', ' my ']
        stop = ' where '
        for stopword in stopwords:
            stop = stop+'|'+stopword
        for i in range(self.size):
            self.str.append(self.array[i].lower())
            self.str[i] = re.sub('[\\\~!@#$%^&*()<>,./?\\{}\[\]_+-=`\"\'|]', ' ', self.str[i])
            self.str[i] = re.sub('\d+', ' ', self.str[i])
            self.str[i] = re.sub(' s ', ' ', self.str[i])# 删除单独的s
            self.str[i] = re.sub(' a ', ' ', self.str[i])
            self.str[i] = re.sub(stop, ' ', self.str[i])
            self.str[i] = self.str[i].lstrip()
            self.str[i] = re.sub('\s+', ' ', self.str[i])
            for word in self.str[i].strip().split():
                self.vocab[' '.join(list(word)) + ' </w>'] += 1
            self.str[i] = re.sub(' ', '</w> ', self.str[i])
        return self.str

    def get_tokens(self):
        """建立词表"""
        for key, value in self.vocab.items():
            word_tokens = key.split()
            for token in word_tokens:
                self.tokens[token] += value
        return self.tokens

    def get_pairs(self):
        """字符配对"""
        pairs = collections.defaultdict(int)
        for key, value in self.vocab.items():
            c = key.split()
            for i in range(len(c) - 1):
                pairs[(c[i], c[i + 1])] += value
        return pairs

    def merge(self, pairs):
        """合并配对字符"""
        best = max(pairs, key=pairs.get)
        new_token = ''.join(best)
        # 增加新的token
        self.tokens[new_token] = pairs[best]
        # 原来的token减去合并token的数量
        self.tokens[best[0]] -= self.tokens[new_token]
        self.tokens[best[1]] -= self.tokens[new_token]
        if self.tokens[best[0]] == 0:
            del self.tokens[best[0]]
        if self.tokens[best[1]] == 0:
            del self.tokens[best[1]]
        new_vocab = collections.defaultdict(int)
        for key, value in self.vocab.items():
            new_key = ' '+key+' '
            pair1 = ' '+best[0]+' '+best[1]+' '
            pair2 = ' '+new_token+' '
            new_key = re.sub(pair1, pair2, new_key)
            new_key.strip()
            new_vocab[new_key] = self.vocab[key]
        self.vocab = new_vocab
        return self.tokens

    def get_fina_token(self):
        while len(self.tokens) < self.num_words:
            self.merge(self.get_pairs())
        return self.tokens

    def get_ls(self):
        ls = []
        # 获得索引
        while len(self.tokens) > 0:
            ls.append(max(self.tokens, key=lambda x: len(x)))#长度最大的token
            del self.tokens[max(self.tokens, key=lambda x: len(x))]
            print(len(ls))
        return ls


def get_sequence(lss, size, ls):
    squ = []
    for i in range(size):
        for t in range(len(ls)):
            lss[i] = re.sub(ls[t], ' ' + str(t+1) + ' ', lss[i])
        # lss[i] = re.sub('[^[1-9\s]]', ' ' + str(0) + ' ', lss[i])
        squ.append(lss[i].split())
    for i in range(size):
        for t in range(len(squ[i])):
            squ[i][t] = int(squ[i][t])
    return squ








