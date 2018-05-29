from nltk.corpus import stopwords
from random import randint
import math 
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
import tensorflow as tf

def bool_mask_pair(batch_size, pair):
    mask1 = [False for i in range(batch_size)]
    mask2 = [False for i in range(batch_size)]
    for i in range(len(pair)):
        mask1[pair[i][0]] = True
        mask2[pair[i][1]] = True
    return [tf.constant(mask1),tf.constant(mask2)]

def extract_keywords(C, label, n_label):
    ret = []
    KW = np.zeros((label.shape[0],5))
    for i in range(C.shape[0]):
        w =0
        S = np.sort(C[i])[-5:]
        k=0
        for j in range(C[i].shape[0]):
            if C[i][j] in S and k < 5:
                KW[i][k] = j
                k+=1
        for w in range(3):
            if KW[i][w] not in ret:
                ret.append(KW[i][w])
    return np.array(ret)
                
def bool_mask_lex(batch_size):
    idx = np.arange(batch_size)
    m1 = [(i in idx) for i in range(batch_size*2)]
    print m1
    mask1 = tf.constant(m1)
    idx2 = [idx[i]+batch_size for i in range(batch_size)]
    m2 = [(i in idx2) for i in range(batch_size*2)]
    
    mask2 = tf.constant()
    i1 = 0
    i2 = 0
    return mask1, mask2

def mask_prime(X, KW):
    X_prime = np.zeros(X.shape[0])
    for i in range(KW.shape[0]):
        X_prime[KW[i]]=X[KW[i]]
    return X_prime

def extract_pair(labels, nb_pair, idx):
    pair_ml = []
    pair_cl = []
    b1 = True
    b2 = True
    n = labels.shape[0]
    while nb_pair > 0 or b1 or b2:
        tmp = np.random.randint(0, n-1)
        i = idx[tmp]
        j = i
        while j == i:
            tmp = np.random.randint(0, n-1)
            j = idx[tmp]
        if labels[i] == labels[j]:
            pair_ml.append(np.array([i,j]))
            b1 = False
        else:
            b2 = False
            pair_cl.append(np.array([i,j]))
        nb_pair-=1
    return pair_ml, pair_cl

def next_batch(num, data, KW):
    """
    Return a total of `num` random samples.
    """
    indices = np.arange(0, data.shape[0])
    np.random.shuffle(indices)
    indices = indices[:num]
    batch_data = np.asarray([data[i, :] for i in indices])
    batch_lex = []
    for i in range(batch_data.shape[0]):
        batch_lex.append(mask_prime(batch_data[i], KW))
    return indices, np.concatenate((batch_data, np.array(batch_lex)), axis=0)

###################################################
# Vectorize corpus with word frequency            #
#                                                 #
# data         text corpus                        #
###################################################
def word_to_index(data):
    v = Counter()
    nb_doc = 0
    stop = set(stopwords.words('english'))
    for text in data:
        nb_doc+=1
        for w in text:
            if w not in stop:
                v[w.lower()]+=1
    index = {}
    for i,w in enumerate(v):
        index[w.lower()]=i
        
    corpus = np.zeros((nb_doc, len(v)))
    j=0
    for text in data:
        for w in text:
            if w not in stop:
                corpus[j][index[w.lower()]]+=1
        j+=1
    return corpus

###################################################
# Compute wid                                     #
#                                                 #
# k            index of word                      #
# doc          document                           #
###################################################
def wid(k, doc):
    sum_ = 0
    for i in range(doc.shape[0]):
        sum_ += doc[i]
    return float(doc[k]) / float(sum_)
        

###################################################
# Compute df                                      #
#                                                 #
# k            index of word                      #
# data         corpus                             #
###################################################
def df(k, data):
    res = 0
    for i in range(data.shape[0]):
        res += 1 if data[i][k]>0 else 0
    return res
        
###################################################
# Compute tfidf                                   #
#                                                 #
# corpus       the corpus                         #
###################################################
def TFIDFvectorize(corpus):
    data_TFIDFvectorized = np.zeros((corpus.shape[0], corpus.shape[1]))
    for i in range(corpus.shape[0]):
        for j in range(corpus.shape[1]):
            data_TFIDFvectorized[i][j] = wid(j, corpus[i])*\
                                         math.log1p(corpus.shape[0] / \
                                                     math.log1p(df(j, corpus)))
    return data_TFIDFvectorized
