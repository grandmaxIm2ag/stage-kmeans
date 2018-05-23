from random import randint
import math 
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from collections import Counter


###################################################
# Extract pairs                                   #
#                                                 #
# nb_pair      the number of pair                 #
# label        set of label                       #
###################################################
def extract_pair(labels, nb_pair):
    pair_ml = []
    pair_cl = []
    b1 = True
    b2 = True
    n = labels.shape[0]
    while nb_pair > 0 or b1 or b2:
        i = randint(0, n-1)
        j = i
        while j == i:
            j = randint(0,n-1)
        if labels[i] == labels[j]:
            pair_ml.append(np.array([i,j]))
            b1 = False
        else:
            b2 = False
            pair_cl.append(np.array([i,j]))
        nb_pair-=1
    return pair_ml, pair_cl
        
###################################################
# Vectorize corpus with word frequency            #
#                                                 #
# data         text corpus                        #
###################################################
def word_to_index(data):
    v = Counter()
    nb_doc = 0
    for text in data:
        nb_doc+=1
        for w in text:
            v[w.lower()]+=1
    index = {}
    for i,w in enumerate(v):
        index[w.lower()]=i
        
    corpus = np.zeros((nb_doc, len(v)))
    j=0
    for text in data:
        for w in text:
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
