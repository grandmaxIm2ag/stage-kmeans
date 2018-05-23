from random import randint
import numpy as np
import PriorityQueue as pq

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

def extract_keywords_from_vector(doc, nb_kw):
    res = np.zeros(nb_kw)
    i = 0
    prio = pq.PriorityQueueMax()
    for e in doc:
        prio.push(i, e)
        i+=1
    for i in range(nb_kw):
        res[i] = prio.pop()
    return res[i]

def extract_keywords_from_corpus(corpus, labels, label, nb_kw):
    doc = np.zeros(corpus[0].shape[0])
    k = 0
    for i in range(labels.shape[0]):
        if(label == labels[i]):
            k+=1
            doc += corpus[i]
    doc /= k
    return extract_keywords_from_vector(doc, nb_kw)    
