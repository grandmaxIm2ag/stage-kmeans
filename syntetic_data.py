#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
from random import randint
from math import log
from sklearn.cluster import KMeans

def wid(k, doc):
    sum_ = 0
    for i in range(doc.shape[0]):
        sum_ += doc[i]
    return float(doc[k]) / float(sum_) 
def df(k, data):
    res = 0
    for i in range(data.shape[0]):
        res += 1 if data[i][k]>0 else 0
    return res
                
class syntetic:
    def __init__(self, n, m, max_value, nb_label):
        self.data_ = np.zeros((n, m))
        for i in range (n):
            for j in range(m):
                self.data_[i][j] = randint(0, max_value)
        kmeans = KMeans(n_clusters=nb_label, random_state=0).fit(self.data_)
        self.labels_ = kmeans.labels_
        
    def TFIDFvectorize(self):
        data_TFIDFvectorized = np.zeros((self.data_.shape[0], self.data_.shape[1]))
        for i in range(self.data_.shape[0]):
            for j in range(self.data_.shape[1]):
                data_TFIDFvectorized[i][j] = wid(j, self.data_[i])*log(self.data_.shape[0] / log(df(j, self.data_)))
        return data_TFIDFvectorized
