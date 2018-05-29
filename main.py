#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import random
import math
import autoencoder as au
import matplotlib.pyplot as plt
import sys
import utils
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import PlaintextCorpusReader

###################################################
# Load data for clustering                        #
###################################################
def load_data():
    corpus_root = "/media/maxence/SD_MAXENCE/cours/m1Informatique/S2/stage/kmeans_lexical_ml_cl/resources/text"
    wordlists = PlaintextCorpusReader(corpus_root, ".*")
    brut_data = []
    label = np.zeros(len(wordlists.fileids()))
    i=0
    for filename in wordlists.fileids():
        brut_data.append(wordlists.words(filename))
        label[i] = random.randint(0,1)#### !!!!!!!!!!
        i+=1
        corpus = utils.TFIDFvectorize(utils.word_to_index(brut_data))
    return corpus, label    

#Main
if __name__ == "__main__":
    corpus, label = load_data()
    n = corpus.shape[1]
    batch_size = corpus.shape[0]
    kw = utils.extract_keywords(corpus,label,2)
    print kw
    idx, corpus = utils.next_batch(batch_size, corpus, np.array([1]))
    pml, pcl = utils.extract_pair(label, 50, idx)
    autoencoder = au.Autoencoder(n, batch_size, 200, 300, corpus, \
                                 kw, pml, pcl)
    autoencoder.init_placeholder()
    autoencoder.init_weights()
    autoencoder.init_biases()
    autoencoder.init_layers()
    autoencoder.init_mask()
    autoencoder.init_losses()
    autoencoder.train(1500, 0.01, [float(sys.argv[2]), float(sys.argv[3]), \
                                   float(sys.argv[4])])
    autoencoder.plot_loss(sys.argv[5], \
                          "Variation des losses : a0= %s a1= %s a2= %s "\
                          % (sys.argv[2],sys.argv[3],sys.argv[4]))
