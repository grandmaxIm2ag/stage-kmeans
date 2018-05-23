#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import random
import math
import autoencoder as au
import matplotlib.pyplot as plt
import sys
import syntetic_data as synt
from utils import extract_pair
from sklearn.metrics.cluster import normalized_mutual_info_score

def load_syntetic_data():
    data = synt.syntetic(100,50,100, 5)
    return data.TFIDFvectorize(), data.labels_

#Main
if __name__ == "__main__":
    corpus, label = load_syntetic_data()
    pml, pcl = extract_pair(label, int(sys.argv[1]))
    autoencoder = au.Autoencoder(50, 12, 10, corpus, np.array([1]), pml, pcl)
    autoencoder.init_placeholder()
    autoencoder.init_weights()
    autoencoder.init_biases()
    autoencoder.init_layers()
    autoencoder.init_losses()
    autoencoder.train(1500, 0.01, [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])])
    autoencoder.plot_loss(sys.argv[6], "Variation des losses : a0= %s a1= %s a2= %s a3= %s " % (sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]))
