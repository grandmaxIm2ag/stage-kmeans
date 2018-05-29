#!/usr/bin/env python
#-*- coding: utf-8 -*-

from sklearn.metrics.cluster import normalized_mutual_info_score
import math
import sys
import numpy as np
import numpy.linalg as lin
import subprocess
import random as rand
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from clustering import kmeans, COPKmeans
from utils import extract_pair
if __name__ == "__main__":
    dataset = load_iris()
    y = dataset.target #On recupert les clusters reels
    ML, CL = extract_pair(y, 1)
    C = dataset.data
    M = C[0].shape[0]
    N = C.shape[0]
    k = int(sys.argv[1])
    e = float(sys.argv[2])
    S = COPKmeans(C, k, e, CL, ML)
    #Projection des donnees dans le plan PCA
    pca = PCA(n_components=2)
    x_r = pca.fit(C).transform(C) #C apres application du PCA
    #Visualisation des clusters
    k = len(set(S)) #Nombre de cluster
    target_names = range(k) #Label des clusters obtenus
    plt.figure()
    c = ['red','green','blue', 'yellow', 'pink']
    for i, target_name in zip (range(k), target_names):
        plt.scatter(x_r[S == i, 0], x_r[S == i, 1], label=target_name, color=c[i])
    plt.legend(loc='best')
    plt.title("Partition_obtenu_par_k_means_sur_la_collection_Iris")
    plt.savefig(sys.argv[3])
    #Calcule nmi
    score = normalized_mutual_info_score(S, y)
    print score
