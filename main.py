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

def load_syntetic_data():
    data = synt.syntetic(500,50,100, 15)
    return data.TFIDFvectorize(), data.labels_
#Main
if __name__ == "__main__":
    corpus, label = load_syntetic_data()
    pml, pcl = extract_pair(label, 100)
    autoencoder = au.Autoencoder(50, 12, 10, corpus, np.array([1]), pml, pcl)
    autoencoder.init_placeholder()
    autoencoder.init_weights()
    autoencoder.init_biases()
    autoencoder.init_layers()
    autoencoder.init_losses()
    autoencoder.train(1500, 0.01, [0.25, 0.25, 0.25, 0.25])
    autoencoder.plot_loss()
