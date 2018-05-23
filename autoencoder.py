#-*- coding: utf-8 -*-

import numpy as np
import random
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

MARGIN = 2

###################################################
# Mask document X with keywords kw                #
#                                                 #
# X            document                           #
# KW           Index of keywords                  #
###################################################
def mask(X, KW):
    X_prime = np.zeros(X.shape[0])
    for i in range(KW.shape[0]):
        X_prime[KW[i]]=X[KW[i]]
    return X_prime

###################################################
# Generate pair if document                       #
#                                                 #
# X            corpus                             #
# KW           Index of pairs                     #
###################################################
def pairewise_const(X, P):
    P1 = []
    P2 = []
    for i in range(len(P)):
        P1.append(X[P[i][0]])
        P2.append(X[P[i][1]])
    return P1, P2

class Autoencoder:            
    ###################################################
    # Constructor of the class Autoencoder            #
    #                                                 #
    # n            size of input                      #
    # n_hidden     size of hidden layer               #
    # n_encode     size of encode layer               #
    # batch        batch for training                 #
    # KW           Index of keywords                  #
    # ML_pair      Index of similar pair              #
    # CL_pair      Index of disimilar pair            #
    ###################################################
    def __init__(self, n, n_hidden, n_encode, batch, KW, ML_pair, CL_pair):
        self.n = n
        self.n_hidden = n_hidden
        self.n_encode = n_encode
        self.batch = batch
        self.batch_KW1 = []
        for i in range(batch.shape[0]):
            self.batch_KW1.append(batch[i])
        self.batch_KW2 = []
        for i in range(batch.shape[0]):
            self.batch_KW2.append(mask(batch[i], KW))
        self.batch_ML1, self.batch_ML2 = pairewise_const(batch, ML_pair)
        self.batch_CL1, self.batch_CL2 = pairewise_const(batch, CL_pair) 

    ###################################################
    # Initialize placeholders                         #
    ###################################################
    def init_placeholder(self):
        self.X = tf.placeholder(tf.float32, shape=[None,self.n], name = 'X')
        self.X_KW1 = tf.placeholder(tf.float32, shape=[None,self.n], name = 'KW1')
        self.X_KW2 = tf.placeholder(tf.float32, shape=[None,self.n], name = 'KW2')
        self.X_ML1 = tf.placeholder(tf.float32, shape=[None,self.n], name = 'ML1')
        self.X_ML2 = tf.placeholder(tf.float32, shape=[None,self.n], name = 'ML2')
        self.X_CL1 = tf.placeholder(tf.float32, shape=[None,self.n], name = 'CL1')
        self.X_CL2 = tf.placeholder(tf.float32, shape=[None,self.n], name = 'CL2')
        
    ###################################################
    # Initialize atrix of weights in the map weights  #
    ###################################################
    def init_weights(self):
        with tf.name_scope('weights'):
            self.weights = {
                'h1': tf.Variable(tf.truncated_normal([self.n, self.n_hidden])\
                                  , name="h1"),
                'h2': tf.Variable(tf.truncated_normal\
                                  ([self.n_encode, self.n_hidden]),name="h2"),
                'encode': tf.Variable(tf.truncated_normal\
                                      ([self.n_hidden, self.n_encode]),\
                                      name="encode"),
                'decode': tf.Variable(tf.truncated_normal([self.n_hidden, self.n])\
                                      ,name="decode")
            }

    ###################################################
    # Initialize biais vectors                        #
    ###################################################
    def init_biases(self):
        with tf.name_scope('biases'):
            self.biases = {
                'b1': tf.Variable(tf.zeros([self.n_hidden]), name="b1"),
                'encode': tf.Variable(tf.zeros([self.n_encode]), \
                                      name="b_encode"),
                'b2': tf.Variable(tf.zeros([self.n_hidden]), name="b2"),
                'decode': tf.Variable(tf.zeros([self.n]), name="b_decode")
            }

    
    ###################################################
    # Initialize the layer for the placeholder X      #
    #                                                 #
    # X            Placeholder                        #
    ###################################################
    def init_net_layers(self, X):
        h1 = tf.nn.sigmoid(tf.add(tf.matmul\
                                  (X,\
                                   self.weights['h1']),\
                                  self.biases['b1']))
        enc = tf.nn.softplus(tf.add(tf.matmul\
                                    (h1,\
                                     self.weights['encode']),\
                                    self.biases['encode']))
        h2 = tf.nn.sigmoid(tf.add(tf.matmul\
                                  (enc,\
                                   self.weights['h2']),\
                                  self.biases['b2']))
        dec = tf.nn.softplus(tf.add(tf.matmul\
                                    (h2,\
                                     self.weights['decode']),\
                                    self.biases['decode']))
        return h1, h2, dec, enc
        
    ###################################################
    # Initialize layers                               #       
    ###################################################
    def init_layers(self):
        self.hidden1_layer,self.hidden1_layer, self.decode_layer, \
            self.encode_layer = self.init_net_layers(self.X)
        self.hidden1_layer_KW1,self.hidden2_layer_KW1, self.decode_layer_KW1, \
            self.encode_layer_KW1 = self.init_net_layers(self.X_KW1)
        self.hidden1_layer_KW2,self.hidden2_layer_KW2, self.decode_layer_KW2, \
            self.encode_layer_KW2 = self.init_net_layers(self.X_KW2)
        self.hidden1_layer_CL1,self.hidden2_layer_CL1, self.decode_layer_CL1, \
            self.encode_layer_CL1 = self.init_net_layers(self.X_ML1)
        self.hidden1_layer_CL2,self.hidden2_layer_CL2, self.decode_layer_CL2, \
            self.encode_layer_CL2 = self.init_net_layers(self.X_ML2)
        self.hidden1_layer_ML1,self.hidden2_layer_ML1, self.decode_layer_ML1, \
            self.encode_layer_ML1 = self.init_net_layers(self.X_CL1)
        self.hidden1_layer_CL2,self.hidden2_layer_ML2, self.decode_layer_ML2, \
            self.encode_layer_ML2 = self.init_net_layers(self.X_CL2)
        
    ###################################################
    # Initialize loss functions                       #
    ###################################################
    def init_losses(self):
        self.losses = {
            'rec': tf.reduce_sum(tf.norm(self.X - self.decode_layer)),
            'lex': tf.reduce_sum(tf.norm(self.encode_layer_KW1 - \
                                         self.encode_layer_KW2)),
            'ML' : tf.reduce_sum(tf.norm(self.encode_layer_ML1 - \
                                         self.encode_layer_ML2)),
            'CL' : tf.reduce_sum(tf.maximum(0., MARGIN - \
                                            tf.norm(self.encode_layer_CL1 - \
                                                    self.encode_layer_CL2)))
        }

    ###################################################
    # Fonction d'apprentissage de l'autoencoder       #
    #                                                 #
    # epoches      number of epoches                  #
    # rate         training rate                      #
    # alpha        hyperparameters alpha              #
    ###################################################
    def train(self, epoches, rate, alpha):
        self.loss_rec = []
        self.loss_lex = []
        self.loss_ml = []
        self.loss_cl = []
        self.ep = []
        train_step = tf.train.GradientDescentOptimizer(rate).\
                     minimize(alpha[0]*self.losses['rec']+
                              alpha[1]*self.losses['lex']+
                              alpha[2]*self.losses['ML']+
                              alpha[3]*self.losses['CL'])
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for e in range(1, epoches+1):
                sess.run(train_step, feed_dict={self.X: self.batch,
                                                self.X_KW1: self.batch_KW1,
                                                self.X_KW2: self.batch_KW2,
                                                self.X_ML1: self.batch_ML1,
                                                self.X_ML2: self.batch_ML2,
                                                self.X_CL1: self.batch_CL1,
                                                self.X_CL2: self.batch_CL2})
                if e % 100 == 0:
                    print "epochs : "+str(e)
                if e % 10 == 0:
                    _, l1 = sess.run([train_step, self.losses['rec']], \
                                     feed_dict={self.X: self.batch,
                                                self.X_KW1: self.batch_KW1,
                                                self.X_KW2: self.batch_KW2,
                                                self.X_ML1: self.batch_ML1,
                                                self.X_ML2: self.batch_ML2,
                                                self.X_CL1: self.batch_CL1,
                                                self.X_CL2: self.batch_CL2})
                    _, l2 = sess.run([train_step, self.losses['lex']], \
                                     feed_dict={self.X: self.batch,
                                                self.X_KW1: self.batch_KW1,
                                                self.X_KW2: self.batch_KW2,
                                                self.X_ML1: self.batch_ML1,
                                                self.X_ML2: self.batch_ML2,
                                                self.X_CL1: self.batch_CL1,
                                                self.X_CL2: self.batch_CL2})
                    _, l3 = sess.run([train_step, self.losses['ML']], \
                                     feed_dict={self.X: self.batch,
                                                self.X_KW1: self.batch_KW1,
                                                self.X_KW2: self.batch_KW2,
                                                self.X_ML1: self.batch_ML1,
                                                self.X_ML2: self.batch_ML2,
                                                self.X_CL1: self.batch_CL1,
                                                self.X_CL2: self.batch_CL2})
                    _, l4 = sess.run([train_step, self.losses['CL']], \
                                     feed_dict={self.X: self.batch,
                                                self.X_KW1: self.batch_KW1,
                                                self.X_KW2: self.batch_KW2,
                                                self.X_ML1: self.batch_ML1,
                                                self.X_ML2: self.batch_ML2,
                                                self.X_CL1: self.batch_CL1,
                                                self.X_CL2: self.batch_CL2})
                    self.loss_rec.append(l1)
                    self.loss_lex.append(l2)
                    self.loss_ml.append(l3)
                    self.loss_cl.append(l4)
                    self.ep.append(e)
                    
    def plot_loss(self, filename, title):
        plt.title(title)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.plot(self.ep, self.loss_cl, label = 'cl')
        plt.plot(self.ep, self.loss_ml, label = 'ml')
        plt.plot(self.ep, self.loss_rec, label = 'rec')
        plt.plot(self.ep, self.loss_lex, label = 'lex')
        plt.legend()
        plt.savefig(filename, bbox_inches='tight')
