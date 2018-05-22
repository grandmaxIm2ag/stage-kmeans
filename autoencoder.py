#-*- coding: utf-8 -*-

import numpy as np
import random
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

MARGIN = 2

###################################################
# Masque le document X avec les mot clés de KW    #
#                                                 #
# X            Le document                        #
# KW           Les indices des mots clés          #
###################################################
def mask(X, KW):
    X_prime = np.zeros(X.shape[0])
    for i in range(KW.shape[0]):
        X_prime[KW[i]]=X[KW[i]]
    return X_prime

def pairewise_const(X, P):
    P1 = []
    P2 = []
    for i in range(len(P)):
        P1.append(X[P[i][0]])
        P2.append(X[P[i][1]])
    return P1, P2

class Autoencoder:            
    ###################################################
    # Constructeur de la classse Autoencoder          #
    #                                                 #
    # n            La taille de l'entrée              #
    # n_hidden     La taille de la couche cachée      #
    # n_encode     La taille de la couche d'encodage  #
    # batch        Jeu de données pour l'apprentissage#
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
    # Initialise les placeholders X                   #
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
    # Initialise les matrices de poids dans le        #
    # dictionnaire weights                            #
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
    # Initialise les vecteurs de biais dans le        #
    # dictionnaire biases                             #
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
    # Initialise les couches du réseaux               #       
    ###################################################
    def init_layers(self):
        self.hidden1_layer,self.hidden1_layer, self.decode_layer, self.encode_layer = self.init_net_layers(self.X)
        self.hidden1_layer_KW1,self.hidden2_layer_KW1, self.decode_layer_KW1, self.encode_layer_KW1 = self.init_net_layers(self.X_KW1)
        self.hidden1_layer_KW2,self.hidden2_layer_KW2, self.decode_layer_KW2, self.encode_layer_KW2 = self.init_net_layers(self.X_KW2)
        self.hidden1_layer_CL1,self.hidden2_layer_CL1, self.decode_layer_CL1, self.encode_layer_CL1 = self.init_net_layers(self.X_ML1)
        self.hidden1_layer_CL2,self.hidden2_layer_CL2, self.decode_layer_CL2, self.encode_layer_CL2 = self.init_net_layers(self.X_ML2)
        self.hidden1_layer_ML1,self.hidden2_layer_ML1, self.decode_layer_ML1, self.encode_layer_ML1 = self.init_net_layers(self.X_CL1)
        self.hidden1_layer_CL2,self.hidden2_layer_ML2, self.decode_layer_ML2, self.encode_layer_ML2 = self.init_net_layers(self.X_CL2)
        
    ###################################################
    # Initialise les fonctions de coûts :             #
    #   - reconstruction                              #
    #   - sparse penalties*                           #
    ###################################################
    def init_losses(self):
        self.losses = {
            'rec': tf.reduce_sum(tf.pow(self.X - self.decode_layer, 2)),
            'lex': tf.reduce_sum(tf.pow(self.encode_layer_KW1 - self.encode_layer_KW2, 2)),
            'ML' : tf.reduce_sum(tf.pow(self.encode_layer_ML1 - self.encode_layer_ML2, 2)),
            'CL' : tf.maximum(0., MARGIN - tf.reduce_sum(tf.pow(self.encode_layer_CL1 - self.encode_layer_CL2, 2)))
        }

    ###################################################
    # Fonction d'apprentissage de l'autoencoder       #
    #                                                 #
    # epoches      Nombre d'epoches                   #
    # rate         Pas d'apprentissage                #
    ###################################################
    def train(self, epoches, rate, hyperparam):
        self.loss_rec = []
        self.loss_lex = []
        self.loss_ml = []
        self.loss_cl = []
        self.ep = []
        train_step = tf.train.GradientDescentOptimizer(rate).\
                     minimize(hyperparam[0]*self.losses['rec']+
                              hyperparam[1]*self.losses['lex']+
                              hyperparam[2]*self.losses['ML']+
                              hyperparam[3]*self.losses['CL'])
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
                    _, l1 = sess.run([train_step, self.losses['rec']], feed_dict={self.X: self.batch,
                                                                                 self.X_KW1: self.batch_KW1,
                                                                                 self.X_KW2: self.batch_KW2,
                                                                                 self.X_ML1: self.batch_ML1,
                                                                                 self.X_ML2: self.batch_ML2,
                                                                                 self.X_CL1: self.batch_CL1,
                                                                                 self.X_CL2: self.batch_CL2})
                    _, l2 = sess.run([train_step, self.losses['lex']], feed_dict={self.X: self.batch,
                                                                                 self.X_KW1: self.batch_KW1,
                                                                                 self.X_KW2: self.batch_KW2,
                                                                                 self.X_ML1: self.batch_ML1,
                                                                                 self.X_ML2: self.batch_ML2,
                                                                                 self.X_CL1: self.batch_CL1,
                                                                                 self.X_CL2: self.batch_CL2})
                    _, l3 = sess.run([train_step, self.losses['ML']], feed_dict={self.X: self.batch,
                                                                                 self.X_KW1: self.batch_KW1,
                                                                                 self.X_KW2: self.batch_KW2,
                                                                                 self.X_ML1: self.batch_ML1,
                                                                                 self.X_ML2: self.batch_ML2,
                                                                                 self.X_CL1: self.batch_CL1,
                                                                                 self.X_CL2: self.batch_CL2})
                    _, l4 = sess.run([train_step, self.losses['CL']], feed_dict={self.X: self.batch,
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
    def plot_loss(self):
        plt.title('Variation des loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.plot(self.ep, self.loss_cl, label = 'cl')
        plt.plot(self.ep, self.loss_ml, label = 'ml')
        plt.plot(self.ep, self.loss_rec, label = 'rec')
        plt.plot(self.ep, self.loss_lex, label = 'lex')
        plt.legend()
        plt.savefig('rec.png', bbox_inches='tight')
