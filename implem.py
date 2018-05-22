#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import random
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

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
        
class Autoencoder:    
    ###################################################
    # Constructeur de la classse Autoencoder          #
    #                                                 #
    # n            La taille de l'entrée              #
    # n_hidden     La taille de la couche cachée      #
    # n_encode     La taille de la couche d'encodage  #
    # batch        Jeu de données pour l'apprentissage#
    ###################################################
    def __init__(self, n, n_hidden, n_encode, batch, KW):
        self.n = n
        self.n_hidden = n_hidden
        self.n_encode = n_encode
        self.batch = batch
        self.batch_prime = []
        for i in range(batch.shape[0]):
            self.batch_prime.append(mask(batch[i], KW))

    ###################################################
    # Initialise les placeholders X et Y              #
    ###################################################
    def init_placeholder(self):
        self.X = tf.placeholder(tf.float32, shape=[None,self.n], name = 'X')
        self.X_prime = tf.placeholder(tf.float32, shape=[None,self.n], name = 'X_prime')

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


    ###################################################
    # Initialise les couches du réseaux               #       
    ###################################################
    def init_layers(self):
        self.hidden1_layer = tf.nn.sigmoid(tf.add(tf.matmul\
                                                  (self.X,\
                                                   self.weights['h1']),\
                                                  self.biases['b1']))
        self.encode_layer = tf.nn.softplus(tf.add(tf.matmul\
                                                 (self.hidden1_layer,\
                                                  self.weights['encode']),\
                                                 self.biases['encode']))
        self.hidden2_layer = tf.nn.sigmoid(tf.add(tf.matmul\
                                                  (self.encode_layer,\
                                                   self.weights['h2']),\
                                                  self.biases['b2']))
        self.decode_layer = tf.nn.softplus(tf.add(tf.matmul\
                                                  (self.hidden2_layer,\
                                                   self.weights['decode']),\
                                                  self.biases['decode']))
        self.hidden1_layer_prime = tf.nn.sigmoid(tf.add(tf.matmul\
                                                  (self.X_prime,\
                                                   self.weights['h1']),\
                                                  self.biases['b1']))
        self.encode_layer_prime = tf.nn.softplus(tf.add(tf.matmul\
                                                 (self.hidden1_layer_prime,\
                                                  self.weights['encode']),\
                                                 self.biases['encode']))
        self.hidden2_layer_prime = tf.nn.sigmoid(tf.add(tf.matmul\
                                                  (self.encode_layer_prime,\
                                                   self.weights['h2']),\
                                                  self.biases['b2']))
        self.decode_layer_prime = tf.nn.softplus(tf.add(tf.matmul\
                                                  (self.hidden2_layer_prime,\
                                                   self.weights['decode']),\
                                                  self.biases['decode']))

    ###################################################
    # Initialise les fonctions de coûts :             #
    #   - reconstruction                              #
    #   - sparse penalties*                           #
    ###################################################
    def init_losses(self):
        self.losses = {
            'rec': tf.reduce_sum(tf.pow(self.X - self.decode_layer, 2)),
            'lex': tf.reduce_sum(tf.pow(self.encode_layer - self.encode_layer_prime, 2))
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
        self.ep = []
        train_step = tf.train.GradientDescentOptimizer(rate).\
                     minimize(hyperparam[0]*self.losses['rec']+
                              hyperparam[1]*self.losses['lex'])
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for e in range(1, epoches+1):
                sess.run(train_step, feed_dict={self.X: self.batch,
                                                self.X_prime: self.batch_prime})
                if e % 10 == 0:
                    _, x = sess.run([train_step, self.losses['rec']], feed_dict={self.X: self.batch,
                                                self.X_prime: self.batch_prime})
                    _, y = sess.run([train_step, self.losses['lex']], feed_dict={self.X: self.batch,
                                                self.X_prime: self.batch_prime})
                    self.loss_rec.append(x)
                    self.loss_lex.append(y)
                    self.ep.append(e)
            print "epoch : "+str(e)
            pred = sess.run(self.encode_layer,feed_dict=\
                            {self.X: self.batch,
                             self.X_prime: self.batch_prime})
            pred2 = sess.run(self.encode_layer_prime,feed_dict=\
                             {self.X: self.batch,
                              self.X_prime: self.batch_prime})
            print "Estimation encode : \n"+str(pred)+"\n"+str(pred2)
            pred = sess.run(self.decode_layer,feed_dict=\
                            {self.X: self.batch})
            pred2 = sess.run(self.decode_layer_prime,feed_dict=\
                             {self.X_prime: self.batch_prime})
            print "Estimation decode : \n"+str(pred)+"\n"+str(pred2)
            print str(self.ep)
    def plot_loss(self):
        plt.title('Variation des loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.plot(self.ep, self.loss_rec, label = 'rec')
        plt.plot(self.ep, self.loss_lex, label = 'lex')
        plt.legend()
        plt.savefig('rec.png', bbox_inches='tight')
