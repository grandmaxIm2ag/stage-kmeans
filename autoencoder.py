#-*- coding: utf-8 -*-
from __future__ import print_function 
import numpy as np
import random
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import utils
from tensorflow.python import debug as tf_debug

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

def loss_pair(layer, pair, batch_size):

    mask1 = tf.constant([False for i in range(batch_size)])
    mask2 = tf.constant([False for i in range(batch_size)])
    
    lay = tf.identity(layer)

    for elem in pair:
        m1, m2 = utils.bool_mask_pair(batch_size, np.array([elem]))
        mask1 = tf.concat([mask1, m1],0)
        mask2 = tf.concat([mask2, m2],0)
        lay = tf.concat([lay, layer], 0)

    l1 = tf.boolean_mask(lay, mask1)
    l2 = tf.boolean_mask(lay, mask2)
    return tf.norm(l1 - l2)
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
    def __init__(self, n,batch_size, n_hidden, n_encode, batch, KW, ML_pair, CL_pair):
        self.n = n
        self.n_hidden = n_hidden
        self.n_encode = n_encode
        self.batch = batch
        self.batch_size = batch_size
        self.ML = ML_pair
        self.CL = CL_pair
        self.KW = KW
    ###################################################
    # Initialize boolean mask                         #
    ###################################################
    def init_mask(self):
        self.mask_kw1, self.mask_kw2 = utils.bool_mask_lex(self.batch_size)

    ###################################################
    # Initialize placeholders                         #
    ###################################################
    def init_placeholder(self):
        self.X = tf.placeholder(tf.float32, shape=[None,self.n], name = 'X')
        
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


    ###################################################
    # Initialize loss functions                       #
    ###################################################
    def init_losses(self):
        self.losses = {
            'rec': tf.reduce_sum(tf.norm(tf.boolean_mask(self.X, self.mask_kw1) - \
                                         tf.boolean_mask(self.decode_layer, \
                                                         self.mask_kw1))),
            'lex': tf.reduce_sum(tf.norm(tf.boolean_mask(self.encode_layer, \
                                                         self.mask_kw1) - \
                                         tf.boolean_mask(self.encode_layer, \
                                                         self.mask_kw2))),
            'ML' : tf.reduce_sum(loss_pair(tf.boolean_mask(self.encode_layer, self.mask_kw1), self.ML, self.batch_size)),
            'CL' : tf.reduce_sum(tf.maximum(0.,5 - tf.norm(loss_pair(tf.boolean_mask(self.encode_layer, self.mask_kw1), self.CL, self.batch_size))))
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
                     minimize(self.losses['rec']+
                              self.losses['lex']+
                              self.losses['ML']+
                              self.losses['CL'])
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for e in range(1, epoches+1):
                sess.run(train_step, feed_dict={self.X: self.batch})
                if e % 100 == 0:
                    print ("epochs : %s" % str(e))
                    
                if e % 10 == 0:
                    _, l1 = sess.run([train_step, self.losses['rec']], \
                                     feed_dict={self.X: self.batch})
                    _, l2 = sess.run([train_step, self.losses['lex']], \
                                     feed_dict={self.X: self.batch})
                    _, l3 = sess.run([train_step, self.losses['ML']], \
                                     feed_dict={self.X: self.batch})
                    _, l4 = sess.run([train_step, self.losses['CL']], \
                                     feed_dict={self.X: self.batch})
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
