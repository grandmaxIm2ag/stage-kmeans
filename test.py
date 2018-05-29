import numpy as np
import random
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import utils
from tensorflow.python import debug as tf_debug

def loss_pair(layer, pair, batch_size):
    mask1 = tf.constant([False, False, False, False])
    mask2 = tf.constant([False, False, False, False])
    lay = tf.identity(layer)

    for elem in pair:
        m1, m2 = utils.bool_mask_pair(batch_size, np.array([elem]))
        mask1 = tf.concat([mask1, m1],0)
        mask2 = tf.concat([mask2, m2],0)
        lay = tf.concat([lay, layer], 0)

    l1 = tf.boolean_mask(lay, mask1)
    l2 = tf.boolean_mask(lay, mask2)

    return tf.norm(l1 - l2)

tab = tf.constant([[1.0,5.0,7.0],[2.0,9.0, 7.0],[7.0,8.0,3.0],[9.0,0.0,76.0]])
ml = [[0,1],[3,2]]
with tf.Session() as sess:
    print (sess.run(tf.reduce_sum(loss_pair(tab, ml, 4))))
