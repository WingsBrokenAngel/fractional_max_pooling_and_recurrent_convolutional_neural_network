# -*- coding:utf-8 -*-

'''
Author: Haoran Chen
Date: 2018-05-02
'''


import tensorflow as tf
import tensorflow.keras.layers as layers
from config import Config

cfg = Config()
NUM_FILTERS = 160
CONV_SIZE = 2
NUM_CHANNELS = 3
NUM_OUTPUT = 10
RATIO = [1.0, 2**(1/3), 2**(1/3), 1.0]
PSEUDO_RANDOM = cfg.pseudo_random
OVERLAPPING = cfg.overlapping


def single_conv_layer(input_tensor, num_filters, pool_flag, name):
    
    with tf.variable_scope(name):
        conv = layers.Conv2D(num_filters, 2, activation=tf.nn.relu)
        x = conv(input_tensor)
        if pool_flag:
            x, row_sq, col_sq = tf.nn.fractional_max_pool(
                    x, RATIO, PSEUDO_RANDOM, OVERLAPPING)
        print(x.name, x.get_shape(), flush=True)
    return x


def inference(input_tensor, keep_rate):
    x = single_conv_layer(input_tensor, NUM_FILTERS, True, 'conv1')
    x = single_conv_layer(x, NUM_FILTERS*2, True, 'conv2')
    x = single_conv_layer(x, NUM_FILTERS*3, True, 'conv3')
    x = single_conv_layer(x, NUM_FILTERS*4, True, 'conv4')
    x = single_conv_layer(x, NUM_FILTERS*5, True, 'conv5')
    x = single_conv_layer(x, NUM_FILTERS*6, True, 'conv6')
    x = single_conv_layer(x, NUM_FILTERS*7, True, 'conv7')
    x = single_conv_layer(x, NUM_FILTERS*8, True, 'conv8')
    x = single_conv_layer(x, NUM_FILTERS*4, False, 'conv9')
    x = tf.nn.dropout(x, keep_rate)
    conv = layers.Conv2D(NUM_OUTPUT, 1)
    x = conv(x) 
    x = tf.squeeze(x)
    print('final', x.name, x.get_shape(), flush=True)
    return x

