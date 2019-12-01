# -*- coding:utf-8 -*-

'''
Author: Haoran Chen
Date: 2018-05-02
'''


import tensorflow as tf
import tensorflow.keras.layers as layers

NUM_FILTERS = 32
CONV_SIZE = 2
NUM_CHANNELS = 3
NUM_OUTPUT = 10
RATIO = [1.0, 1.4142135623730951, 1.4142135623730951, 1.0]
ALPHA = 0.02
PSEUDO_RANDOM = True
OVERLAPPING = True


def single_conv_layer(input_tensor, num_filters, pool_flag, name):
    
    with tf.variable_scope(name):
        conv = layers.Conv2D(num_filters, 2, activation=tf.nn.relu)
        x = conv(input_tensor)
        if bool_flag:
            x, row_sq, col_sq = tf.nn.fractional_max_pool(
                    x, RATIO, PSEUDO_RANDOM, OVERLAPPING)
        print(x.name, x.get_shape())
    return x


def inference(input_tensor):
    x = single_conv_layer(input_tensor, 32, True, 'conv1')
    x = single_conv_layer(x, 32*2, True, 'conv2')
    x = single_conv_layer(x, 32*3, True, 'conv3')
    x = single_conv_layer(x, 32*4, True, 'conv4')
    x = single_conv_layer(x, 32*5, True, 'conv5')
    x = single_conv_layer(x, 32*6, True, 'conv6')
    x = single_conv_layer(x, 32*6, False, 'conv7')
    conv = layers.Conv2D(10, 1)
    x = conv(x) 

    return x

