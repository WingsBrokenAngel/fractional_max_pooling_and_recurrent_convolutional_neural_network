# -*- coding:utf-8 -*-

'''
Author: Haoran Chen
Date: 2018-05-02
Date2: 2019-12-12
'''

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.datasets.cifar10 as cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.activations import relu
import tensorflow.keras.backend as K

NUM_FILTERS = 160
KERNEL_SIZE = 2
NUM_OUTPUT = 10
RATIO = [1.0, 2**(1/3), 2**(1/3), 1.0]
PSEUDO_RANDOM = True
OVERLAPPING = True


def fractional_max_pool(x):
    return tf.nn.fractional_max_pool(x, RATIO, PSEUDO_RANDOM, OVERLAPPING)[0]


class FMP:
    def __init__(self, FILTERS, WEIGHT_DECAY, DROP_RATE):
        self.weight_decay = WEIGHT_DECAY
        self.filters = FILTERS
        self.drop_rate = DROP_RATE

        config1 = {'padding': 'same', 'activation': tf.nn.relu, 
                    'kernel_regularizer': tf.keras.regularizers.l2(WEIGHT_DECAY), 
                    'kernel_size': KERNEL_SIZE}

        config2 = {'padding': 'valid', 'activation': tf.nn.relu, 
                    'kernel_regularizer': tf.keras.regularizers.l2(WEIGHT_DECAY), 
                    'kernel_size': KERNEL_SIZE}

        self.layer1 = layers.Conv2D(FILTERS, **config1)
        self.layer1_pool = layers.Lambda(fractional_max_pool)

        self.layer2 = layers.Conv2D(FILTERS*2, **config1)
        self.layer2_pool = layers.Lambda(fractional_max_pool)

        self.layer3 = layers.Conv2D(FILTERS*3, **config1)
        self.layer3_pool = layers.Lambda(fractional_max_pool)

        self.layer4 = layers.Conv2D(FILTERS*4, **config1)
        self.layer4_pool = layers.Lambda(fractional_max_pool)

        self.layer5 = layers.Conv2D(FILTERS*5, **config1)
        self.layer5_pool = layers.Lambda(fractional_max_pool)

        self.layer6 = layers.Conv2D(FILTERS*6, **config1)
        self.layer6_pool = layers.Lambda(fractional_max_pool)

        self.layer7 = layers.Conv2D(FILTERS*7, **config1)
        self.layer7_pool = layers.Lambda(fractional_max_pool)

        self.layer8 = layers.Conv2D(FILTERS*8, **config1)
        self.layer8_pool = layers.Lambda(fractional_max_pool)

        self.layer9 = layers.Conv2D(FILTERS*9, **config1)
        self.layer9_pool = layers.Lambda(fractional_max_pool)

        self.layer10 = layers.Conv2D(FILTERS*10, **config2)
        self.layer10_dp = layers.Dropout(self.drop_rate)

        self.layer11 = layers.Conv2D(10, kernel_size=1, activation='softmax')
        self.flatten = layers.Flatten()


    def __call__(self, imgs, train=True):
        x = self.layer1_pool(self.layer1(imgs))

        x = self.layer2_pool(self.layer2(x))

        x = self.layer3_pool(self.layer3(x))

        x = self.layer4_pool(self.layer4(x))

        x = self.layer5_pool(self.layer5(x))

        x = self.layer6_pool(self.layer6(x))

        x = self.layer7_pool(self.layer7(x))

        x = self.layer8_pool(self.layer8(x))

        x = self.layer9_pool(self.layer9(x))

        x = self.layer10_dp(self.layer10(x))

        y = self.layer11(x)
        y = self.flatten(y)
        return y
