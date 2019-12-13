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

KERNEL_SIZE = 2
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

        config1 = {'padding': 'same', 'activation': None, 
                    'kernel_regularizer': tf.keras.regularizers.l2(WEIGHT_DECAY), 
                    'kernel_size': KERNEL_SIZE}

        config2 = {'padding': 'valid', 'activation': None, 
                    'kernel_regularizer': tf.keras.regularizers.l2(WEIGHT_DECAY), 
                    'kernel_size': KERNEL_SIZE}

        self.relu = layers.Lambda(lambda x: relu(x))
        self.pool = layers.Lambda(fractional_max_pool)

        self.layer1 = layers.Conv2D(FILTERS, **config1)

        self.layer2 = layers.Conv2D(FILTERS*2, **config1)

        self.layer3 = layers.Conv2D(FILTERS*3, **config1)

        self.layer4 = layers.Conv2D(FILTERS*4, **config1)

        self.layer5 = layers.Conv2D(FILTERS*5, **config1)

        self.layer6 = layers.Conv2D(FILTERS*6, **config1)

        self.layer7 = layers.Conv2D(FILTERS*7, **config1)

        self.dp = layers.Dropout(self.drop_rate)
        self.flatten = layers.Flatten()
        self.layer8 = layers.Dense(100, activation='softmax')


    def __call__(self, imgs, train=True):
        x = self.pool(self.relu(
            layers.BatchNormalization()(self.layer1(imgs), training=train)))

        x = self.pool(self.relu(
            layers.BatchNormalization()(self.layer2(x), training=train)))

        x = self.pool(self.relu(
            layers.BatchNormalization()(self.layer3(x), training=train)))

        x = self.pool(self.relu(
            layers.BatchNormalization()(self.layer4(x), training=train)))

        x = self.pool(self.relu(
            layers.BatchNormalization()(self.layer5(x), training=train)))

        x = self.pool(self.relu(
            layers.BatchNormalization()(self.layer6(x), training=train)))

        x = self.pool(self.relu(
            layers.BatchNormalization()(self.layer7(x), training=train)))

        x = self.dp(self.flatten(x), training=train)

        y = self.layer8(x)
        return y
