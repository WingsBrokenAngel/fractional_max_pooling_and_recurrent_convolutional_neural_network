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

NUM_FILTERS = 96
KERNEL_SIZE = 3
NUM_OUTPUT = 10


class CONVNET:
    def __init__(self, FILTERS, WEIGHT_DECAY, DROP_RATE):
        self.weight_decay = WEIGHT_DECAY
        self.filters = FILTERS
        self.drop_rate = DROP_RATE

        config1 = {'padding': 'same', 'activation': None, 'kernel_size': KERNEL_SIZE, 
                   'kernel_regularizer': tf.keras.regularizers.l2(WEIGHT_DECAY)}

        config2 = {'padding': 'valid', 'activation': tf.nn.relu, 
                    'kernel_size': KERNEL_SIZE}

        self.relu = layers.Lambda(lambda x: relu(x))
        self.pool = layers.MaxPool2D()

        self.layer1 = layers.Conv2D(FILTERS, **config1)

        self.layer2 = layers.Conv2D(FILTERS*2, **config1)

        self.layer3 = layers.Conv2D(FILTERS*3, **config1)

        self.layer4 = layers.Conv2D(FILTERS*4, **config1)

        self.layer5 = layers.Conv2D(FILTERS*5, **config1)
        self.flatten = layers.Flatten()
        self.dp = layers.Dropout(self.drop_rate)
        self.layer6 = layers.Dense(10, activation='softmax')


    def __call__(self, imgs, train=True):
        x = self.pool(self.relu(layers.BatchNormalization()(self.layer1(imgs))))

        x = self.pool(self.relu(layers.BatchNormalization()(self.layer2(x))))

        x = self.pool(self.relu(layers.BatchNormalization()(self.layer3(x))))

        x = self.pool(self.relu(layers.BatchNormalization()(self.layer4(x))))

        x = self.pool(self.relu(layers.BatchNormalization()(self.layer5(x))))
        x = self.dp(self.flatten(x), training=train)
        y = self.layer6(x)

        return y