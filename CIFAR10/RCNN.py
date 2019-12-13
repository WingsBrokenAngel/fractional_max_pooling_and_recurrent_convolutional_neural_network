'''
Author: Haoran Chen
Date: 12/10/2019
'''
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.datasets.cifar10 as cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.activations import relu
import tensorflow.keras.backend as K

class RCNN:
    def __init__(self, FILTERS, WEIGHT_DECAY, RATE):
        self.decay_rate = WEIGHT_DECAY
        self.filters = FILTERS
        self.rate = RATE

        config1 = {'padding': 'same', 'filters': FILTERS, 'kernel_size': 5, 
                    'kernel_regularizer': tf.keras.regularizers.l2(WEIGHT_DECAY)}

        config2 = {'filters':FILTERS, 'kernel_size': 3, 'padding': 'same', 
                    'kernel_regularizer': tf.keras.regularizers.l2(WEIGHT_DECAY)}

        self.relu = layers.Lambda(lambda x: relu(x))
        self.pool = layers.MaxPool2D(3, 2, 'same')

        self.layer1 = layers.Conv2D(**config1)

        self.layer2_forward = layers.Conv2D(**config2)
        self.layer2_recurrent = layers.Conv2D(**config2)

        self.layer3_forward = layers.Conv2D(**config2)
        self.layer3_recurrent = layers.Conv2D(**config2)

        self.layer4_forward = layers.Conv2D(**config2)
        self.layer4_recurrent = layers.Conv2D(**config2)

        self.layer5_forward = layers.Conv2D(**config2)
        self.layer5_recurrent = layers.Conv2D(**config2)
        self.layer5_gpool = layers.GlobalMaxPool2D()
        self.layer5_dp = layers.Dropout(self.rate)

        self.layer6_dense = layers.Dense(10, activation='softmax')


    def __call__(self, imgs, train=True, recur=3):
        x = self.layer1(imgs)
        x = layers.BatchNormalization()(x, training=train)
        x = self.pool(self.relu(x))

        x = self._recurrent_layer(
            x, self.layer2_forward, self.layer2_recurrent, 
            recur, self.pool, train)

        x = self._recurrent_layer(
            x, self.layer3_forward, self.layer3_recurrent, 
            recur, self.pool, train)

        x = self._recurrent_layer(
            x, self.layer4_forward, self.layer4_recurrent, 
            recur, self.pool, train)

        x = self._recurrent_layer(
            x, self.layer5_forward, self.layer5_recurrent, 
            recur, self.layer5_gpool, train)
        x = self.layer5_dp(x, training=train)

        y = self.layer6_dense(x)
        return y

    def _recurrent_layer(self, x, forward, recurrent, recur, pool, train):
        x_f = x
        x_r = x
        for step in range(1 + recur):
            if step == 0:
                x_r = self.relu(layers.BatchNormalization()(forward(x_f), training=train))
            else:
                x_r = self.relu(layers.BatchNormalization()(
                    layers.add([recurrent(x_r), forward(x_f)]), training=train))
        
        x = pool(x_r)
        return x