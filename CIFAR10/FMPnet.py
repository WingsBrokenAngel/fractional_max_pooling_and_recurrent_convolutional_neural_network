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
from config import Config

NUM_FILTERS = 160
KERNEL_SIZE = 3
NUM_OUTPUT = 10
RATIO = [1.0, 2**(1/3), 2**(1/3), 1.0]
PSEUDO_RANDOM = True
OVERLAPPING = True


def fractional_max_pool(x):
    return tf.nn.fractional_max_pool(x, RATIO, PSEUDO_RANDOM, OVERLAPPING)


class FMP:
    def __init__(self, FILTERS, WEIGHT_DECAY, DROP_RATE):
        self.weight_decay = WEIGHT_DECAY
        self.filters = FILTERS
        self.drop_rate = DROP_RATE

        config = {'padding': 'same', 'activation': tf.nn.relu, 
                    'kernel_regularizer': tf.keras.regularizers.l2(WEIGHT_DECAY), 
                    'kernel_size': KERNEL_SIZE}

        self.layer1 = layers.Conv2D(FILTERS, **config1)
        self.layer1_pool = layers.Lambda(fractional_max_pool)
        self.layer1_dp = layers.Dropout(self.drop_rate)

        self.layer2 = layers.Conv2D(FILTERS*2, **config1)
        self.layer2_pool = layers.Lambda(fractional_max_pool)
        self.layer2_dp = layers.Dropout(self.drop_rate)

        self.layer3 = layers.Conv2D(FILTERS*3, **config1)
        self.layer3_pool = layers.Lambda(fractional_max_pool)
        self.layer3_dp = layers.Dropout(self.drop_rate)

        self.layer4 = layers.Conv2D(FILTERS*4, **config1)
        self.layer4_pool = layers.Lambda(fractional_max_pool)
        self.layer4_dp = layers.Dropout(self.drop_rate)

        self.layer5 = layers.Conv2D(FILTERS*5, **config1)
        self.layer5_pool = layers.Lambda(fractional_max_pool)
        self.layer5_dp = layers.Dropout(self.drop_rate)

        self.layer6 = layers.Conv2D(FILTERS*6, **config1)
        self.layer6_pool = layers.Lambda(fractional_max_pool)
        self.layer6_dp = layers.Dropout(self.drop_rate)

        self.layer7 = layers.Conv2D(FILTERS*7, **config1)
        self.layer7_pool = layers.Lambda(fractional_max_pool)
        self.layer7_dp = layers.Dropout(self.drop_rate)


        self.layer8 = layers.Conv2D(FILTERS*8, **config1)
        self.layer8_dp = layers.Dropout(self.drop_rate)

        self.layer9 = layers.Conv2D(10, kernel_size=1, activation='softmax')


    def __call__(self, imgs, train=True):
        x = self.layer1_dp(self.layer1_pool(self.layer1(x)), training=train)

        x = self.layer2_dp(self.layer2_pool(self.layer2(x)), training=train)

        x = self.layer3_dp(self.layer3_pool(self.layer3(x)), training=train)

        x = self.layer4_dp(self.layer4_pool(self.layer4(x)), training=train)

        x = self.layer5_dp(self.layer5_pool(self.layer5(x)), training=train)

        x = self.layer6_dp(self.layer6_pool(self.layer6(x)), training=train)

        x = self.layer7_dp(self.layer7_pool(self.layer7(x)), training=train)

        x = self.layer8_dp(self.layer8(x), training=train)

        y = self.layer9(x)

        return y


if __name__ == "__main__":
    tf.app.flags.DEFINE_string('name', 'fmp', 'name of model')
    tf.app.flags.DEFINE_float('lr', 0.001, 'Learning rate of the model')
    tf.app.flags.DEFINE_float('drop', 0.05, 'Drop rate for dropout layers')
    tf.app.flags.DEFINE_integer('filters', 160, 'Filter number')
    tf.app.flags.DEFINE_float('wdecay', 1e-4, 'Weight decay rate')
    flags = tf.app.flags.FLAGS

    data = cifar10.load_data()
    (train_data, train_labels), (test_data, test_labels) = data
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    train_datagen = ImageDataGenerator(
        rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, 
        zoom_range=0.1, horizontal_flip=True, rescale=1/255.)
    train_generator = train_datagen.flow(
        train_data[:-5000], train_labels[:-5000], batch_size=128)

    val_datagen = ImageDataGenerator(rescale=1/255.)
    val_generator = val_datagen.flow(train_data[-5000:], 
        train_labels[-5000:], batch_size=128, shuffle=False)
    
    test_datagen = ImageDataGenerator(rescale=1/255.)
    test_generator = test_datagen.flow(
        test_data, test_labels, batch_size=128, shuffle=False)

    fmp = FMP(flags.filters, flags.wdecay, flags.drop)
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor_train = fmp(input_tensor, True)
    output_tensor_test = fmp(input_tensor, False)
    train_model = tf.keras.Model(input_tensor, output_tensor_train)
    test_model = tf.keras.Model(input_tensor, output_tensor_test)
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./model/%s-%d-%g-%g-%g-best.h5'%(
                flags.name, flags.filters, flags.lr, flags.drop, flag.wdecay), 
            monitor='val_acc', save_best_only=True), 
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_acc', factor=0.1, patience=7, min_lr=flags.lr/1000.)]
    train_model.summary()
    train_model.compile(
        optimizer=tf.keras.optimizers.SGD(0.001, 0.9, nesterov=True), 
        loss='categorical_crossentropy', metrics=['acc'])

    history = train_model.fit_generator(
        train_generator, epochs=128, 
        validation_data=val_generator, max_queue_size=128, workers=2, 
        callbacks=callbacks_list)
    
    train_model.load_weights('./model/%s-%d-%g-%g-%g-best.h5'%(
                flags.name, flags.filters, flags.lr, flags.drop, flags.wdecay))
    test_result = train_model.evaluate_generator(test_generator)
    print(test_result)

    test_model.load_weights('./model/%s-%d-%g-%g-%g-best.h5'%(
                flags.name, flags.filters, flags.lr, flags.drop, flags.wdecay))
    test_result = test_model.evaluate_generator(test_generator)
    print(test_result)
