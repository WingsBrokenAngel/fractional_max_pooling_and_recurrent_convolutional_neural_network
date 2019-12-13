# -*- coding:utf-8 -*-
'''
Author: Haoran Chen
Date: 2017-05-02
'''

from FMPnet import FMP
import numpy as np
from datetime import datetime
import os
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.datasets.cifar10 as cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.activations import relu
import tensorflow.keras.backend as K



if __name__ == "__main__":
    tf.app.flags.DEFINE_string('name', 'fmp', 'name of model')
    tf.app.flags.DEFINE_string('gpu', '9', 'gpu index')
    tf.app.flags.DEFINE_float('lr', 0.001, 'Learning rate of the model')
    tf.app.flags.DEFINE_float('drop', 0.5, 'Drop rate for dropout layers')
    tf.app.flags.DEFINE_integer('filters', 160, 'Filter number')
    tf.app.flags.DEFINE_float('wdecay', 1e-4, 'Weight decay rate')
    flags = tf.app.flags.FLAGS
    
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu

    data = cifar10.load_data()
    (train_data, train_labels), (test_data, test_labels) = data
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    train_datagen = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, 
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
                flags.name, flags.filters, flags.lr, flags.drop, flags.wdecay), 
            monitor='val_acc', save_best_only=True), 
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_acc', factor=0.5, patience=5, min_lr=flags.lr/1000.)]
    train_model.summary()
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(flags.lr), 
        loss='categorical_crossentropy', metrics=['acc'])
    test_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    history = train_model.fit_generator(
        train_generator, epochs=128, 
        validation_data=val_generator, max_queue_size=128, workers=2, 
        callbacks=callbacks_list)

    test_model.load_weights('./model/%s-%d-%g-%g-%g-best.h5'%(
                flags.name, flags.filters, flags.lr, flags.drop, flags.wdecay))
    test_result = test_model.evaluate_generator(test_generator)
    print('Test:', test_result)
