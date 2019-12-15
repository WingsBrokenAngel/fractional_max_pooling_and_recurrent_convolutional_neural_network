'''
Author: Haoran Chen
Date: 12/10/2019
'''
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.activations import relu
import tensorflow.keras.backend as K
from RCNN import RCNN
from FMPnet import FMP
import os
import numpy as np


def schedule_func(epochs, lr):
    if epochs == 0:
        return lr
    else:
        return lr*((1.0e-3)**(1./128))


if __name__ == "__main__":
    tf.app.flags.DEFINE_string('name', 'rcnn', 'name of model: rcnn, fmp')
    tf.app.flags.DEFINE_string('gpu', '8', 'gpu index')
    tf.app.flags.DEFINE_float('lr', 0.001, 'Learning rate of the model')
    tf.app.flags.DEFINE_float('drop', 0.5, 'Drop rate for dropout layers')
    tf.app.flags.DEFINE_integer('filters', 96, 'Filter number')
    tf.app.flags.DEFINE_float('wdecay', 0.0001, 'Weight Decay')
    flags = tf.app.flags.FLAGS

    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    
    data = mnist.load_data()
    (train_data, train_labels), (test_data, test_labels) = data
    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    train_datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=6, height_shift_range=6, 
        horizontal_flip=True, rescale=1./255)
    train_generator = train_datagen.flow(
        train_data, train_labels, batch_size=256)
    '''
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow(train_data[-5000:], 
        train_labels[-5000:], batch_size=256, shuffle=False)
    '''
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow(
        test_data, test_labels, batch_size=256, shuffle=False)

    try:
        if flags.name == "rcnn":
            model = RCNN(flags.filters, flags.wdecay, flags.drop)
            input_tensor = tf.keras.Input(shape=(28, 28, 1))
            output_tensor_train = model(input_tensor, True, 3)
            output_tensor_test = model(input_tensor, False, 3)
        elif flags.name == "fmp":
            model = FMP(flags.filters, flags.wdecay, flags.drop)
            input_tensor = tf.keras.Input(shape=(28, 28, 1))
            output_tensor_train = model(input_tensor, True)
            output_tensor_test = model(input_tensor, False)
        else:
            raise NameError
    except TypeError:
        print("Incorrect Model Name")

    train_model = tf.keras.Model(input_tensor, output_tensor_train)
    test_model = tf.keras.Model(input_tensor, output_tensor_test)

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./model/%s-%d-%g-%g-%g-best.h5'%(
                flags.name, flags.filters, flags.lr, flags.wdecay, flags.drop), 
            monitor='acc', save_best_only=True), 
        tf.keras.callbacks.LearningRateScheduler(schedule_func, 1)]
    train_model.summary()
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(flags.lr, amsgrad=True), 
        loss='categorical_crossentropy', metrics=['acc'])

    test_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    history = train_model.fit_generator(
        train_generator, epochs=128, 
        # validation_data=val_generator, max_queue_size=256, workers=2, 
        callbacks=callbacks_list)

    test_model.load_weights('./model/%s-%d-%g-%g-%g-best.h5'%(
                flags.name, flags.filters, flags.lr, flags.wdecay, flags.drop))
    test_result = test_model.evaluate_generator(test_generator)
    print(test_result)

