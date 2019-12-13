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
from RCNN import RCNN


if __name__ == "__main__":
    tf.app.flags.DEFINE_string('name', 'rcnn', 'name of model')
    tf.app.flags.DEFINE_float('lr', 0.001, 'Learning rate of the model')
    tf.app.flags.DEFINE_float('drop', 0.2, 'Drop rate for dropout layers')
    tf.app.flags.DEFINE_integer('filters', 96, 'Filter number')
    tf.app.flags.DEFINE_float('wdecay', 0.0001, 'Weight Decay')
    flags = tf.app.flags.FLAGS

    data = cifar10.load_data()
    (train_data, train_labels), (test_data, test_labels) = data
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    train_datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, 
        brightness_range=[0.8, 1.2], horizontal_flip=True, rescale=1./255)
    train_generator = train_datagen.flow(
        train_data[:-5000], train_labels[:-5000], batch_size=256)

    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow(train_data[-5000:], 
        train_labels[-5000:], batch_size=256, shuffle=False)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow(
        test_data, test_labels, batch_size=256, shuffle=False)

    rcnn = RCNN(flags.filters, flags.wdecay, flags.drop)
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor_train = rcnn(input_tensor, True, 3)
    output_tensor_test = rcnn(input_tensor, False, 3)
    train_model = tf.keras.Model(input_tensor, output_tensor_train)
    test_model = tf.keras.Model(input_tensor, output_tensor_test)

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./model/%s-%d-%g-%g-%g-best.h5'%(
                flags.name, flags.filters, flags.lr, flags.wdecay, flags.drop), 
            monitor='val_acc', save_best_only=True), 
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_acc', factor=0.5, patience=5, min_lr=1e-6)]
    train_model.summary()
    train_model.compile(
        optimizer=tf.keras.optimizers.SGD(flags.lr, 0.9, nesterov=True), 
        loss='categorical_crossentropy', metrics=['acc'])

    test_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    history = train_model.fit_generator(
        train_generator, epochs=256, 
        validation_data=val_generator, max_queue_size=256, workers=2, 
        callbacks=callbacks_list)

    test_model.load_weights('./model/%s-%d-%g-%g-%g-best.h5'%(
                flags.name, flags.filters, flags.lr, flags.wdecay, flags.drop))
    test_result = test_model.evaluate_generator(test_generator)
    print(test_result)

