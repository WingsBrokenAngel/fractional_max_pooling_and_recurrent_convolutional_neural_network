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
    test_labels = to_categorical(test_labels)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow(
        test_data, test_labels, batch_size=256, shuffle=False)

    rcnn = RCNN(flags.filters, flags.wdecay, flags.drop)
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor_test = rcnn(input_tensor, False, 3)
    test_model = tf.keras.Model(input_tensor, output_tensor_test)

    test_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    print(test_model.evaluate_generator(test_generator))
    test_model.load_weights('./model/%s-%d-%g-%g-%g-best.h5'%(
                flags.name, flags.filters, flags.lr, flags.wdecay, flags.drop))
    test_result = test_model.evaluate_generator(test_generator)
    predicts = test_model.predict(test_generator)

    print(test_result)

