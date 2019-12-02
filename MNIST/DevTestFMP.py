# -*- coding:utf-8 -*-

import tensorflow as tf
import trainFMP as tfmp
import FMPnet as nfmp
import preprocess as pps
from datetime import datetime
import os
import numpy as np
import time


def dev_and_test(data, kind):
    x = tf.placeholder(tf.float32, [None, tfmp.IMAGE_SIZE, tfmp.IMAGE_SIZE, 3], name='dev-x-input')
    y_ = tf.placeholder(tf.float32, [None, nfmp.NUM_OUTPUT], name='dev-y-input')
    c = {}
    c['train'] = tf.Variable(False, trainable=False)
    c['regularizer'] = None
    y, parameters = nfmp.inference(x, c)
    arg_y = tf.argmax(y, 1)
    arg_y_ = tf.argmax(y_, 1)

    accuracy = tf.equal(arg_y, arg_y_)
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy)

    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter(
            os.path.join(tfmp.LOG_DIR, kind), tf.get_default_graph())
    
    variable_averages = tf.train.ExponentialMovingAverage(
            tfmp.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    flag = True

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(tfmp.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            acc_score = 0
            iter_cnt = int(data.length // tfmp.BATCH_SIZE)

            for i in range(iter_cnt):
                feed_x, feed_x_norm, feed_y = data.get_next_batch(tfmp.BATCH_SIZE, False)
                tmp = np.zeros((tfmp.BATCH_SIZE, tfmp.IMAGE_SIZE, tfmp.IMAGE_SIZE, 3))
                tmp[:,31:63,31:63,:] = feed_x_norm
                feed_x_norm = tmp
                feed = {x:feed_x_norm, y_:feed_y}
                acc_tmp, ay, ay_, rec = sess.run(
                        [accuracy, arg_y, arg_y_, merged],
                        feed_dict=feed)
                acc_score += acc_tmp
                writer.add_summary(rec, int(global_step))

            acc_score /= iter_cnt
            print('\n', datetime.now(), 'After %s training step(s), %s acc=%g'%(global_step, kind, acc_score))
            if int(global_step) == tfmp.TRAINING_STEPS:
                flag = False
        else:
            print('No Checkpoint Found')
    writer.close()
        


def main(argv=None):
    kind = input('Do you want to develop or test?(dev, test)')
    assert kind in ['test', 'dev']
    data = pps.get_data(kind)
    dev_and_test(data, kind)

if __name__ == "__main__":
    tf.app.run()

