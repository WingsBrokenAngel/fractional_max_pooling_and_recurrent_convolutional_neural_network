# -*- coding:utf-8 -*-

import tensorflow as tf
import FMPnet as fmp
import train_dev_test_FMP as tdtfmp
import preprocess as pps
from datetime import datetime
import os
import numpy as np
import time
from collections import Counter
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def bulid_model():
    variables = {}
    x = tf.placeholder(
            tf.float32, 
            [None,tdtfmp.IMAGE_SIZE,tdtfmp.IMAGE_SIZE,fmp.NUM_CHANNELS], 
            name='x-input')
    variables['x'] = x
    gt = tf.placeholder(
            tf.int64,
            [None],
            name='y-input')
    variables['gt'] = gt
    logits = fmp.inference(x)
    loss = tf.losses.sparse_softmax_cross_entropy(gt, logits + 1e-8)
    variables['loss'] = loss
    tf.summary.scalar('loss', loss)
    prediction = tf.argmax(logits, 1)
    variables['prediction'] = prediction
    accuracy = tf.equal(prediction, gt)
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    variables['accuracy'] = accuracy
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    variables['merged'] = merged

    saver = tf.train.Saver()    
    variables['saver'] = saver
    return variables


def test(test_data, variables, sess):
    prediction = variables['prediction']
    x = variables['x']
    gt = variables['gt']
    acc_ave = 0
    results = {}
    for i in range(12):
        results[i] = []
        while test_data.start_index != test_data.length:
            ret_images, ret_labels = test_data.get_next_batch(tdtfmp.BATCH_SIZE, False)
            pred = sess.run(prediction, feed_dict={x:ret_images, gt:ret_labels})
            results[i].append(pred)
        test_data.start_index = 0
        results[i] = np.concatenate(results[i], axis=0)
    results = np.stack([results[key] for key in results], axis=1)
    print('All result shape:', results.shape, 'dtype:', results.dtype)
    vote_results = []
    for arr in results:
        vote_results.append(Counter(arr).most_common(1)[0][0])
    labels = test_data.labels
    acc_ave = (vote_results == labels).mean()
    print('\n', datetime.now())
    print("Accuracy is %.4f"%(acc_ave))


def main(argv=None):
    test_data = pps.get_data('test', tdtfmp.IMAGE_SIZE)
    variables = bulid_model()
    saver = variables['saver']
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, os.path.join(tdtfmp.MODEL_SAVE_PATH, tdtfmp.MODEL_NAME))
        test(test_data, variables, sess)

if __name__ == "__main__":
    tf.app.run()

