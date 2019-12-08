# -*- coding:utf-8 -*-
'''
Author: Haoran Chen
Date: 2017-05-02
'''

import tensorflow as tf
import FMPnet as fmp
import numpy as np
from datetime import datetime
import os
import preprocess as pps

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
IMAGE_SIZE = 46
BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.0001
EPOCH = 128
WEIGHT_DECAY_RATE = 1.
MODEL_SAVE_PATH = os.path.join('.', 'model')
MODEL_NAME = 'model.ckpt'
LOG_DIR = os.path.join('.', 'logs')


def bulid_model():
    variables = {}
    x = tf.placeholder(tf.float32, [None,IMAGE_SIZE,IMAGE_SIZE,fmp.NUM_CHANNELS], name='x-input')
    variables['x'] = x
    gt = tf.placeholder(tf.int64, [None], name='y-input')
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
    writer = tf.summary.FileWriter(
            os.path.join(LOG_DIR,'train'), tf.get_default_graph())
    variables['writer'] = writer
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 1000, WEIGHT_DECAY_RATE)
    variables['global_step'] = global_step
    optimizer = tf.train.AdamOptimizer(learning_rate)
    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    grads, vs = zip(*optimizer.compute_gradients(loss, trainable_variables))
    grads, global_norm = tf.clip_by_global_norm(grads, 10)
    train_step = optimizer.apply_gradients(zip(grads, vs), global_step)
    variables['train_step'] = train_step
    variables['global_step'] = global_step
    saver = tf.train.Saver()    
    variables['saver'] = saver
    return variables

def train(train_data, sess, variables):
    train_step = variables['train_step']
    loss = variables['loss']
    global_step = variables['global_step']
    accuracy = variables['accuracy']
    merged = variables['merged']
    x = variables['x']
    gt = variables['gt']
    writer = variables['writer']
    acc_ave = 0
    loss_ave = 0.
    cnt = 0
    while train_data.start_index != train_data.length:
        ret_images, ret_labels = train_data.get_next_batch(BATCH_SIZE, False)
        _, loss_value, step, acc, summary = sess.run(
                [train_step, loss, global_step, accuracy, merged],
                feed_dict={x:ret_images, gt:ret_labels})
        acc_ave += acc * ret_labels.shape[0]
        loss_ave += loss_value * ret_labels.shape[0]
        cnt += ret_images.shape[0]
    acc_ave /= cnt
    loss_ave /= cnt
    cnt = 0
    print('\n', datetime.now())
    print("After %d training step(s), loss on training batch is %.4f, accuracy is %.4f"%(
        step, loss_ave, acc_ave))
    writer.add_summary(summary, step)


def develop(dev_data, sess, variables):
    loss = variables['loss']
    accuracy = variables['accuracy']
    x = variables['x']
    gt = variables['gt']
    acc_ave = 0
    loss_ave = 0.
    cnt = 0
    while dev_data.start_index != dev_data.length:
        ret_images, ret_labels = dev_data.get_next_batch(BATCH_SIZE, False)
        loss_value, acc = sess.run(
                [loss, accuracy],
                feed_dict={x:ret_images, gt:ret_labels})
        acc_ave += acc * ret_labels.shape[0]
        loss_ave += loss_value * ret_labels.shape[0]
        cnt += ret_images.shape[0]
    acc_ave /= cnt
    loss_ave /= cnt
    print('\n', datetime.now())
    print("Loss on develop set is %.4f, accuracy is %.4f"%(loss_ave, acc_ave), flush=True)
    return acc_ave


def test(test_data, sess, variables):
    loss = variables['loss']
    accuracy = variables['accuracy']
    x = variables['x']
    gt = variables['gt']
    acc_ave = 0
    loss_ave = 0.
    cnt = 0
    while test_data.start_index != test_data.length:
        ret_images, ret_labels = test_data.get_next_batch(BATCH_SIZE, False)
        loss_value, acc = sess.run(
                [loss, accuracy],
                feed_dict={x:ret_images, gt:ret_labels})
        acc_ave += acc * ret_labels.shape[0]
        loss_ave += loss_value * ret_labels.shape[0]
        cnt += ret_images.shape[0]
    acc_ave /= cnt
    loss_ave /= cnt
    print('\n', datetime.now())
    print("Loss on test set is %.4f, accuracy is %.4f"%(loss_ave, acc_ave))


def train_dev_test(train_data, dev_data, test_data, variables):
    best_acc = 0.
    saver = variables['saver']
    writer = variables['writer']
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(EPOCH):
            train(train_data, sess, variables)
            train_data.start_index = 0
            train_data.shuffle()
            acc = develop(dev_data, sess, variables)
            dev_data.start_index = 0
            if acc > best_acc:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
                best_acc = acc
                print('Epoch:', i, 'saved')
        saver.restore(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        test(test_data, sess, variables)
    writer.close()


def main(argv=None):
    assert len(argv) == 4, "python train_dev_test_FMP.py WEIGHT_DECAY_RATE GPU_DEVICE_NUM MODEL_NAME"
    global WEIGHT_DECAY_RATE, MODEL_NAME
    WEIGHT_DECAY_RATE = float(argv[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = argv[2]
    MODEL_NAME = argv[3]
    train_data = pps.get_data('train', IMAGE_SIZE)
    dev_data = pps.get_data('dev', IMAGE_SIZE)
    test_data = pps.get_data('test', IMAGE_SIZE)
    variables = bulid_model()
    train_dev_test(train_data, dev_data, test_data, variables)

if __name__ == '__main__':
    tf.app.run()
