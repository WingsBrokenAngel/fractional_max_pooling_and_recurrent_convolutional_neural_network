# -*- coding: utf-8 -*-
'''
Author: Haoran Chen
Date: 2018-04-18
'''

import numpy as np
import os
from PIL import Image
import tensorflow.keras.datasets.cifar100 as cifar100


data = cifar100.load_data()
train_dev_data, test_data = data[0], data[1]
train_dev_samples, train_dev_labels = train_dev_data
train_dev_labels = np.squeeze(train_dev_labels)
print('CIFAR100 train dev samples:', train_dev_samples.shape)
print('CIFAR100 train dev labels:', train_dev_labels.shape)
test_samples, test_labels = test_data
test_labels = np.squeeze(test_labels)
print('CIFAR100 test samples:', test_samples.shape)
print('CIFAR100 test labels:', test_labels.shape)

class get_data():
    def __init__(self, flag, image_size):
        self.flag = flag
        self.image_size = image_size
        if flag == "train":
            # images shape: (45000, 32, 32, 1), labels shape: (45000,)
            self.images, self.labels = train_dev_samples[:-5000], train_dev_labels[:-5000]

        elif flag == "dev":
            # images shape: (5000, 32, 32, 1), labels shape: (5000,)
            self.images, self.labels = train_dev_samples[-5000:], train_dev_labels[-5000:]
        
        elif flag == "test":
            # images shape: (10000, 32, 32, 1), labels shape: (10000,)
            self.images, self.labels = test_samples, test_labels

        else:
            raise ValueError

        tmp = []
        for img in self.images:
            img_resize = Image.fromarray(img).resize((image_size, image_size))
            tmp.append(img_resize)
        tmp = np.stack(tmp, axis=0)
        print('Resized images:', tmp.shape)
        self.images = tmp
        self.images = self.images.astype(np.float32)
        # normalize the images
        self.images = (self.images / 256 - 0.5) * 2
        self.start_index = 0
        self.length = self.images.shape[0]
        self.indices = np.arange(self.length)

    def shuffle(self):
        np.random.shuffle(self.indices) 
    
    def get_next_batch(self, size, if_arg):
        end = min(self.start_index + size, self.length)
        size = end - self.start_index

        ret_images = self.images[self.indices[self.start_index:end]]
        ret_labels = self.labels[self.indices[self.start_index:end]]
        if if_arg:
            for i in range(size):
                if_need_to_flip = np.random.choice([True, False])
                if if_need_to_flip:
                    ret_images[i] = np.fliplr(ret_images[i])
                                            
        self.start_index = end
        return ret_images, ret_labels

