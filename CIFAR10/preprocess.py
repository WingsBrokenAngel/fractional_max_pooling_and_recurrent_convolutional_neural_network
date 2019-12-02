# -*- coding: utf-8 -*-
'''
Author: Haoran Chen
Date: 2018-04-18
'''

import numpy as np
import os
from PIL import Image


def unpickle(file_name):
    import pickle
    with open(file_name, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


TRAIN_FILE_NAME = 'data_batch_'
TEST_FILE_NAME = 'test_batch'


def load_train_data():
    data = []
    labels = []
    for i in range(1, 6):
        print("batch:", i, "completed")
        file_name = os.path.join('..', TRAIN_FILE_NAME + str(i))
        data_dict = unpickle(file_name)
        data_tmp = data_dict[b'data']
        data_tmp = data_tmp.reshape((10000, 32, 32, 3), order='F')
        data += [data_tmp]
        labels_tmp = data_dict[b'labels']
        labels += [labels_tmp]
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    return data, labels


def load_test_data():
    file_name = os.path.join('..', TEST_FILE_NAME)
    data_dict = unpickle(file_name)
    data = data_dict[b'data'].reshape((10000, 32, 32, 3), order='F')
    labels = np.array(data_dict[b'labels'])
    
    return data, labels  


class get_data():
    def __init__(self, flag, image_size):
        self.flag = flag
        self.image_size = image_size
        if flag == "train":
            self.images, self.labels = load_train_data()
            self.images, self.labels = self.images[:-5000], self.labels[:-5000]
            self.images = self.images.transpose(0, 2, 1, 3)

        elif flag == "dev":
            self.images, self.labels = load_train_data()
            self.images, self.labels = self.images[-5000:], self.labels[-5000:]
            self.images = self.images.transpose(0, 2, 1, 3)
        
        elif flag == "test":
            self.images, self.labels = load_test_data()
            self.images = self.images.transpose(0, 2, 1, 3)

        else:
            raise ValueError

        # convert the images from uint8 to float32
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

    def shuffle(self):
        indices = np.arange(self.length)
        np.random.shuffle(indices)
        self.images = self.images[indices]
        self.labels = self.labels[indices]
    
    
    def get_next_batch(self, size, if_arg):
        end = min(self.start_index + size, self.length)
        size = end - self.start_index

        ret_images = self.images[self.start_index:end]
        ret_labels = self.labels[self.start_index:end]
        if if_arg:
            for i in range(size):
                if_need_to_flip = np.random.choice([True, False])
                if if_need_to_flip:
                    ret_images[i] = np.fliplr(ret_images[i])
                                            
        self.start_index = end
        return ret_images, ret_labels

