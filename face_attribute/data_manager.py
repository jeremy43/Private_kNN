from __future__ import print_function, absolute_import
import os, csv
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import pickle
import h5py
import network
import scipy.io as sio
import config as config
config = config.config
#from scipy.misc import imsave
import utils
from utils import mkdir_if_missing, write_json, read_json
from PIL import Image




class CelebA(object):
    """
    CelebA Attribute Dataset

    """
    dataset_dir = 'celeba'

    def __init__(self, root='data', **kwargs):
        self.dataset_dir = '../list'
        self.image_dir = '../list'

        label_path = os.path.join(self.dataset_dir,'celeba_2d_train_filelist.txt')
        with open(label_path, 'r') as txt:
            line = txt.readlines()
            train_line = line
        train_line = train_line
        est_path = os.path.join(self.dataset_dir, 'celeba_2d_test_filelist.txt')

        with open(test_path,'r') as txt:
            line = txt.readlines()
            test_line = line

        #self._check_before_run()

        train_data, train_label, num_train_imgs = self._process_dir(train_line)
        test_data, test_label, num_test_imgs = self._process_dir(test_line)
        num_total_imgs = num_train_imgs + num_test_imgs
        print("=> CelebA loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # images")
        print("  ------------------------------")
        print("  train    | {:8d}".format(num_train_imgs))
        print("  test     | {:8d}".format(num_test_imgs))
        print("  ------------------------------")
        print("  total    | {:8d}".format(num_total_imgs))
        print("  ------------------------------")

        self.train_data = train_data
        self.test_data = test_data
        self.train_label = train_label
        self.test_label = test_label
 
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.image_dir):
            raise RuntimeError("'{}' is not available".format(self.image_dir))

    def _process_dir(self, lines):
        data =[]
        label =[]
        for img_idx, img_info in enumerate(lines):
            img_path = img_info.split(' ', 1)[0]
            cur_label = img_info.split(' ', 1)[1].split()
            img_path = osp.join(self.image_dir, img_path)
            if os.path.exists(img_path):
                pid = np.array(list(map(int, cur_label)))
                img = Image.open(img_path).convert('RGB')
            data.append(img)
            label.append(pid)

        num_imgs = len(data)
        label = np.array(label,dtype=np.int32)
        print('len of label list',len(label))
        return data,label, num_imgs

    def _data_partition(self,nb_teachers,teacher_id):

        #return a partion of train private data
        assert int(teacher_id) < int(nb_teachers)

        batch_len = int(len(self.train_data)/nb_teachers)
        start = teacher_id * batch_len
        end = (teacher_id + 1) * batch_len

        # Slice partition off
        partition_data = self.train_data[start:end]
        partition_labels = self.train_label[start:end]
        return partition_data,partition_labels


"""Create dataset"""

__img_factory = {
        'celeba':CelebA}
def get_names():
    return list(__img_factory.keys())

def init_img_dataset(name, **kwargs):
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))
    return __img_factory[name](**kwargs)

