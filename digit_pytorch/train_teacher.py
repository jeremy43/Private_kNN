# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import os

import svhn_config as config
from torchvision import datasets as dataset
config = config.config
from PIL import Image
import numpy as np
import aggregation
import transforms as T
import pickle
import autodp

#from dataset_loader import ImageDataset
import network
from torch.utils.data import DataLoader
import aggregation
import utils
import sys
import os
sys.path.append('..')



    

def train_tracher():
    """
    Partition the entire private (training) data into config.nb_teacher subsets and train config.nb_teacher  teacher models 
    """
    # Load the dataset
    if config.dataset == 'mnist':
        train_dataset = dataset.MNIST(root=config.data_dir, train=True, download=True)
        test_dataset = dataset.MNIST(root=config.data_dir, train=False, download=True)
        ori_train_data= [ data[0] for idx, data in enumerate(train_dataset)]
        ori_test_data = [ data[0] for idx, data in enumerate(test_dataset)]
        test_labels = test_dataset.targets
        train_labels = train_dataset.targets
    elif config.dataset =='svhn':
        train_dataset = dataset.SVHN(root=config.data_dir, split='train', download=True)
        extra_dataset = dataset.SVHN(root=config.data_dir, split='extra', download=True)
        test_dataset = dataset.SVHN(root=config.data_dir, split='test', download=True)
        ori_train_data = np.concatenate((train_dataset.data,extra_dataset.data),axis=0)
        print('ori data shape', ori_train_data.shape)
        ori_train_data = np.transpose(ori_train_data, (0, 2, 3, 1))
        print('orig data shape', ori_train_data.shape)
        #ori_train_data= [ data[0] for idx, data in enumerate(train_dataset.data)]
        #for data in extra_dataset.data:
        #    ori_train_data.append(data)
        #ori_test_data = [ data[0] for idx, data in enumerate(test_dataset.data)]
        ori_test_data = np.transpose(test_dataset.data, (0,2,3,1))
        test_labels = test_dataset.labels
        extra_labels = extra_dataset.labels
        train_labels = [ll for ll in train_dataset.labels]
        for ll in extra_labels:
            train_labels.append(ll)
    batch_len = int(len(ori_train_data)/config.nb_teachers)
    for i in range(0,1):
        dir_path = os.path.join(config.save_model,'pate_'+str(config.nb_teachers))
        utils.mkdir_if_missing(dir_path)
        filename = os.path.join(dir_path, str(config.nb_teachers) + '_teachers_' + str(i) + config.arch+'.checkpoint.pth.tar')
        print('save_path for teacher{}  is {}'.format(i,filename))
        start = i * batch_len
        end = (i + 1) * batch_len
        t_data = ori_train_data[start : end]
        t_labels = train_labels[start: end] 
        network.train_each_teacher(config.teacher_epoch, t_data, t_labels, ori_test_data, test_labels, filename)


train_tracher()

