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

import mnist_config as config
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
sys.path.append('../autodp/autodp')

import rdp_bank, dp_acct, rdp_acct, privacy_calibrator
import metrics
prob = 0.15 # The sub-sampling ratio we used for MNIST is 0.15 and 0.05 for SVHN
acct = rdp_acct.anaRDPacct()
dependent_acct = rdp_acct.anaRDPacct()
delta = config.delta
sigma = config.sigma1  # gaussian parameter
gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
gaussian2 = lambda x: rdp_bank.RDP_inde_pate_gaussian({'sigma': config.gau_scale}, x)
#gaussian is used for noisy screening and gaussian2 is used for noisy aggregation, note the the global sensitivity of the latter is 2 instead of 1, thus we apply RDP_pate_gaussian
dir_path = config.save_model 


    
def extract_feature(train_img, test_img, path=None):
    """
    This help to compute feature for knn from pretrained network
    :param FLAGS:
    :param ckpt_path:
    :return:
    """

    # check if a certain variable has been saved in the model
    if config.extract_feature == 'feature':
        # Update the feature extractor using the student model(filename) in the last iteration. 
        # Replace the filename with the saved student model, the following in an example of the checkpoint
        filename  = 'save_model/svhn/knn_num_neighbor_800/800_stdnt_.checkpoint.pth.tar'
        train_feature = network.pred(train_img, filename, return_feature=True)
        test_feature = network.pred(test_img, filename, return_feature=True)
        return train_feature, test_feature
    train_img = [np.asarray(data) for data in train_img]
    test_img = [np.asarray(data) for data in test_img]


    if config.extract_feature == 'hog':
        # usually the file to save all hog is too large. we decompose it into 10 pieces.
        train_data = None
        each_length = int((9+len(train_img))/10)
        for idx in range(10):
            #Save pkl into several small pieces, incase the size of private dataset is too large 

            train_hog_path = os.path.join(config.hog_path, config.dataset + str(idx)+ '_train_hog.pkl')
            if os.path.exists(train_hog_path) == False:
                p1 = idx*each_length
                p2 = min((idx+1)*each_length,len(train_img))
                print('save_hog_pkl for interval{} : {}'.format(p1,p2))
                utils.save_hog(train_img[p1:p2],train_hog_path)

            with open(train_hog_path, 'rb') as f:
                if train_data is not None:
                    train_data = np.vstack((train_data,pickle.load(f)))
                else:
                    train_data = pickle.load(f)
            print('load hog feature shape', train_data.shape)
        test_hog_path = os.path.join(config.hog_path, config.dataset + '_test_hog.pkl')
        if os.path.exists(test_hog_path) == False:
            utils.save_hog(test_img, test_hog_path)
        with open(test_hog_path, 'rb') as f:
            test_data = pickle.load(f)

        return train_data, test_data
    if config.extract_feature =='pca':
        return utils.pca(test_img, train_img)


def prepare_student_data( save=False):
    """
    Takes a dataset name and the size of the teacher ensemble and prepares
    training data for the student model, according to parameters indicated
    in flags above.
    :param dataset: string corresponding to mnist, cifar10, or svhn
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :param save: if set to True, will dump student training labels predicted by
                 the ensemble of teachers (with Laplacian noise) as npy files.
                 It also dumps the clean votes for each class (without noise) and
                 the labels assigned by teachers
    :return: pairs of (data, labels) to be used for student training and testing

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
        train_dataset = dataset.SVHN(root=config.data_dir, split = 'train', download=True)
        extra_dataset = dataset.SVHN(root=config.data_dir, split='extra', download=True)
        test_dataset = dataset.SVHN(root=config.data_dir, split='test', download=True)
        ori_train_data = np.concatenate((train_dataset.data,extra_dataset.data),axis=0)
        ori_train_data = np.transpose(ori_train_data, (0,2,3,1))
        print('orig data shape', ori_train_data.shape)
        
        ori_test_data = np.transpose(test_dataset.data, (0,2,3,1))
        test_labels = test_dataset.labels
        extra_labels = extra_dataset.labels
        train_labels = [ll for ll in train_dataset.labels]
        for ll in extra_labels:
            train_labels.append(ll)

    numpy_test_data = ori_test_data
    if config.dataset == 'mnist':
        # UDA only accepts the numpy format instead of the original Image format.
        # If do not use UDA, ignore this part
        numpy_test_data = [np.asarray(x) for x in ori_test_data]
        numpy_test_data = np.array(numpy_test_data)
        numpy_test_data = numpy_test_data.reshape([-1, 28*28])
    else:
        numpy_test_data = numpy_test_data.reshape([-1, 32 * 32 * 3])
        
    print('numpy_test_data',numpy_test_data.shape)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    print('train_label shape', train_labels.shape)
    if config.use_uda_data == True:
        # Load the pseudo-labels predicted by UDA/VAT
        # When you run uda, make sure to save the images and pseudo_labels, then we will train a student model based on these images & labels. The size of images (pseudo_labels) is as large as the public dataset (except for the testing dataset).
        uda_path = 'record/svhn_UDA.npy'
        uda_labels = np.load(uda_path)
        uda_labels = np.array(uda_labels, dtype = np.int32)
        uda_img_path = uda_path.split('.npy')[0]+ '_ori_img.npy'
        uda_img = np.load(uda_img_path)
        uda_img = uda_img.reshape([-1,32,32,3])
        uda_img = uda_img.astype(np.uint8)
        
        return  uda_img, uda_labels, ori_test_data[config.stdnt_share:], test_labels[config.stdnt_share:]
    
    # Make sure there is data leftover to be used as a test set
    assert config.stdnt_share < len(ori_test_data)

    train_data, test_data = extract_feature(ori_train_data, ori_test_data)

    stdnt_data = test_data[:config.stdnt_share]
    num_train = train_data.shape[0]
    
    teachers_preds = np.zeros([stdnt_data.shape[0], config.nb_teachers])

    for idx in range(len(stdnt_data)):
        query_data = stdnt_data[idx]
        select_teacher = np.random.choice(train_data.shape[0], int(prob * num_train))
        dis = np.linalg.norm(train_data[select_teacher] - query_data, axis=1)
        k_index = select_teacher[np.argsort(dis)[:config.nb_teachers]]
        # sum over the number of teachers, which make it easy to compute their votings
        teachers_preds[idx] = np.array(train_labels[k_index], dtype = np.int32)
    teachers_preds = np.asarray(teachers_preds, dtype=np.int32)
       
    dir_path = os.path.join(config.save_model, 'knn_num_neighbor_' + str(config.nb_teachers))
    
    
    acct.compose_poisson_subsampled_mechanisms(gaussian, prob,coeff = len(teachers_preds))

    print("Composition of student   Gaussian mechanisms gives {} ".format(acct.get_eps(delta), delta))
    idx, stdnt_labels,remain_idx = aggregation.aggregation_knn(teachers_preds, config.gau_scale)
    print('answer {} queries over {}'.format(len(stdnt_labels),len(teachers_preds)))
    acct.compose_poisson_subsampled_mechanisms(gaussian2, prob,coeff = len(stdnt_labels))

    if config.data_dependent_rdp:
        print('Not implemented here')
        # we provide the examples of data-dependent analysis in the privacy analysis folder, which you need to prepare a teachers' prediction file. 
    else:
        print('Composition gives {}'.format(acct.get_eps(delta), delta))
    # Print accuracy of aggregated label
    ac_ag_labels = metrics.accuracy(stdnt_labels, test_labels[:config.stdnt_share][idx])
    print("Accuracy of the aggregated labels: " + str(ac_ag_labels))
    current_eps = acct.get_eps(config.delta)
    # Store unused part of test set for use as a test set after student training
    stdnt_test_data = ori_test_data[config.stdnt_share:]
    stdnt_test_labels = test_labels[config.stdnt_share:]
    #ori_test_data = np.array(ori_test_data)
    if config.use_uda:
        utils.convert_vat(idx, remain_idx,numpy_test_data, test_labels, stdnt_labels)
    if save:
        # Prepare filepath for numpy dump of labels produced by noisy aggregation
        dir_path = os.path.join(config.save_model, 'knn_num_neighbor_' + str(config.nb_teachers))
        utils.mkdir_if_missing(dir_path)
        filepath = dir_path + 'answer_'+str(len(stdnt_labels))+'_knn_voting.npy'  # NOLINT(long-line)
        label_file = dir_path + 'answer_'+str(len(stdnt_labels))+'.pkl'

        # Dump student noisy labels array
        with open(filepath, 'wb') as file_obj:
            np.save(file_obj, teachers_preds)
    
    #condident data are those which pass the noisy screening 
    confident_data = [ ori_test_data[:config.stdnt_share][i] for i in idx[0]]
    return confident_data, stdnt_labels, stdnt_test_data, stdnt_test_labels


def train_student(nb_teachers):
    """
    This function trains a student using predictions made by an ensemble of
    teachers. The student and teacher models are trained using the same
    neural network architecture.
    :param dataset: string corresponding to celeba
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :return: True if student training went well
    """
    # Call helper function to prepare student data using teacher predictions
    dir_path = os.path.join(config.save_model, 'knn_num_neighbor_' + str(config.nb_teachers))
    stdnt_dataset = prepare_student_data( save=True)
    stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels = stdnt_dataset
    utils.mkdir_if_missing(dir_path)
    filename = os.path.join(dir_path,str(config.nb_teachers) + '_stdnt_.checkpoint.pth.tar')
    print('save_file', filename)
    print('stdnt_label used for train', stdnt_labels.shape)
    network.train_each_teacher(config.student_epoch, stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels,
                               filename)

    return True


def main(argv=None):  
    train_student(config.nb_teachers)


main()
