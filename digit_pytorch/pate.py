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
sys.path.append('..')
#import autodp
from autodp1.autodp import rdp_bank, dp_acct, rdp_acct, privacy_calibrator
import metrics
prob = 0.15  # subsample probability for i
acct = rdp_acct.anaRDPacct()
dependent_acct = rdp_acct.anaRDPacct()
delta = config.delta
sigma = config.sigma1  # gaussian parameter
gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
gaussian2 = lambda x: rdp_bank.RDP_pate_gaussian({'sigma': config.gau_scale}, x)
dir_path = config.save_model 




    
def ensemble_preds(nb_teachers, stdnt_data):

    result_shape = (nb_teachers, len(stdnt_data))
    # Create array that will hold result
    result = np.zeros(result_shape, dtype=np.float32)
    dir_path = os.path.join(config.save_model,'pate_'+str(config.nb_teachers))
    for teacher_id in range(config.nb_teachers):
        filename = os.path.join(dir_path, str(config.nb_teachers) + '_teachers_' + str(teacher_id) + config.arch+'.checkpoint.pth.tar')
        result[teacher_id] = network.pred(stdnt_data, filename)
        print("Computed Teacher " + str(teacher_id) + " softmax predictions")
    result = np.asarray(result, dtype = np.int32)
    return result.transpose()
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
        test_dataset = dataset.MNIST(root=config.data_dir, train=False, download=True)
        ori_test_data = [ data[0] for idx, data in enumerate(test_dataset)]
    elif config.dataset =='svhn':
        test_dataset = dataset.SVHN(root=config.data_dir, train=False, download=True)
        ori_test_data = [ data[0] for idx, data in enumerate(test_dataset)]
    #print('whether img or numpy', type(ori_train_data[0]))
    test_labels = test_dataset.targets
    test_labels = np.array(test_labels)
    stdnt_data = ori_test_data[:config.stdnt_share]
    # Compute teacher predictions for student training data
    teachers_preds = ensemble_preds(config.nb_teachers, stdnt_data)

       
    dir_path = os.path.join(config.save_model, 'pate_' + str(config.nb_teachers))
    
    
    acct.compose_mechanism(gaussian, coeff = len(teachers_preds))

    print("Composition of student   Gaussian mechanisms gives {} ".format(acct.get_eps(delta), delta))
    idx, stdnt_labels,remain_idx = aggregation.aggregation_knn(teachers_preds, config.gau_scale)
    print('answer {} queries over {}'.format(len(stdnt_labels),len(teachers_preds)))
    acct.compose_mechanism(gaussian2, coeff = len(stdnt_labels))

    if config.data_dependent_rdp:
        # we do not implement data-dependent here, if want it please copy it from cifar10
        print('data_indepent give total eps {} dependent give total eps{}'.format(acct.get_eps(delta), dependent_acct.get_eps(delta)))
    else:
        print('Composition 2 gives {}'.format(acct.get_eps(delta), delta))
    # Print accuracy of aggregated label
    ac_ag_labels = metrics.accuracy(stdnt_labels, test_labels[:config.stdnt_share][idx])
    print("Accuracy of the aggregated labels: " + str(ac_ag_labels))
    # Store unused part of test set for use as a test set after student training
    stdnt_test_data = ori_test_data[config.stdnt_share:]
    stdnt_test_labels = test_labels[config.stdnt_share:]
    #ori_test_data = np.array(ori_test_data)
    if config.use_uda:
        utils.convert_vat(idx, remain_idx,numpy_test_data, test_labels, stdnt_labels)
    if save:
        # Prepare filepath for numpy dump of labels produced by noisy aggregation
        utils.mkdir_if_missing(dir_path)
        filepath = dir_path + 'answer_'+str(len(stdnt_labels))+'_pate_voting.npy'  # NOLINT(long-line)
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
    dir_path = os.path.join(config.save_model, 'pate_' + str(config.nb_teachers))
    stdnt_dataset = prepare_student_data( save=True)
    stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels = stdnt_dataset
    utils.mkdir_if_missing(dir_path)
    filename = os.path.join(dir_path,str(config.nb_teachers) + '_stdnt.checkpoint.pth.tar')

    print('stdnt_label used for train', stdnt_labels.shape)
    network.train_each_teacher(config.student_epoch, stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels,
                               filename)

    return True


def main(argv=None):  # pylint: disable=unused-argument
    train_student(config.nb_teachers)


main()
