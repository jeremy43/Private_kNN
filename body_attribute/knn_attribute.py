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

import market_config as config
config = config.config
from PIL import Image
import numpy as np
import sys
import aggregation
import network
import pickle
from dataset_loader import ImageDataset
import aggregation
import utils
from utils import Hamming_Score as hamming_accuracy
sys.path.append('../autodp/autodp')
sys.path.append('../dataset/duke')
import rdp_bank, dp_acct, rdp_acct, privacy_calibrator
from datafolder.folder import Test_Dataset
from datafolder.folder import Train_Dataset
from utils import hamming_precision as hamming_precision

prob = 0.05  # subsample probability for private knn
acct = rdp_acct.anaRDPacct()
delta = config.delta
# For muli-label tasks, we only have noisy aggregation
sigma = config.gau_scale  # gaussian parameter for noisy aggregation
gaussian = lambda x: rdp_bank.RDP_inde_pate_gaussian({'sigma': int(sigma/config.tau)}, x)
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}

def tau_limit(labels):
    """
        for multi-label problem, limit the attribute of each neighbor to be smaller than tau, where tau could be served as a composition cofficient
    
    """
    votes = np.zeros(labels.shape)
    for idx in range(len(labels)):
        record = labels[idx]
        if np.sum(record) == 0:
            votes[idx] = record
        else:
            
            votes[idx] = record*min(config.tau/float(np.sum(record)),1)
    return np.sum(votes, axis= 0)
    return votes

def extract_feature(train_img, test_img, path=None):
    """
    This help to compute feature for knn from pretrained network
    :param FLAGS:
    :param ckpt_path:
    :return:
    """

    # check if a certain variable has been saved in the model

    if config.extract_feature == 'feature':
        dir_path = os.path.join(config.save_model, config.dataset)
        dir_path = os.path.join(dir_path, 'knn_num_neighbor_' + str(config.nb_teachers))
        filename = str(config.nb_teachers) + '_stdnt_resnet.checkpoint.pth.tar'
        filename = os.path.join(dir_path, filename)
        train_feature = network.pred(train_img, filename, return_feature=True)
        test_feature = network.pred(test_img, filename, return_feature=True)
        print('shape of extract feature', train_feature.shape)
        return train_feature, test_feature
        #return utils.pca(test_feature, train_feature)
    if config.extract_feature == 'hog':
        print('return hog feature')
        train_data = None
        each_length = int((9+len(train_img))/10)
        for idx in range(10):
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

def prepare_student_data(nb_teachers, save=False):
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
    if config.dataset == 'celeba':
        dataset = data_manager.init_img_dataset(root=config.data_dir, name=config.dataset)
        test_data = dataset.test_data
        test_labels = dataset.test_label
        train_data = dataset.train_data
        train_labels = dataset.train_label

    elif config.dataset =='market':
        data_dir = '../dataset/market1501'
        train_dataset = Train_Dataset(data_dir, dataset_name=dataset_dict[config.dataset],
                                        train_val='train')
        test_dataset = Test_Dataset(data_dir, dataset_name=dataset_dict[config.dataset],
                                             query_gallery='gallery')

        train_data = train_dataset.train_data
        train_labels = train_dataset.train_label
        test_data = test_dataset.data
        test_labels = test_dataset.label
        train_labels = np.array(train_labels,dtype =np.int32)
        test_labels = np.array(test_labels,dtype = np.int32)
        print('len of total test data in market',len(test_labels))
    else:
        return False



    # Make sure there is data leftover to be used as a test set
    assert config.stdnt_share < len(test_data)



    ori_test_data = test_data
    # for test


    train_data, test_data = extract_feature(train_data, test_data)

    stdnt_data = test_data[:config.stdnt_share]
    # the remaining 1000 records is the holdout for evaluating
    share_index =np.random.choice(test_data[:-1000].shape[0],config.stdnt_share)
    stdnt_data = test_data[share_index]
    picked_stdnt_data = [ori_test_data[idx] for idx in share_index]
    num_train = train_data.shape[0]
    teachers_preds = np.zeros([stdnt_data.shape[0], config.nb_labels])

    tau_teachers_preds=[]
    # a weighted teacher predtion with clippling
    for idx in range(len(stdnt_data)):
        if idx % 100 == 0:
            print('idx=', idx)
        query_data = stdnt_data[idx]
        select_teacher = np.random.choice(train_data.shape[0], int(prob * num_train))
        dis = np.linalg.norm(train_data[select_teacher] - query_data, axis=1)
        k_index = select_teacher[np.argsort(dis)[:config.nb_teachers]]
        # sum over the number of teachers, which make it easy to compute their votings
        if config.use_tau:
            tau_teachers_preds.append(tau_limit(train_labels[k_index,:]))
        teachers_preds[idx] = np.sum(train_labels[k_index, :], axis=0)


    teachers_preds = np.asarray(teachers_preds, dtype=np.int32)
    if config.use_tau:
    
        preds_tau = np.asarray(tau_teachers_preds, dtype = np.float32)
        acct.compose_poisson_subsampled_mechanisms(gaussian, prob, coeff=config.stdnt_share)
        count_zero_list = config.nb_teachers * np.ones([config.stdnt_share,config.nb_labels]) - teachers_preds
        idx, stdnt_labels = aggregation.aggregation_knn(teachers_preds, config.gau_scale,count_zero_list=count_zero_list)
    else:    
        acct.compose_poisson_subsampled_mechanisms(gaussian, prob, coeff=config.stdnt_share)
        idx, stdnt_labels = aggregation.aggregation_knn(teachers_preds, config.gau_scale)
    # compute privacy loss
    print("Composition of student  subsampled Gaussian mechanisms gives ", (acct.get_eps(delta), delta))

    # Print accuracy of aggregated label
    #ac_ag_labels = hamming_accuracy(stdnt_labels, test_labels[:config.stdnt_share], torch=False)
    ac_ag_labels = hamming_accuracy(stdnt_labels, test_labels[share_index], torch=False)
    precision = hamming_precision(stdnt_labels, test_labels[share_index], torch=False)
    print("Accuracy of the aggregated labels: " + str(ac_ag_labels))
    print('precision of the aggregated labels'+str(precision))
    current_eps = acct.get_eps(config.delta)
    # Store unused part of test set for use as a test set after student training
    stdnt_test_data = ori_test_data[-1000:]
    stdnt_test_labels = test_labels[-1000:]

    if save:
      # Prepare filepath for numpy dump of labels produced by noisy aggregation
      dir_path = os.path.join(config.save_model, 'knn_num_neighbor_' + str(config.nb_teachers))
      utils.mkdir_if_missing(dir_path)
      filepath = dir_path + '_knn_voting.npy' #NOLINT(long-line)

      # Dump student noisy labels array
      with open(filepath, 'wb') as file_obj:
        np.save(file_obj, teachers_preds)

    return picked_stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels
    #return ori_test_data[:config.stdnt_share], stdnt_labels, stdnt_test_data, stdnt_test_labels


def train_student( nb_teachers):
    """
    This function trains a student using predictions made by an ensemble of
    teachers. The student and teacher models are trained using the same
    neural network architecture.
    :param dataset: string corresponding to celeba
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :return: True if student training went well
    """
    # Call helper function to prepare student data using teacher predictions
    stdnt_dataset = prepare_student_data(nb_teachers, save=True)

    # Unpack the student dataset
    stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels = stdnt_dataset
    dir_path = os.path.join(config.save_model, config.dataset)
    dir_path = os.path.join(dir_path, 'knn_num_neighbor_' + str(config.nb_teachers))
    utils.mkdir_if_missing(dir_path)
    if config.resnet:
        filename = os.path.join(dir_path,str(config.nb_teachers) + '_stdnt_resnet.checkpoint.pth.tar')

    print('stdnt_label used for train', stdnt_labels.shape)
    network.train_each_teacher(config.student_epoch, stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels,
                               filename)
    return True

def main(argv=None):  # pylint: disable=unused-argument
    train_student(config.nb_teachers)



main()

