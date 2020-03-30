import torch
import numpy as np
import config as config
config = config.config


import utils
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import network
#from utils import Hamming_Score as hamming_accuracy
import os
import data_manager
from dataset_loader import ImageDataset
import aggregation
#from utils import Hamming_Score as hamming_accuracy
from utils import hamming_precision as hamming_accuracy
from knn_attribute import tau_limit
import sys
sys.path.append('../dataset/duke')
sys.path.append('../autodp/autodp')
import rdp_bank, dp_acct, rdp_acct, privacy_calibrator
from datafolder.folder import Test_Dataset
nb_teachers = config.nb_teachers
acct = rdp_acct.anaRDPacct()
gaussian = lambda x: rdp_bank.RDP_inde_pate_gaussian({'sigma': int(config.gau_scale/config.tau)}, x)
def ensemble_preds( nb_teachers, stdnt_data):
  """
  Given a dataset, a number of teachers, and some input data, this helper
  function queries each teacher for predictions on the data and returns
  all predictions in a single array. (That can then be aggregated into
  one single prediction per input using aggregation.py (cf. function
  prepare_student_data() below)
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :param stdnt_data: unlabeled student training data
  :return: 3d array (teacher id, sample id, probability per class)
  """


  result_shape = (nb_teachers, len(stdnt_data), config.nb_labels)

  # Create array that will hold result
  result = np.zeros(result_shape, dtype=np.float32)
  # Get predictions from each teacher
  for teacher_id in range(nb_teachers):
    # Compute path of checkpoint file for teacher model with ID teacher_id
    if config.dataset =='celeba':
      dir_path = os.path.join(config.save_model,'pate_num_teacher_'+str(config.nb_teachers))
    elif config.dataset =='market':
      dir_path = os.path.join(config.save_model,'pate_'+config.dataset+str(config.nb_teachers))
    utils.mkdir_if_missing(dir_path)
    filename = os.path.join(dir_path,str(config.nb_teachers) + '_teachers_' + str(teacher_id) + config.arch+'.checkpoint.pth.tar')
    result[teacher_id] = network.pred(stdnt_data, filename)
        
    # This can take a while when there are a lot of teachers so output status
    print("Computed Teacher " + str(teacher_id) + " softmax predictions")
  
  return result


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

  else:
    print("Check value of dataset flag")
    return False

  # Make sure there is data leftover to be used as a test set
  assert config.stdnt_share < len(test_data)

  # Prepare [unlabeled] student training data (subset of test set)
  stdnt_data = test_data[:config.stdnt_share]
  # Compute teacher predictions for student training data
  if config.reuse_vote:
    #reuse previous saved clean votes, but stdnt_share maybe various
    #dir_path = os.path.join(config.save_model,'pate_'+str(config.nb_teachers))
    dir_path = os.path.join(config.save_model,config.dataset)
    dir_path = os.path.join(dir_path,'pate_num_teacher_'+str(config.nb_teachers))
    utils.mkdir_if_missing(dir_path)
    filepath = dir_path + '/_teacher_votes.npy'
    # Prepare filepath for numpy dump of labels produced by noisy aggregation
    teachers_preds = np.load(filepath)
    teachers_preds = teachers_preds[:config.stdnt_share]
    ori_filepath = dir_path + '_ori_teacher_votes.npy'
    ori_teachers_preds = np.load(ori_filepath)
  else:
    teachers_preds = ensemble_preds(nb_teachers, stdnt_data)
    ori_teachers_preds = teachers_preds # in the shape of (nb_teacher, nb_data, dim)
    teachers_preds = np.sum(teachers_preds,axis=0)
    dir_path = os.path.join(config.save_model,config.dataset)
    dir_path = os.path.join(dir_path,'pate_num_teacher_'+str(config.nb_teachers))
    utils.mkdir_if_missing(dir_path)
    filepath = dir_path + '/_teacher_votes.npy'
    ori_filepath = dir_path + '_ori_teacher_votes.npy'
    with open(filepath, mode='wb') as file_obj:
      np.save(file_obj, teachers_preds)
    with open(ori_filepath, mode = 'wb') as file_obj:
      np.save(file_obj, ori_teachers_preds)


  if config.use_tau:
    tau_teachers_preds = np.zeros(teachers_preds.shape)
    for idx in range(len(tau_teachers_preds)):
      tau_teachers_preds[idx] = tau_limit(ori_teachers_preds[:,idx,:])
    
    preds_tau = np.asarray(tau_teachers_preds, dtype = np.float32)
    print('preds_tau',preds_tau[1,])
    count_zero_list = config.nb_teachers * np.ones([config.stdnt_share,config.nb_labels]) - teachers_preds
    print('shape of count_zero', count_zero_list.shape)
    idx, stdnt_labels = aggregation.aggregation_knn(teachers_preds, config.gau_scale,count_zero_list=count_zero_list)
    acct.compose_mechanism(gaussian,coeff=config.stdnt_share)
  else:
    acct.compose_mechanism(gaussian,  coeff=config.stdnt_share)
    idx, stdnt_labels = aggregation.aggregation_knn(teachers_preds, config.gau_scale)
  print('shape of teachers_pred',teachers_preds.shape)
  # Aggregate teacher predictions to get student training labels


  # Print accuracy of aggregated label
  ac_ag_labels = hamming_accuracy(stdnt_labels, test_labels[:config.stdnt_share],torch=False)
  print("Accuracy of the aggregated labels: " + str(ac_ag_labels))
  current_eps = acct.get_eps(config.delta)
  print('eps after data independent composition', current_eps)
  # Store unused part of test set for use as a test set after student training
  stdnt_test_data = test_data[config.stdnt_share:]
  stdnt_test_labels = test_labels[config.stdnt_share:]
  
  return stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels


def train_student(nb_teachers):
  """
  This function trains a student using predictions made by an ensemble of
  teachers. The student and teacher models are trained using the same
  neural network architecture.
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :return: True if student training went well
  """


  # Call helper function to prepare student data using teacher predictions
  stdnt_dataset = prepare_student_data( nb_teachers, save=True)

  # Unpack the student dataset
  stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels = stdnt_dataset

  if config.resnet:
      dir_path = os.path.join(config.save_model,config.dataset)
      dir_path = os.path.join(dir_path,'pate_num_teacher_'+str(config.nb_teachers))
      #dir_path = os.path.join(config.save_model,'pate_'+str(config.nb_teachers))
      utils.mkdir_if_missing(dir_path)
      filename = os.path.join(dir_path, '_stndent_resnet.checkpoint.pth.tar')

  print('stdnt_label used for train',stdnt_labels.shape)
  network.train_each_teacher(config.student_epoch,stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels, filename)



  final_preds = network.pred(stdnt_test_data, filename)

  precision = hamming_accuracy(final_preds, stdnt_test_labels, torch=False)
  print('Precision of student after training: ' + str(precision))

  return True

def main(argv=None): # pylint: disable=unused-argument
  # Run student training according to values specified in flags
  assert train_student( nb_teachers)

main()
