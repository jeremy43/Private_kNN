from __future__ import absolute_import
import os
import sys
import svhn_config as config
config = config.config
import errno
import pickle
import shutil
import json
import os.path as osp
import numpy as np
import torch
import scipy
sys.path.append('.')
from sklearn.decomposition import PCA, KernelPCA
import math
def convert_vat(idx_keep, remain_idx,test_data, test_labels, noisy_labels):
    
    log = {}
    log['keep_idx'] = idx_keep
    log['labeled_train_images'] = test_data[:config.stdnt_share,:][idx_keep]
    print('labeled train images shape', log['labeled_train_images'].shape)
    log['labeled_train_labels'] = noisy_labels
    log['train_images'] = test_data[:-1000]
    log['train_labels'] = test_labels[0:-1000]
    print('unlabeled shape', log['train_labels'].shape)
    log['test_images'] = test_data[:-1000]
    log['test_labels'] = test_labels[:-1000]
    
    save_path = os.path.join('../uda/log','svhn_query='+str(len(log['labeled_train_images']))+'.pkl')
    print('save_path_for_vat',save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(log,f)





def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def pca(teacher, student):
  pca = PCA(n_components=200)
  pca.fit(teacher)
  max_component = pca.components_.T
  teacher = np.dot(teacher, max_component)
  student = np.dot(student, max_component)
  return  student, teacher

def Hamming_Score(y_true, y_pred, torch=True,cate=False):
    """
    torch = true mean y_pred is torch tensor
    if torch=false mean y_pred=numpy
    """
    acc_list = []
    if torch:
        from sklearn.metrics import accuracy_score
    for i in range(len(y_true)):
        if torch:

            summary = y_true[i] == y_pred[i].double()

            num = np.sum(summary.numpy())
        else:
            summary = y_true[i] == y_pred[i]
            num = np.sum(summary)
        tmp_a = num / float(len(y_true[i]))
        acc_list.append(tmp_a)
    #print('mean score from hamming',np.mean(acc_list))
    return np.mean(acc_list)
def hamming_precision(y_true, y_pred,torch = True, cate = True):
    acc_list = []
    if torch:
        from sklearn.metrics import accuracy_score
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
    for i in range(len(y_true)):
        
        set_true = set( np.where(y_true[i]==1)[0] )
        set_pred = set( np.where(y_pred[i]==1)[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
            float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)
class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def save_hog(data, path):

  from skimage import color
  import pickle
  #print('save_hog train_data shape', data.shape)
  from skimage.feature import hog

  train_gray = [color.rgb2gray(i) for i in data]
  hog_data = [hog(img, orientations=8, block_norm='L2') for img in train_gray]
  hog_data = np.array(hog_data, dtype = np.float32)

  with open(path, 'wb') as f:
    pickle.dump(hog_data, f)


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def set_bn_to_eval(m):
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

