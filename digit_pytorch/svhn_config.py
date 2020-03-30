

import numpy as np
from easydict import EasyDict as edict
config=edict()

config.dataset='svhn'
config.nb_labels = 10
#config.data_dir ='../list'
config.data_dir = '/tmp'
config.save_model='save_model/svhn'
#where to save model
config.resnet = True
config.batch_size = 512
config.hog_path = '/tmp'
config.nb_teachers =300
config.reuse_vote = True
# if set to true, then reuse clean votes
config.extract_feature = 'feature'
#extract_feature = False return original image
#extract feature = feature extract feature from config.save_path
#extract_feature = hog return hog feature
#extract_feature = pca
config.save_path = ''
config.confident = True
config.data_dependent_rdp = False
config.use_tau  = False
#if set tau = True, then we do a clip of multilabel problem
config.network =edict()
config.network.deeper=False
config.network.resnet = True
config.use_cpu=False
config.seed =1
config.arch='resnet50m'
config.workers = 20
config.gpu_devices='0'
config.evaluate =False
config.width = 32
config.height = 32
config.train_batch = 128

config.test_batch = 128
#
config.sigma1 = 85
config.threshold = 210
config.gau_scale = 5
config.use_uda = True # whether use uda for semi-supervised training


# if this is a confidence based methods, sigma1 is used for selection, and gau_scale is added to voting
config.delta = 1e-5
config.stdnt_share =  1000

config.extra = 0

#num of answered queries in students
config.workers = 4
config.stepsize =20
config.use_uda_data = False
config.teacher_epoch = 5
config.student_epoch = 5
config.optim='adam'
config.lr = 3e-4
config.weight_decay =5e-4
config.print_freq = 2
config.gamma = 0.1
config.toy = False

