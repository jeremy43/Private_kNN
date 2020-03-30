

import numpy as np
from easydict import EasyDict as edict
config=edict()

config.dataset='celeba'
config.nb_labels = 40
config.data_dir ='../list'
#config.data_dir = '/tmp'
config.save_model='save_model'
#where to save model
config.resnet = True
config.batch_size = 256
config.hog_path = '/tmp'
config.nb_teachers =5

config.reuse_vote = True
# if set to true, then reuse clean votes
config.extract_feature = 'feature'
#extract_feature = False return original image
#extract feature = feature extract feature from config.save_path
#extract_feature = hog return hog feature
#extract_feature = pca
config.save_path = ''
config.confident = False

config.tau = 10
config.use_tau  = True
#if set tau = True, then we do a clip of multilabel problem
config.network =edict()
config.network.deeper=False
config.network.resnet = True
config.use_cpu=False
config.seed =1
config.arch='resnet50m'
config.workers = 20
config.gpu_devices='0,1'
config.evaluate =False
config.width = 100
config.height = 100
config.train_batch = 256
config.test_batch = 256
config.sigma1 = 50
config.gau_scale = 50
# if this is a confidence based methods, sigma1 is used for selection, and gau_scale is added to voting
config.delta = 1e-6
config.stdnt_share = 500
#num of answered queries in students
config.workers = 4
config.stepsize =20
config.teacher_epoch = 1
config.student_epoch = 18
config.optim='adam'
config.lr = 4e-4
config.weight_decay =5e-4
config.print_freq = 2
config.gamma = 0.1
config.toy = False

#whether use toy sample for debug
config.toy_data_path = "celeba_toy_2000.txt"
