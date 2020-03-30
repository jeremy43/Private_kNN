

import numpy as np
from easydict import EasyDict as edict
config=edict()

config.dataset='duke'
config.attribute_only = True
config.nb_labels =30 #delete age

config.num_train_pids = 751
config.data_dir ='../dataset'
config.save_model='save_model'
#where to save model
config.resnet = True
config.hog_path = '/tmp'
config.nb_teachers = 300
config.reuse_vote = True
# if set to true, then reuse clean votes. The clean votes are the original votes of each teacher regarding every query in the public,
# noting that once you change the number of public queries, we need to set reuse_vote = false so as to record teachers' votes again
config.extract_feature = 'feature'
#extract_feature = False return original image
#extract feature = feature extract feature from config.save_path
#extract_feature = hog return hog feature
#extract_feature = pca
config.save_path = ''
config.confident = False

config.tau = 5
config.use_tau  = True
#if set tau = True, then we do a clip of multilabel problem
config.network =edict()
config.network.deeper=False
config.network.resnet = True
config.use_cpu=False
config.seed =1
#config.arch = 'reset50ma2'
config.arch = 'resnet50'
config.workers = 20
config.gpu_devices='1,2'
config.evaluate =False
config.num_down_att = 10
config.num_up_att = 9

config.width = 144
config.height = 288
config.batch_size = 64
config.train_batch = 128
config.test_batch = 128
#sigma1 is used for noisy screening and gau_scale is used for noisy aggregation
#we do not need the noisy screening for muli-label tasks
config.sigma1 = 200  
config.gau_scale = 200
# if this is a confidence based methods, sigma1 is used for selection, and gau_scale is added to voting
#delta is used for the DP defnition
config.delta = 1e-5
config.stdnt_share = 100
#num of answered queries in students
config.workers = 4
config.stepsize =20
#teacher_epoch is the number of epoch to train each teacher model
#student_epoch is the number of epoch to re-train a student model in the public domain
config.teacher_epoch = 12
config.student_epoch =12
config.optim='adam'
config.lr = 1e-4
config.weight_decay =5e-4
config.print_freq = 2
config.gamma = 0.1

