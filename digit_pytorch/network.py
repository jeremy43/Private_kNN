
from __future__ import print_function, absolute_import
import os, csv
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import scipy.io as sio
from sklearn.metrics import hamming_loss
import svhn_config as config
config = config.config
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import lr_scheduler
import math
from dataset_loader import ImageDataset
import transforms as T
import models
from utils import AverageMeter, Logger, save_checkpoint, Hamming_Score
from optimizers import init_optim

def learning_rate(init, epoch):
    step = 25000*epoch
    #step = int(600000/config.nb_teachers)*epoch
    optim_factor = 0
    if(step >150000 ):
        optim_factor = 3
    elif(step > 100000):
        optim_factor = 2
    elif(step > 50000):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

def train_each_teacher(num_epoch,train_data, train_label,test_data,test_label,save_path):

    torch.manual_seed(config.seed)
    print('len of train_data in network', len(train_data))
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    print('it is training now')
    use_gpu = torch.cuda.is_available()
    if config.use_cpu: use_gpu = False
    print('whether evaluate', config.evaluate)

    if use_gpu:
        print("Currently using GPU {}".format(config.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(config.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")


    if config.dataset == 'mnist':
        transform_train = T.Compose([
            T.Random2DTranslation(config.height, config.width),
            #T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.1307], std=[0.3081]),
        ])

        transform_test = T.Compose([
            T.Resize((config.height, config.width)),
            T.ToTensor(),
            T.Normalize(mean=[0.1307], std=[0.3081]),
        ])
    else:
        transform_train = T.Compose([
            #T.Random2DTranslation(config.height, config.width),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            #T.Resize(32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = T.Compose([
            T.ToPILImage(),
            #T.Resize(32),
            #T.Resize((config.height, config.width,3)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    pin_memory = True if use_gpu else False
    print('train_data',len(train_data), 'train_label', len(train_label))
    trainloader = DataLoader(
        ImageDataset(train_data, label=train_label, transform=transform_train),
        batch_size=config.train_batch, shuffle=True, num_workers=config.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    testloader = DataLoader(
        ImageDataset(test_data,label=test_label, transform=transform_test),
        batch_size=config.test_batch, shuffle=False, num_workers=config.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    model = models.init_model(name=config.arch, num_classes=config.nb_labels, loss={'xent'}, use_gpu=use_gpu)
    if use_gpu:
        model = nn.DataParallel(model).cuda()
    criterion =torch.nn.CrossEntropyLoss()

    #optimizer = init_optim(config.optim, model.parameters(), config.lr, config.weight_decay)
    #if config.stepsize > 0:
    #    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.stepsize, gamma=config.gamma)


    print("==> Start training")

    start_time = time.time()
    for epoch in range( num_epoch):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate(config.lr,epoch), momentum=0.9, weight_decay=0.0005)
        print('\n=> Training Epoch #%d, LR=%.4f' %(epoch,learning_rate(config.lr, epoch)))
        train(epoch, model, criterion, optimizer, trainloader, use_gpu)
        #if config.stepsize > 0: scheduler.step()
        rank1 = test(model, testloader, use_gpu)
    
    rank1 = test(model, testloader, use_gpu)

    if use_gpu:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    print('save model',save_path)
    torch.save(state_dict, save_path)

    #print("==>  Hamming Score {:.3%}".format(rank1))

    elapsed = round(time.time() - start_time)


    print("Finished. Training time (h:m:s): {}.".format(elapsed))


# Hamming score computation on only positives



def train(epoch, model, criterion, optimizer, trainloader, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    print('len of trainloader',len(trainloader))
    for batch_idx, (imgs, pids) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        pids = pids.type(torch.cuda.FloatTensor)
        outputs = model(imgs)
        #print('outputs shape', outputs.shape, 'label shape', pids.shape)
        #loss = F.nll_loss(outputs, pids.long())
        loss = criterion(outputs, pids.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        losses.update(loss.item(), pids.size(0))

        if (batch_idx + 1) % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Total Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader), loss=losses))


def test(model, testloader, use_gpu):
    batch_time = AverageMeter()

    model.eval()
    total = 0
    correct =0
    with torch.no_grad():
        hamming_score, pred_list = [], []
        precision =[]
        for batch_idx, (imgs, pids) in enumerate(testloader):
            if use_gpu:
                imgs = imgs.cuda()
            end = time.time()
            features, predA = model(imgs)
            # print('predA',predA.shape)
            # features,predA = model(imgs)
            batch_time.update(time.time() - end)
            predA = predA.cpu()
            _, predicted = torch.max(predA.data, 1)
            total += pids.size(0)
            #print('predA',predicted,'  pids', pids)
            
            correct += (predicted == pids).sum().item()
        precision = correct*1.0 /total
        print(' precision count one {:.2%}'.format(precision))
        return precision


def pred(data,save_path,return_feature = False):
    torch.manual_seed(config.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    use_gpu = torch.cuda.is_available()
    if config.use_cpu: use_gpu = False

    if use_gpu:
        print("Currently using GPU {}".format(config.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(config.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")


    if config.dataset =='mnist':
        transform_test = T.Compose([
        T.Resize((config.height, config.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.1307], std=[0.3081]),])
    else:
        transform_test = T.Compose([
        T.ToPILImage(),
        T.Resize((config.height, config.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    pin_memory = True if use_gpu else False


    testloader = DataLoader(
        ImageDataset(data, transform=transform_test),
        batch_size=512, shuffle=False, num_workers=config.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    model = models.init_model(name=config.arch, num_classes=config.nb_labels, loss={'xent'}, use_gpu=use_gpu)

    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint)
    if use_gpu:
        model = nn.DataParallel(model).cuda()
    model.eval()
    
    with torch.no_grad():
        hamming_score, pred_list , feature_list= [], [],[]
        float_logit_list =[]
        for batch_idx, (imgs) in enumerate(testloader):
            if use_gpu:
                imgs = imgs.cuda()
            if batch_idx % 50 ==0:
                print('batch {}/{}', batch_idx, len(testloader))
            end = time.time()
            features, predA = model(imgs)
            predA = predA.cpu()
            #print('features shape {} predA shape'.format(features.shape, predA.shape))
            float_logit_list.append(torch.sigmoid(predA))
            if return_feature is True:
                feature_list.append(features.cpu())
            _, predicted = torch.max(predA.data, 1)
            #print('predAs', predicted)
            pred_list.append(predicted)
        predA_t = (((torch.cat(pred_list, 0)).float()).numpy()).tolist()
        predA_t = np.array(predA_t)
        #float_logit_list = (((torch.cat(float_logit_list, 0)).float()).numpy()).tolist()
        #float_logit_list = np.array(float_logit_list)
    if return_feature == True:
        feature_list = (((torch.cat(feature_list, 0)).float()).numpy()).tolist()
        feature_list = np.array(feature_list)
        return feature_list
        return predA_t
    else:
        return predA_t

