
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
import market_config as config
config = config.config
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataset_loader import ImageDataset
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, DeepSupervision
from utils import AverageMeter, Logger, save_checkpoint, Hamming_Score
from optimizers import init_optim
from utils import hamming_precision as hamming_precision
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



    transform_train = T.Compose([
        T.Random2DTranslation(config.height, config.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((config.height, config.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False
    trainloader = DataLoader(
        ImageDataset(train_data, label=train_label, transform=transform_train),
        batch_size=config.train_batch, shuffle=True, num_workers=config.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    testloader = DataLoader(
        ImageDataset(test_data,label=test_label, transform=transform_test),
        batch_size=config.test_batch, shuffle=False, num_workers=config.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format('resnet50m'))
    model = models.init_model(name=config.arch, num_classes=config.nb_labels, loss={'xent'}, use_gpu=use_gpu)
    if use_gpu:
        model = nn.DataParallel(model).cuda()
    criterion = nn.MultiLabelSoftMarginLoss()

    optimizer = init_optim(config.optim, model.parameters(), config.lr, config.weight_decay)

    if config.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.stepsize, gamma=config.gamma)


    print("==> Start training")

    start_time = time.time()
    for epoch in range( num_epoch):
        train(epoch, model, criterion, optimizer, trainloader, use_gpu)
        if config.stepsize > 0: scheduler.step()
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
        if isinstance(outputs, tuple):
            loss = DeepSupervision(criterion, outputs, pids)
        else:
            loss = criterion(outputs, pids)
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
            predAs = torch.round(torch.sigmoid(predA))
            hamming_score.append(Hamming_Score(pids, predAs))
            precision.append(hamming_precision(pids,predAs))
            pred_list.append(predA)
        predA_t = (((torch.cat(pred_list, 0)).float()).numpy()).tolist()
        mean_hamming_score = np.mean(hamming_score)
        mean_precision = np.mean(precision)
        print("mean_hamminng_score: {:.2%}".format(mean_hamming_score))
        print('mean precision count one {:.2%}'.format(mean_precision))
        return mean_hamming_score


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



    transform_test = T.Compose([
        T.Resize((config.height, config.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False


    testloader = DataLoader(
        ImageDataset(data, transform=transform_test),
        batch_size=config.test_batch, shuffle=False, num_workers=config.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    model = models.init_model(name=config.arch, num_classes=config.nb_labels, loss={'xent'}, use_gpu=use_gpu)

    checkpoint = torch.load(save_path)
    #model.load_state_dict(checkpoint['state_dict'])
    #original is checkpoint as a set (from celeba.py)
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
            end = time.time()
            features, predA = model(imgs)
            predA = predA.cpu()
            # print('features shape {} predA shape'.format(features.shape, predA.shape))
            float_logit_list.append(torch.sigmoid(predA))
            if return_feature is True:
                feature_list.append(features.cpu())

            predAs = torch.round(torch.sigmoid(predA))
            pred_list.append(predAs)
        predA_t = (((torch.cat(pred_list, 0)).float()).numpy()).tolist()
        predA_t = np.array(predA_t)
        float_logit_list = (((torch.cat(float_logit_list, 0)).float()).numpy()).tolist()
        float_logit_list = np.array(float_logit_list)
    if return_feature == True:
        feature_list = (((torch.cat(feature_list, 0)).float()).numpy()).tolist()
        feature_list = np.array(feature_list)
        #print('return feature shape',feature_list.shape)
        #return float_logit_list
        return feature_list
        return predA_t
    else:
        return predA_t

