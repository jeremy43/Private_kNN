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

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import data_manager
from dataset_loader import ImageDataset
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, DeepSupervision
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from optimizers import init_optim

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
# Datasets
parser.add_argument('--root', type=str, default='.', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='celeba',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=100,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=100,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")
# CUHK03-specific setting
#parser.add_argument('--cuhk03-labeled', action='store_true',
#                   help="whether to use labeled images, if false, detected images are used (default: False)")
#parser.add_argument('--cuhk03-classic-split', action='store_true',
#                   help="whether to use classic split by Li et al. CVPR'14 (default: False)")
#parser.add_argument('--use-metric-cuhk03', action='store_true',
#                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=128, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=20, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50m', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=1, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--load-weights', type=str, default='',
                    help="load pretrained weights but ignores layers that don't match in size")
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=2,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    print('whether evaluate', args.evaluate)
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset,
    )

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
       # T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    pin_memory = True if use_gpu else False
    trainloader = DataLoader(
        ImageDataset(dataset.train_data,dataset.train_label, transform=transform_train),
        batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    testloader = DataLoader(
        ImageDataset(dataset.test_data,dataset.test_label, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=40, loss={'xent'}, use_gpu=use_gpu)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.MultiLabelSoftMarginLoss()

    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.load_weights:
        # load pretrained weights but ignore layers that don't match in size
       checkpoint = torch.load(args.load_weights)
       pretrain_dict = checkpoint['state_dict']
       model_dict = model.state_dict()
       pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
       model_dict.update(pretrain_dict)
       model.load_state_dict(model_dict)
       print("Loaded pretrained weights from '{}'".format(args.load_weights))


    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, testloader, use_gpu)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)
        
        if args.stepsize > 0: scheduler.step()
        
        if (epoch+1) > args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, testloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'hamming_score': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print("==> Best Hamming Score {:.3%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


# Hamming score computation on only positives
def Hamming_Score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    #print('y_true={} y_pred={}'.format(y_true.shape, y_pred.shape))
    acc_list = []
    for i in range(y_true.shape[0]):
        summary = y_true[i] == y_pred[i].double()
        
        num = np.sum(summary.numpy())
        tmp_a = num/float(len(y_true[i]))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def train(epoch, model, criterion, optimizer, trainloader, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, pids) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
	#print('img.shape {}  pids.shape'.format(imgs.shape, pids.shape))
        # measure data loading time
        pids = pids.type(torch.cuda.FloatTensor)
        data_time.update(time.time() - end)
        outputs  = model(imgs)
        if isinstance(outputs, tuple):
            loss = DeepSupervision(criterion, outputs, pids)
        else:
            loss = criterion(outputs, pids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
		  #'Identity Loss {lossI.val:.4f} ({lossI.avg:.4f})\t'
		  #'Attribute Loss {lossA.val:.4f} ({lossA.avg:.4f})\t'
                  'Total Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def test(model, testloader, use_gpu):
    batch_time = AverageMeter()
    
    model.eval()

    with torch.no_grad():
        gf, g_pids, g_camids, hamming_score, pred_list = [], [], [], [], []
        for batch_idx, (imgs, pids) in enumerate(testloader):
            if use_gpu: 
                imgs = imgs.cuda()
            end = time.time()	
            features,predA = model(imgs)
	    #print('predA',predA.shape)
            #features,predA = model(imgs)
            batch_time.update(time.time() - end)
            predA = predA.cpu()
            predAs = torch.round(torch.sigmoid(predA))
            hamming_score.append(Hamming_Score(pids,predAs))
            pred_list.append(predA)
        predA_t = (((torch.cat(pred_list,0)).float()).numpy()).tolist()
        mean_hamming_score = np.mean(hamming_score)
        print("mean_hamminng_score: {:.2%}".format(mean_hamming_score))
        return mean_hamming_score


if __name__ == '__main__':
    main()
