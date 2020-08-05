
import argparse
import json
import logging
import os
import random
import sys
import time
import warnings
from functools import partial

import pandas as pd
import time, datetime
import gc

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import recall_score
import logging.handlers

import matplotlib.pyplot as plt
import joblib

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import dis_util
import sagemaker_containers
import util

## Apex import package
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example.")


## augmentation for setting
from albumentations import (
    Rotate,HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)
from albumentations.pytorch import ToTensor, ToTensorV2

import dis_util
import sagemaker_containers
import util

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parser_args():
    parser = argparse.ArgumentParser()

    # Default Setting
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    parser.add_argument('--channels-last', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    # Hyperparameter Setting
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--vld_fold_idx', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 200)')

    # APEX Setting for Distributed Training
    parser.add_argument('--apex', type=bool, default=False)
    parser.add_argument('--opt-level', type=str, default='O0')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')
    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')

    # SageMaker Container environment
    parser.add_argument('--hosts', type=list,
                        default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str,
                        default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int,
                        default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--output_data_dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR'))

    args = parser.parse_args()
    return args

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(pretrained=True)
        last_hidden_units = self.model.fc.out_features
        self.classifer_model = nn.Linear(last_hidden_units, 186)
    @staticmethod
    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
        x = self.classifer_model(features)
        return x
    
    
class BangaliDataset(Dataset):
    def __init__(self, imgs, label_df=None, transform=None):
        self.imgs = imgs
        self.label_df = label_df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        
        img_idx = self.label_df.iloc[idx].id
        img = (self.imgs[img_idx]).astype(np.uint8)
        img = 255 - img
    
        img = img[:,:,np.newaxis]
        img = np.repeat(img, 3, axis=2)
        
        if self.transform is not None:
            img = self.transform(image=img)['image']        
        
        if self.label_df is not None:
            label_1 = self.label_df.iloc[idx].grapheme_root
            label_2 = self.label_df.iloc[idx].vowel_diacritic
            label_3 = self.label_df.iloc[idx].consonant_diacritic           
            return img, np.array([label_1, label_2, label_3])        
        else:
            return img
        
        

def _rand_bbox(size, lam):
    '''
    CutMix Helper function.
    Retrieved from https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    '''
    W = size[2]
    H = size[3]
    # 폭과 높이는 주어진 이미지의 폭과 높이의 beta distribution에서 뽑은 lambda로 얻는다
    cut_rat = np.sqrt(1. - lam)
    
    # patch size 의 w, h 는 original image 의 w,h 에 np.sqrt(1-lambda) 를 곱해준 값입니다.
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # patch의 중심점은 uniform하게 뽑힘
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def _set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    mx.random.seed(seed)

def _get_images(args, data_type='train'):

    logger.info("=== Getting Labels ===")
    logger.info(args.data_dir)
    
    label_df = pd.read_csv(os.path.join(args.data_dir, 'train_folds.csv'))
    #label_df = pd.read_csv(f'{train_dir}/train_folds.csv')
     
    trn_fold = [i for i in range(args.num_folds) if i not in [args.vld_fold_idx]]
    vld_fold = [args.vld_fold_idx]

    trn_idx = label_df.loc[label_df['fold'].isin(trn_fold)].index
    vld_idx = label_df.loc[label_df['fold'].isin(vld_fold)].index

    logger.info("=== Getting Images ===")    
    #files = sorted(glob2.glob(f'{train_dir}/{data_type}_*.parquet'))
    files = [f'{args.data_dir}/{data_type}_image_data_{i}.parquet' for i in range(4)]
    logger.info(files)
    
    image_df_list = [pd.read_parquet(f) for f in files]
    imgs = [df.iloc[:, 1:].values.reshape(-1, args.height, args.width) for df in image_df_list]
    del image_df_list
    gc.collect()
    args.imgs = np.concatenate(imgs, axis=0)
    
    args.trn_df = label_df.loc[trn_idx]
    args.vld_df = label_df.loc[vld_idx]
    
    return args 


def _get_train_data_loader(args, **kwargs):
    logger.info("Get train data loader")
    train_transforms = Compose([
        Rotate(20),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=.1),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),            
            ], p=0.3),
            HueSaturationValue(p=0.3),
        ToTensor()
        ], p=1.0)
    
    dataset = BangaliDataset(imgs=args.imgs, label_df=args.trn_df, transform=train_transforms)
    train_sampler = data.distributed.DistributedSampler(
        dataset, num_replicas=int(args.world_size), rank=int(args.rank)) if args.multigpus_distributed else None
    return data.DataLoader(dataset, batch_size=args.batch_size, shuffle=train_sampler is None,
                                       sampler=train_sampler, collate_fn=dis_util.fast_collate, **kwargs), train_sampler


def _get_test_data_loader(args, **kwargs):
    logger.info("Get test data loader")   

    dataset = BangaliDataset(imgs=args.imgs, label_df=args.vld_df)
    val_sampler = data.distributed.DistributedSampler(dataset) if args.multigpus_distributed else None
    return data.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, 
                           sampler=val_sampler, collate_fn=dis_util.fast_collate, **kwargs)


def train(current_gpu, args):
    best_acc1 = -1
    model_history = {}
    model_history = util.init_modelhistory(model_history)
    train_start = time.time()

    ## choose model from pytorch model_zoo
    model = util.torch_model(args.model_name, pretrained=True)
    loss_fn = nn.CrossEntropyLoss().cuda()

    ## distributed_setting 
    model, args = dis_util.dist_setting(current_gpu, model, loss_fn, args)

    ## CuDNN library will benchmark several algorithms and pick that which it found to be fastest
    cudnn.benchmark = False if args.seed else True

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.apex:
        model, optimizer = dis_util.apex_init(model, optimizer, args)
    
    
#     args.collate_fn = partial(dis_util.fast_collate, memory_format=args.memory_format)
   
    args = _get_images(args, data_type='train')
    train_loader, train_sampler = _get_train_data_loader(args, **args.kwargs)
    test_loader = _get_test_data_loader(args, **args.kwargs)

    logger.info("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.info("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))

    for epoch in range(1, args.num_epochs + 1):
        ## 
        batch_time = util.AverageMeter('Time', ':6.3f')
        data_time = util.AverageMeter('Data', ':6.3f')
        losses = util.AverageMeter('Loss', ':.4e')
        top1 = util.AverageMeter('Acc@1', ':6.2f')
        top5 = util.AverageMeter('Acc@5', ':6.2f')
        progress = util.ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        model.train()
        end = time.time()
        
        ## Set epoch count for DistributedSampler
        if args.multigpus_distributed:
            train_sampler.set_epoch(epoch)
        
        
        prefetcher = util.data_prefetcher(train_loader)
        input, target = prefetcher.next()
        batch_idx = 0
        while input is not None:

            batch_idx += 1
            
            if args.prof >= 0 and batch_idx == args.prof:
                print("Profiling begun at iteration {}".format(batch_idx))
                torch.cuda.cudart().cudaProfilerStart()
                
            if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(batch_idx))

            util.adjust_learning_rate(optimizer, epoch, batch_idx, len(train_loader), args)
            
            ##### DATA Processing #####
            targets_gra = targets[:, 0]
            targets_vow = targets[:, 1]
            targets_con = targets[:, 2]

            # 50%의 확률로 원본 데이터 그대로 사용    
            if np.random.rand() < 0.5:
                logits = model(input)
                grapheme = logits[:, :168]
                vowel = logits[:, 168:179]
                cons = logits[:, 179:]

                loss1 = loss_fn(grapheme, targets_gra)
                loss2 = loss_fn(vowel, targets_vow)
                loss3 = loss_fn(cons, targets_con) 

            else:

                lam = np.random.beta(1.0, 1.0) 
                rand_index = torch.randperm(input.size()[0])
                shuffled_targets_gra = targets_gra[rand_index]
                shuffled_targets_vow = targets_vow[rand_index]
                shuffled_targets_con = targets_con[rand_index]

                bbx1, bby1, bbx2, bby2 = _rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                # 픽셀 비율과 정확히 일치하도록 lambda 파라메터 조정  
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                
                logits = model(input)
                grapheme = logits[:,:168]
                vowel = logits[:, 168:179]
                cons = logits[:, 179:]

                loss1 = loss_fn(grapheme, targets_gra) * lam + loss_fn(grapheme, shuffled_targets_gra) * (1. - lam)
                loss2 = loss_fn(vowel, targets_vow) * lam + loss_fn(vowel, shuffled_targets_vow) * (1. - lam)
                loss3 = loss_fn(cons, targets_con) * lam + loss_fn(cons, shuffled_targets_con) * (1. - lam)

            loss = 0.5 * loss1 + 0.25 * loss2 + 0.25 * loss3    
            trn_loss.append(loss.item())
            running_loss += loss.item()
            
            #########################################################
            
            
#             # compute output
#             if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
#             output = model(input)
#             if args.prof >= 0: torch.cuda.nvtx.range_pop()
#             loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            
            if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
            if args.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if args.prof >= 0: torch.cuda.nvtx.range_pop()

            if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
            optimizer.step()
            if args.prof >= 0: torch.cuda.nvtx.range_pop()
            # Printing vital information
            if (batch_idx + 1) % (args.log_interval) == 0:
                s = f'[Epoch {epoch} Batch {batch_idx+1}/{len(train_loader)}] ' \
                f'loss: {running_loss / args.log_interval:.4f}'
                print(s)
                running_loss = 0
                
                
#             if True or batch_idx % args.log_interval == 0:
#                 # Every print_freq iterations, check the loss, accuracy, and speed.
#                 # For best performance, it doesn't make sense to print these metrics every
#                 # iteration, since they incur an allreduce and some host<->device syncs.

#                 # Measure accuracy
#                 prec1, prec5 = util.accuracy(output.data, target, topk=(1, 5))

#                 # Average loss and accuracy across processes for logging
#                 if args.multigpus_distributed:
#                     reduced_loss = dis_util.reduce_tensor(loss.data, args)
#                     prec1 = dis_util.reduce_tensor(prec1, args)
#                     prec5 = dis_util.reduce_tensor(prec5, args)
#                 else:
#                     reduced_loss = loss.data

#                 # to_python_float incurs a host<->device sync
#                 losses.update(to_python_float(reduced_loss), input.size(0))
#                 top1.update(to_python_float(prec1), input.size(0))
#                 top5.update(to_python_float(prec5), input.size(0))
                
#                 ## Waiting until finishing operations on GPU (Pytorch default: async)
#                 torch.cuda.synchronize()
#                 batch_time.update((time.time() - end)/args.log_interval)
#                 end = time.time()

#                 if current_gpu == 0:
#                     print('Epoch: [{0}][{1}/{2}]  '
#                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
#                           'Speed {3:.3f} ({4:.3f})  '
#                           'Loss {loss.val:.10f} ({loss.avg:.4f})  '
#                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
#                           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                               epoch, batch_idx, len(train_loader),
#                               args.world_size*args.batch_size/batch_time.val,
#                               args.world_size*args.batch_size/batch_time.avg,
#                               batch_time=batch_time,
#                               loss=losses, top1=top1, top5=top5))
#                     model_history['epoch'].append(epoch)
#                     model_history['batch_idx'].append(batch_idx)
#                     model_history['batch_time'].append(batch_time.val)
#                     model_history['losses'].append(losses.val)
#                     model_history['top1'].append(top1.val)
#                     model_history['top5'].append(top5.val)
                    
#                 if args.prof >= 0: torch.cuda.nvtx.range_push("prefetcher.next()")
#                 input, target = prefetcher.next()
#                 if args.prof >= 0: torch.cuda.nvtx.range_pop()

#                 # Pop range "Body of iteration {}".format(i)
#                 if args.prof >= 0: torch.cuda.nvtx.range_pop()
                    

#                 if args.prof >= 0 and batch_idx == args.prof + 10:
#                     print("Profiling ended at iteration {}".format(batch_idx))
#                     torch.cuda.cudart().cudaProfilerStop()
#                     quit()
               
        acc1 = validate(test_loader, model, loss_fn, epoch, model_history, args)
        
        print(" ****  acc1 :{}".format(acc1))
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multigpus_distributed or (args.multigpus_distributed and args.rank % args.num_gpus == 0):
            util.save_history(os.path.join(args.output_data_dir,
                          'model_history.p'), model_history)

            util.save_model({
                'epoch': epoch + 1,
                'model_name': args.model_name,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'class_to_idx' : train_loader.dataset.class_to_idx,
            }, is_best, args.model_dir)


def validate(val_loader, model, loss_fn, epoch, model_history, args):
    batch_time = util.AverageMeter('Time', ':6.3f')
    losses = util.AverageMeter('Loss', ':.4e')
    top1 = util.AverageMeter('Acc@1', ':6.2f')
    top5 = util.AverageMeter('Acc@5', ':6.2f')
    progress = util.ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    
    val_loss = []
    val_true = []
    val_pred = []    
    
    
    
    # switch to evaluate mode
    model.eval()
    end = time.time()

    prefetcher = util.data_prefetcher(val_loader)
    input, target = prefetcher.next()
    batch_idx = 0
    while input is not None:
        batch_idx += 1
    
        # compute output
        with torch.no_grad():
#             data = data.contiguous(memory_format=args.memory_format)
#             target = target.contiguous()
#             data = data.cuda(non_blocking=True)
#             target = target.cuda(non_blocking=True)



                logits = model(input)
                grapheme = logits[:,:168]
                vowel = logits[:, 168:179]
                cons = logits[:, 179:]

                loss= 0.5* loss_fn(grapheme, targets[:,0]) + 0.25*loss_fn(vowel, targets[:,1]) + \
                0.25*loss_fn(vowel, targets[:,2])
                val_loss.append(loss.item())

                grapheme = grapheme.cpu().argmax(dim=1).data.numpy()
                vowel = vowel.cpu().argmax(dim=1).data.numpy()
                cons = cons.cpu().argmax(dim=1).data.numpy()

                val_true.append(targets.cpu().numpy())
                val_pred.append(np.stack([grapheme, vowel, cons], axis=1))                

        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)
        val_loss = np.mean(val_loss)
        trn_loss = np.mean(trn_loss)

        score_g = recall_score(val_true[:,0], val_pred[:,0], average='macro')
        score_v = recall_score(val_true[:,1], val_pred[:,1], average='macro')
        score_c = recall_score(val_true[:,2], val_pred[:,2], average='macro')
        final_score = np.average([score_g, score_v, score_c], weights=[2,1,1])

        # Printing vital information
        s = f'[Epoch {epoch}] ' \
        f'trn_loss: {trn_loss:.4f}, vld_loss: {val_loss:.4f}, score: {final_score:.4f}, ' \
        f'score_each: [{score_g:.4f}, {score_v:.4f}, {score_c:.4f}]'          
        print(s)

        ################################################################################
        # ==> Save checkpoint and training stats
        ################################################################################        
        if final_score > best_score:
            best_score = final_score
            state_dict = model.cpu().state_dict()
            model = model.cuda()
            torch.save(state_dict, os.path.join(args.model_output_dir, 'model.pt'))

        # Record all statistics from this epoch
        training_stats.append(
            {
                'epoch': epoch + 1,
                'trn_loss': trn_loss,
                'trn_time': trn_time,            
                'val_loss': val_loss,
                'score': final_score,
                'score_g': score_g,
                'score_v': score_v,
                'score_c': score_c            
            }
        )      
        
        # === Save Model Parameters ===
        logger.info("Model successfully saved at: {}".format(args.model_output_dir))      
        
        
#         # measure accuracy and record loss
#         prec1, prec5 = util.accuracy(output.data, target, topk=(1, 5))

#         if args.multigpus_distributed:
#             reduced_loss = dis_util.reduce_tensor(loss.data, args)
#             prec1 = dis_util.reduce_tensor(prec1, args)
#             prec5 = dis_util.reduce_tensor(prec5, args)
#         else:
#             reduced_loss = loss.data

#         losses.update(to_python_float(reduced_loss), input.size(0))
#         top1.update(to_python_float(prec1), input.size(0))
#         top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

#         # TODO:  Change timings to mirror train().
#         if args.current_gpu == 0 and batch_idx % args.log_interval == 0:
#             print('Test: [{0}/{1}]  '
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
#                   'Speed {2:.3f} ({3:.3f})  '
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})  '
#                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
#                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                       batch_idx, len(val_loader),
#                       args.world_size * args.batch_size / batch_time.val,
#                       args.world_size * args.batch_size / batch_time.avg,
#                       batch_time=batch_time, loss=losses,
#                       top1=top1, top5=top5))
#             model_history['val_epoch'].append(epoch)
#             model_history['val_batch_idx'].append(batch_idx)
#             model_history['val_batch_time'].append(batch_time.val)
#             model_history['val_losses'].append(losses.val)
#             model_history['val_top1'].append(top1.val)
#             model_history['val_top5'].append(top5.val)
#         input, target = prefetcher.next()

#     print('  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
#           .format(top1=top1, top5=top5))
#     model_history['val_avg_epoch'].append(epoch)
#     model_history['val_avg_batch_time'].append(batch_time.avg)
#     model_history['val_avg_losses'].append(losses.avg)
#     model_history['val_avg_top1'].append(top1.avg)
#     model_history['val_avg_top5'].append(top5.avg)
#     return top1.avg


def main():
    args = parser_args()
    args.use_cuda = args.num_gpus > 0
    print("args.use_cuda : {} , args.num_gpus : {}".format(
        args.use_cuda, args.num_gpus))
    args.kwargs = {'num_workers': 4,
                   'pin_memory': True} if args.use_cuda else {}
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    dis_util.dist_init(train, args)


if __name__ == '__main__':
    main()
