
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import time, datetime
import gc

import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import Dataset
from sklearn.metrics import recall_score
import logging
import logging.handlers

import matplotlib.pyplot as plt
import joblib


# Import SMDataParallel PyTorch Modules
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
import smdistributed.dataparallel.torch.distributed as dist
dist.init_process_group()


HEIGHT = 137
WIDTH = 236
BATCH_SIZE = 256
NUM_WORKERS = 4

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


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
        
        
def _set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    mx.random.seed(seed)

def _get_images(train_dir, num_folds=5, vld_fold_idx=4, data_type='train'):

    logger.info("=== Getting Labels ===")
    logger.info(train_dir)
    
    label_df = pd.read_csv(os.path.join(train_dir, 'train_folds.csv'))
    #label_df = pd.read_csv(f'{train_dir}/train_folds.csv')
     
    trn_fold = [i for i in range(num_folds) if i not in [vld_fold_idx]]
    vld_fold = [vld_fold_idx]

    trn_idx = label_df.loc[label_df['fold'].isin(trn_fold)].index
    vld_idx = label_df.loc[label_df['fold'].isin(vld_fold)].index

    logger.info("=== Getting Images ===")    
    files = [f'{train_dir}/{data_type}_image_data_{i}.feather' for i in range(4)]
    logger.info(files)
    
    image_df_list = [pd.read_feather(f) for f in files]
    imgs = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    del image_df_list
    gc.collect()
    imgs = np.concatenate(imgs, axis=0)
    
    trn_df = label_df.loc[trn_idx]
    vld_df = label_df.loc[vld_idx]
    
    return imgs, trn_df, vld_df       
                           
                           
def _get_data_loader(imgs, trn_df, vld_df):

    import albumentations as A
    from albumentations import (
        Rotate,HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
        Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
        IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
        IAASharpen, IAAEmboss, Flip, OneOf, Compose
    )
    from albumentations.pytorch import ToTensor, ToTensorV2

    train_transforms = A.Compose([
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


    valid_transforms = A.Compose([
        ToTensor()
    ])

    from torch.utils.data import Dataset, DataLoader
    trn_dataset = BangaliDataset(imgs=imgs, label_df=trn_df, transform=train_transforms)
    vld_dataset = BangaliDataset(imgs=imgs, label_df=vld_df, transform=valid_transforms)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    trn_sampler = torch.utils.data.distributed.DistributedSampler(
        trn_dataset, 
        num_replicas=world_size, # worldsize만큼 분할
        rank=rank)
    
    trn_loader = DataLoader(trn_dataset, 
                            shuffle=False, 
                            num_workers=8,
                            pin_memory=True,
                            batch_size=BATCH_SIZE,
                           sampler=trn_sampler)  
    
    vld_loader = DataLoader(vld_dataset, shuffle=False, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)  
    return trn_loader, vld_loader


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


def _format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
            
def train_model(args):
    from torchvision import datasets, models
    from tqdm import tqdm
    
    imgs, trn_df, vld_df = _get_images(args.train_dir, args.num_folds, args.vld_fold_idx, data_type='train')
    trn_loader, vld_loader = _get_data_loader(imgs, trn_df, vld_df)

    
    logger.info("=== Getting Pre-trained model ===")    
    model = models.resnet18(pretrained=True)
    last_hidden_units = model.fc.in_features
    model.fc = torch.nn.Linear(last_hidden_units, 186)
#     len_buffer =  len(list(module.buffers()))

#     logger.info("=== Buffer ===")    
#     print(f"len_buffer={len_buffer}")
#     print(list(model.buffers()))
    
    # SDP: Pin each GPU to a single process
    # Use SMDataParallel PyTorch DDP for efficient distributed training
    model = DDP(model.to(args.device), broadcast_buffers=False)

    # SDP: Pin each GPU to a single SDP process.
    torch.cuda.set_device(args.local_rank)
    model.cuda(args.local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                          verbose=True, patience=5, 
                                                          factor=0.5)

    best_score = -1
    training_stats = []
    logger.info("=== Start Training ===")    

    for epoch_id in range(args.num_epochs):

        ################################################################################
        # ==> Training phase
        ################################################################################    
        trn_loss = []
        model.train()

        # Measure how long the training epoch takes.
        t0 = time.time()
        running_loss = 0.0

        for batch_id, (inputs, targets) in enumerate((trn_loader)):
            inputs = inputs.cuda()
            targets = targets.cuda()
            targets_gra = targets[:, 0]
            targets_vow = targets[:, 1]
            targets_con = targets[:, 2]

            # 50%의 확률로 원본 데이터 그대로 사용    
            if np.random.rand() < 0.5:
                logits = model(inputs)
                grapheme = logits[:, :168]
                vowel = logits[:, 168:179]
                cons = logits[:, 179:]

                loss1 = loss_fn(grapheme, targets_gra)
                loss2 = loss_fn(vowel, targets_vow)
                loss3 = loss_fn(cons, targets_con) 

            else:

                lam = np.random.beta(1.0, 1.0) 
                rand_index = torch.randperm(inputs.size()[0])
                shuffled_targets_gra = targets_gra[rand_index]
                shuffled_targets_vow = targets_vow[rand_index]
                shuffled_targets_con = targets_con[rand_index]

                bbx1, bby1, bbx2, bby2 = _rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # 픽셀 비율과 정확히 일치하도록 lambda 파라메터 조정  
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

                logits = model(inputs)
                grapheme = logits[:,:168]
                vowel = logits[:, 168:179]
                cons = logits[:, 179:]

                loss1 = loss_fn(grapheme, targets_gra) * lam + loss_fn(grapheme, shuffled_targets_gra) * (1. - lam)
                loss2 = loss_fn(vowel, targets_vow) * lam + loss_fn(vowel, shuffled_targets_vow) * (1. - lam)
                loss3 = loss_fn(cons, targets_con) * lam + loss_fn(cons, shuffled_targets_con) * (1. - lam)

            loss = 0.5 * loss1 + 0.25 * loss2 + 0.25 * loss3    
            trn_loss.append(loss.item())
            running_loss += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Printing vital information
            if (batch_id + 1) % (args.log_interval) == 0:
                s = f'[Epoch {epoch_id} Batch {batch_id+1}/{len(trn_loader)}] ' \
                f'loss: {running_loss / args.log_interval:.4f}'
                print(s)
                running_loss = 0

        # Measure how long this epoch took.
        trn_time = _format_time(time.time() - t0)        


        if args.rank == 0:    
            ################################################################################
            # ==> Validation phase
            ################################################################################
            val_loss = []
            val_true = []
            val_pred = []
            model.eval()

            # === Validation phase ===
            logger.info('=== Start Validation ===')        

            with torch.no_grad():
                for inputs, targets in vld_loader:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    logits = model(inputs)
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
            s = f'[Epoch {epoch_id}] ' \
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
                torch.save(state_dict, os.path.join(args.model_dir, 'model.pt'))

            # Record all statistics from this epoch
            training_stats.append(
                {
                    'epoch': epoch_id + 1,
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
            logger.info("Model successfully saved at: {}".format(args.model_dir))            

        
def parser_args():
    
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--vld_fold_idx', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=10) 

    # SageMaker Container environment    
    parser.add_argument('--train_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])    
    parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
 
    args = parser.parse_args() 
    return args
        
    
if __name__ =='__main__':

    #parse arguments
    args = parser_args() 
    
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    args.local_rank = dist.get_local_rank()
    #print(f"rank={args.rank}, local_rank={args.local_rank}")
    args.batch_size //= args.world_size // 8
    args.batch_size = max(args.batch_size, 1)
    
    args.use_cuda = args.num_gpus > 0
    print("args.use_cuda : {} , args.num_gpus : {}".format(
        args.use_cuda, args.num_gpus))
    args.device = torch.device("cuda" if args.use_cuda else "cpu")

    train_model(args)
    
