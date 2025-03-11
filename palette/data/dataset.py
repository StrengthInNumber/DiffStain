import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import albumentations
import pandas as pd


class CellPaintingDataset(data.Dataset):
    def __init__(self, dataset_path, to_size, clip_percent, phase, binary_cond_path,
                 gt_img_mean, gt_img_std, cond_img_mean, cond_img_std):
        self.dataset_path = dataset_path
        self.dataset_list = [x for x in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, x))]
        self.to_size = (to_size, to_size)
        self.is_train = phase == 'train'
        self.clip_percent = clip_percent
        
        self.binary_cond_path = binary_cond_path
        if binary_cond_path is not None:
            self.binary_cond_list = os.listdir(binary_cond_path)
        
        self.gt_img_mean = np.array(gt_img_mean)
        self.gt_img_std = np.array(gt_img_std)
        self.cond_img_mean = np.array(cond_img_mean)
        self.cond_img_std = np.array(cond_img_std)
        self.clip_percent = clip_percent
        
        if self.is_train:
            self.aug = albumentations.Compose([
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.augmentations.geometric.rotate.RandomRotate90(p=0.5),
            ])
        else:
            self.aug = albumentations.Compose([])
    
    def standardize_image(self, image_in_path, channel, is_binary=False):
        image_in = Image.open(image_in_path).resize(self.to_size)
        
        image_in = np.array(image_in)
        image_in = image_in.astype('float64')
            
        if not is_binary:
            lower_bound = np.percentile(image_in, self.clip_percent)
            upper_bound = np.percentile(image_in, 100-self.clip_percent)
            image_in = np.clip(image_in, lower_bound, upper_bound)
            image_in = (image_in - lower_bound) / (upper_bound - lower_bound)
            if channel < 5:
                image_in = (image_in - self.gt_img_mean[channel]) / self.gt_img_std[channel]
            else:
                image_in = (image_in - self.cond_img_mean[channel-5]) / self.cond_img_std[channel-5]
        
        image_in = self.aug(image=image_in)['image']
        image_in = image_in.astype(np.float32)
        image_in = np.expand_dims(image_in, 0)
        return image_in     
        
    def __len__(self):
        return (len(self.dataset_list))

    def __getitem__(self, idx):
        
        ret = {}
        img_dir = self.dataset_list[idx]
        img_chs_path = [os.listdir(os.path.join(self.dataset_path, img_dir))[i] for i in range(8)]
        img_chs = [self.standardize_image(os.path.join(
            self.dataset_path, img_dir, img_chs_path[i]), i) for i in range(8)]
             
        img_gt = np.concatenate((img_chs[0], img_chs[1], img_chs[2], img_chs[3], img_chs[4]), axis= 0)
        img_cond = np.concatenate((img_chs[5], img_chs[6], img_chs[7]), axis=0)
                
        img_gt = torch.tensor(img_gt, dtype=torch.float)
        img_cond = torch.tensor(img_cond, dtype=torch.float) 

        ret['gt_image'] = img_gt  # (output) Cell Painting 5x
        ret['cond_image'] = img_cond  # (input) Brightfield 3x
        
        if self.binary_cond_path is not None:
            binary_dir = self.binary_cond_list[idx]
            binary_chs_path = [os.listdir(os.path.join(self.binary_cond_path, binary_dir))[i] for i in range(5)]
            binary_chs = [self.standardize_image(os.path.join(
                self.binary_cond_path, binary_dir, binary_chs_path[i]), i, is_binary=True) for i in range(5)]
            img_binary = np.concatenate((binary_chs[0], binary_chs[1], binary_chs[2], binary_chs[3], binary_chs[4]), axis=0)
            img_binary = torch.tensor(img_binary, dtype=torch.float)
            ret['ref_image'] = img_binary

        ret['name'] = self.dataset_list[idx]
        
        return ret
