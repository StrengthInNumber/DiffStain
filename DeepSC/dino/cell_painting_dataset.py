from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np


class CellPaintingDataset(Dataset):
    def __init__(self, data_dir, channel, crop_num, input_size, transform, clip_percent):
        self.channel = channel
        self.data_dir = data_dir
        self.data_dir_list = [i for i in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, i))]
        self.crop_num = crop_num
        self.input_size = (input_size, input_size)
        self.transform = transform
        self.clip_percent = clip_percent
    
    def __len__(self):
        return len(self.data_dir_list) * (self.crop_num * self.crop_num)

    def __getitem__(self, idx):
        dir_idx = idx // (self.crop_num * self.crop_num)
        crop_idx = idx % (self.crop_num * self.crop_num)
        crop_x = crop_idx // self.crop_num
        crop_y = crop_idx % self.crop_num
        whole_img = Image.open(os.path.join(self.data_dir, self.data_dir_list[dir_idx], f'C0{self.channel}.png'))
        whole_img = np.array(whole_img).astype('float64')
        
        lower_bound = np.percentile(whole_img, self.clip_percent)
        upper_bound = np.percentile(whole_img, 100-self.clip_percent)
        whole_img = np.clip(whole_img, lower_bound, upper_bound)
        
        whole_img = (whole_img - lower_bound) / (upper_bound - lower_bound) * 255
        whole_img = Image.fromarray(whole_img.astype('uint8'))
        
        crop_size = (whole_img.size[0] // self.crop_num, whole_img.size[1] // self.crop_num)
        crop_img = whole_img.crop((crop_x * crop_size[0], crop_y * crop_size[1], (crop_x + 1) * crop_size[0], (crop_y + 1) * crop_size[1]))
        crop_img = crop_img.resize(self.input_size).convert('RGB')
        # crop_img.save(f'./test/{self.data_dir_list[dir_idx]}_{crop_x}_{crop_y}.png')
        
        # crop_img = np.array(crop_img).astype('float64')
        # crop_img = (crop_img - crop_img.min()) / (crop_img.max() - crop_img.min())
        # crop_img = torch.tensor(crop_img).permute(2, 0, 1)
        # print(crop_img.size)
        crop_img = self.transform(crop_img)
        return crop_img, "label"