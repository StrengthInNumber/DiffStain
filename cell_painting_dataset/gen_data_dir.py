from PIL import Image
import re
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--download_dir', type=str)
parser.add_argument('--des_dir', type=str)
args = parser.parse_args()

old_data_dir = args.download_dir
data_dir = args.des_dir
if not os.path.exists(args.des_dir):
    os.mkdir(args.des_dir)

plate_name_dict = {}
plate_idx = 1
for file in tqdm(os.listdir(old_data_dir)):
    if file.split('.')[1] != 'tiff':
        continue
    plate_name = file.split('-')[0]
    channel_name = file.split('-')[1].split('.')[0]
    if not os.path.exists(os.path.join(data_dir, plate_name)):
        os.makedirs(os.path.join(data_dir, plate_name))
        plate_name_dict[plate_name] = plate_idx
        plate_idx += 1
        
    tiff_img = Image.open(os.path.join(old_data_dir, file)).resize((512, 512))
    tiff_img_arr = np.array(tiff_img)
    tiff_img_arr = (tiff_img_arr - tiff_img_arr.min()) / (tiff_img_arr.max() - tiff_img_arr.min()) * 255
    tiff_img = Image.fromarray(tiff_img_arr.astype(np.uint8))
    
    reg = r'ch[0-9]'
    channel_num = re.findall(reg, channel_name)[0][-1]
    ch_name = 'C0' + channel_num
    if ch_name == 'C06':
        ch_name = 'C06_1'
    elif ch_name == 'C07':
        ch_name = 'C06_2'
    elif ch_name == 'C08':
        ch_name = 'C06_3'
    tiff_img.save(os.path.join(data_dir, plate_name, ch_name + '.png'))