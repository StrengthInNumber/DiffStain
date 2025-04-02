import json
import argparse
import os
from PIL import Image
import datetime
from torchvision import transforms

import extract
import extract_utils


def crop_img(palette_output_path, save_dir, save_ori_dir, channel, crop_num, resize_size):
    resize_size = (resize_size, resize_size)
    for dir in os.listdir(palette_output_path):
        for file in os.listdir(os.path.join(palette_output_path, dir)):
            if 'Out' in file and f'C0{channel}' in file:
                img_path = os.path.join(palette_output_path, dir, file) 
                original_image = Image.open(img_path).convert('L')
                original_image.save(os.path.join(save_ori_dir, f"{dir}_C0{channel}.png"))
                block_width = original_image.size[0] // crop_num
                block_height = original_image.size[1] // crop_num

                # 划分并放大每个区块
                for i in range(crop_num):
                    for j in range(crop_num):
                        left = j * block_width
                        top = i * block_height
                        right = left + block_width
                        bottom = top + block_height
                        block = original_image.crop((left, top, right, bottom))
                        block = block.resize(resize_size, Image.Resampling.LANCZOS)
                        block_filename = f"block_{i}_{j}_{dir}_C0{channel}.png"
                        block.save(os.path.join(save_dir, block_filename))


def gen_img_list(img_dir, save_dir):
    with open(os.path.join(save_dir, 'images.txt'), 'w') as f:
        for file in os.listdir(img_dir):
            f.write(file + '\n')
            
def reassemble_seg_img(seg_dir, save_dir, crop_num, ori_size):
    ori_size = (ori_size, ori_size)
    images = {}
    for filename in os.listdir(seg_dir):
        parts = filename.split('_')
        img_name = parts[3:]  # file name: block_i_j_imagename.jpg
        img_name = '_'.join(parts[3:])
        if img_name not in images:
            images[img_name] = []
        images[img_name].append((filename, int(parts[1]), int(parts[2])))

    part_size = (ori_size[0] // crop_num, ori_size[1] // crop_num)
    
    for img_name, blocks in images.items():
        full_image = Image.new('L', ori_size)

        for block_filename, i, j in blocks:
            block_path = os.path.join(seg_dir, block_filename)
            block_image = Image.open(block_path).resize(part_size, Image.Resampling.NEAREST)
            left = j * part_size[1]
            top = i * part_size[0]
            full_image.paste(block_image, (left, top))

        # 保存拼接后的图像
        full_image.save(os.path.join(save_dir, img_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='JSON file for configuration')
    parser.add_argument('--channel', type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    intermediate_result_path = os.path.join(cfg['intermediate_result_path'], 
                                            datetime.datetime.now().strftime('%m-%d_%H-%M') + f'C0{args.channel}')
    if not os.path.exists(intermediate_result_path):
        os.makedirs(intermediate_result_path)
    else:
        raise FileExistsError(f'{intermediate_result_path} already exists!')
    
    ori_img_path = os.path.join(intermediate_result_path, 'ori_images')
    images_path = os.path.join(intermediate_result_path, 'images')
    lists_path = os.path.join(intermediate_result_path, 'lists')
    features_path = os.path.join(intermediate_result_path, 'features')
    eigs_path = os.path.join(intermediate_result_path, 'eigs')
    segs_path = os.path.join(intermediate_result_path, 'segs')
    ori_segs_path = os.path.join(intermediate_result_path, 'ori_segs')
    ori_crfs_path = os.path.join(intermediate_result_path, 'crfs')
    
    for path in [ori_img_path, images_path, lists_path, features_path, 
                 eigs_path, segs_path, ori_segs_path, ori_crfs_path]:
        extract_utils.make_output_dir(path)
        
    crop_img(cfg['dataset']['palette_out_path'], images_path, ori_img_path,
             args.channel, cfg['dataset']['crop_num'],
             cfg['dataset']['resize_size'])
    
    gen_img_list(images_path, lists_path)
    
    print('Cropping Images finished!')
    
    val_transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((cfg['dataset']['gt_img_mean'][args.channel]), 
                                                             (cfg['dataset']['gt_img_std'][args.channel]))])
    extract.extract_features(images_path, os.path.join(lists_path, 'images.txt'), 
                             cfg['model_ckpt_path'], cfg['batch_size'], 
                             val_transform, features_path)
    
    extract.extract_eigs(images_path, features_path, eigs_path, K=cfg['K'],
                         image_color_lambda=cfg['image_color_lambda'])
    
    extract.extract_seg(images_path, features_path, eigs_path, segs_path,
                        num_eigenvectors=cfg['K'], 
                        n_clusters=cfg['n_clusters'])
    
    reassemble_seg_img(segs_path, ori_segs_path, cfg['dataset']['crop_num'],
                       cfg['dataset']['input_size'])
    
    extract.extract_crf(ori_img_path, ori_segs_path, ori_crfs_path,
                        crf_params=cfg['crf_params'])
    