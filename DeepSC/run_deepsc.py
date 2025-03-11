import json
import argparse
import os


def preprocess_img(palette_output_path, save_dir, channel, ori_size, crop_num, resize_size):
    for dir in os.listdir(palette_output_path):
        for file in os.listdir(os.path.join(palette_output_path, dir)):
            if 'Out' in file and f'C0{channel}' in file:
                block_width = original_image.size[0] // grid_size
                block_height = original_image.size[1] // grid_size

                # 划分并放大每个区块
                for i in range(grid_size):
                    for j in range(grid_size):
                        left = j * block_width
                        top = i * block_height
                        right = left + block_width
                        bottom = top + block_height
                        block = original_image.crop((left, top, right, bottom))
                        block = block.resize((width, height), Image.Resampling.LANCZOS)
                        block_filename = f"block_{i}_{j}_{filename}"
                        block.save(os.path.join(output_path, block_filename).replace('.png', '.jpg'))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration')
    args = parser.parse_args()

    with open('config.json') as f:
        cfg = json.load(f)
        
    