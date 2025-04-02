# DiffStain

This repository contains the official implementation of the research paper: 

**"DiffStain: Conditioned Diffusion-Based Semantic Virtual Staining with Mask Guidance"**

![overview](./resources/overview.png)

## Requirements and Installation
### System Requirements
+ Python version: 3.8 or higher
+ PyTorch version: 1.8.1 or higher

### Installation
+ To install the required Python packages, execute the following command: `pip install -r requirements.txt`

## Data Preparation
We evaluated our approach using the publicly available JUMP Cell Painting dataset (cpg0000) from the Cell Painting Gallery on the Registry of Open Data on AWS. We use ten plates with each representing different biological phenotypes. Each plate contains 2,000 images with five fluorescent image channels, including nucleus (DNA), endoplasmic reticulum (ER), cytoplasmic RNA (RNA), Actin, Golgi, plasmamembrane (AGP), and mitochondria (Mito), as well as three-channel light field images. For more infomation about this dataset, please view [Cell Painting Gallery Document](https://broadinstitute.github.io/cellpainting-gallery/overview.html).
### Download
To download one plate of images, execute the following commands.
```shell
cd cell_painting_dataset
sh download.sh
```
### Organize
```shell
cd cell_painting_dataset
python gen_data_dir.py --download_dir ./download --des_dir ./dataset
```

## Training
### Palette
The backbone of this implementation is based on [Janspiry/Palette-Image-to-Image-Diffusion-Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models) and is trained in the same way. We provide our .json file "cell_painting.json" which contains the hyperparameters used to train our models. This will require editing to suit the requirements of your file structures and datasets. You can then initiate training with the following commands:
```shell
cd palette
python run.py -p train -c config/diffstain.json
```
### Finetuning Dino
The dino model needs to be finetuned on the cell painting dataset and the checkpoint pretrained on ImageNet is available [here](https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth) (please put it into `./NSC/dino/pretrained_ckpt` or you can adjust the path in the config file). You can start to finetune the dino model with the following commands:
```shell
cd DeepSC/dino
python finetune_dino.py --channel 1 --config_path './vits8Args.json'
```
In order to achieve a better effect of unsupervised deep neural spectral clustering, it is recommended to finetune 5 models on 5 channel images respectively to better learn the features of the 5 channels of fluorescence staining images.

## Testing
1. Generating preliminary virtual fluorescent staining images with Palette.
```shell
cd palette
python run_palette.py -p test -c config/diffstain.json
```
2. Using NSC to generate masks.
```shell
cd ../DeepSC
python run_deepsc -c config_nsc.json
```
3. Generating semantic virtual staining images with mask guidance. 
```shell
cd ../palette
python run_palette.py -p test -c config/diffstain.json
```

## Acknowledge
our work is built upon the following projects, and uses a large amount of their code.
- [Janspiry/Palette-Image-to-Image-Diffusion-Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
- [facebookresearch/dino](https://github.com/facebookresearch/dino)
- [lukemelas/deep-spectral-segmentation](https://github.com/lukemelas/deep-spectral-segmentation)