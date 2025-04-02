from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cv2
import fire
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

import extract_utils as utils


def extract_features(
    images_root: str,
    images_list: str,
    model_path: str,
    batch_size: int,
    val_transform,
    output_dir: str,
    which_block: int = -1,
):
    """
    Extract features from a list of images.

    Example:
        python extract.py extract_features \
            --images_list "./data/VOC2012/lists/images.txt" \
            --images_root "./data/VOC2012/images" \
            --output_dir "./data/VOC2012/features/dino_vits16" \
            --batch_size 1
    """

    # Models
    model, _, patch_size, num_heads = utils.get_model(model_path)

    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    model._modules["blocks"][which_block]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

    # Dataset
    filenames = Path(images_list).read_text().splitlines()
    dataset = utils.ImagesDataset(filenames=filenames, images_root=images_root, transform=val_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8)
    print(f'Dataset size: {len(dataset)=}')
    print(f'Dataloader size: {len(dataloader)=}')

    # Prepare
    # accelerator = Accelerator(fp16=True, cpu=False) 修改为
    accelerator = Accelerator(mixed_precision='fp16', cpu=False)
    # model, dataloader = accelerator.prepare(model, dataloader)
    model = model.to(accelerator.device)

    # Process
    pbar = tqdm(dataloader, desc='Processing')
    for i, (images, files, indices) in enumerate(pbar):
        output_dict = {}

        # Check if file already exists
        id = Path(files[0]).stem
        output_file = Path(output_dir) / f'{id}.pth'
        if output_file.is_file():
            pbar.write(f'Skipping existing file {str(output_file)}')
            continue

        # Reshape image
        P = patch_size
        B, C, H, W = images.shape
        H_patch, W_patch = H // P, W // P
        H_pad, W_pad = H_patch * P, W_patch * P
        T = H_patch * W_patch + 1  # number of tokens, add 1 for [CLS]
        # images = F.interpolate(images, size=(H_pad, W_pad), mode='bilinear')  # resize image
        images = images[:, :, :H_pad, :W_pad]
        images = images.to(accelerator.device)

        model.get_intermediate_layers(images)[0].squeeze(0)
        output_qkv = feat_out["qkv"].reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
        output_dict['k'] = output_qkv[1].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]

        # Metadata
        output_dict['indices'] = indices[0]
        output_dict['file'] = files[0]
        output_dict['id'] = id
        output_dict['patch_size'] = patch_size
        output_dict['shape'] = (B, C, H, W)
        output_dict = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in output_dict.items()}

        # Save
        accelerator.save(output_dict, str(output_file))
        accelerator.wait_for_everyone()
    
    print(f'Saved features to {output_dir}')


def _extract_eig(
    inp: Tuple[int, str], 
    K: int, 
    images_root: str,
    output_dir: str,
    which_features: str = 'k',
    normalize: bool = True,
    lapnorm: bool = True,
    threshold_at_zero: bool = True,
    image_downsample_factor: Optional[int] = None,
    image_color_lambda: float = 10,
):
    index, features_file = inp

    # Load 
    data_dict = torch.load(features_file, map_location='cpu')
    image_id = data_dict['file'][:-4]
    
    # Load
    output_file = str(Path(output_dir) / f'{image_id}.pth')
    if Path(output_file).is_file():
        print(f'Skipping existing file {str(output_file)}')
        return  # skip because already generated

    # Load affinity matrix
    feats = data_dict[which_features].squeeze().cuda()
    if normalize:
        feats = F.normalize(feats, p=2, dim=-1)

    # Eigenvectors of affinity matrix

    # Eigenvectors of matting laplacian matrix
    # Get sizes
    B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)
    if image_downsample_factor is None:
        image_downsample_factor = P
    H_pad_lr, W_pad_lr = H_pad // image_downsample_factor, W_pad // image_downsample_factor

    # Upscale features to match the resolution
    if (H_patch, W_patch) != (H_pad_lr, W_pad_lr):
        feats = F.interpolate(
            feats.T.reshape(1, -1, H_patch, W_patch), 
            size=(H_pad_lr, W_pad_lr), mode='bilinear', align_corners=False
        ).reshape(-1, H_pad_lr * W_pad_lr).T

    ### Feature affinities 
    W_feat = (feats @ feats.T)
    if threshold_at_zero:
        W_feat = (W_feat * (W_feat > 0))
    W_feat = W_feat / W_feat.max()  # NOTE: If features are normalized, this naturally does nothing
    W_feat = W_feat.cpu().numpy()
    ### Color affinities 
    # If we are fusing with color affinites, then load the image and compute
    if image_color_lambda > 0:

        # Load image
        image_file = str(Path(images_root) / f'{image_id}.png')
        image_lr = Image.open(image_file).resize((W_pad_lr, H_pad_lr), Image.BILINEAR)
        image_lr = np.array(image_lr) / 255.

        # Color affinities (of type scipy.sparse.csr_matrix)
        W_lr = utils.knn_affinity(image_lr)
            
        # Convert to dense numpy array
        W_color = np.array(W_lr.todense().astype(np.float32))
        
    else:
        # No color affinity
        W_color = 0

    # Combine
    W_comb = W_feat + W_color * image_color_lambda  # combination
    D_comb = np.array(utils.get_diagonal(W_comb).todense())  # is dense or sparse faster? not sure, should check

    # Extract eigenvectors eigsh函数：提取k个特征值和特征向量
    if lapnorm:
        try:
            eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM', M=D_comb)
        except:
            eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM', M=D_comb)
    else:
        try:
            eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM')
        except:
            eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM')
    eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()

    # Sign ambiguity
    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]

    # Save dict
    output_dict = {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors}
    torch.save(output_dict, output_file)


def extract_eigs(
    images_root: str,
    features_dir: str,
    output_dir: str,
    which_features: str = 'k',
    normalize: bool = True,
    threshold_at_zero: bool = True,
    lapnorm: bool = True,
    K: int = 20,
    image_downsample_factor: Optional[int] = None,
    image_color_lambda: float = 0.0,
    multiprocessing: int = 0
):
    """
    Extracts eigenvalues from features.
    
    Example:
        python extract.py extract_eigs \
            --images_root "./data/VOC2012/images" \
            --features_dir "./data/VOC2012/features/dino_vits16" \
            --which_matrix "laplacian" \
            --output_dir "./data/VOC2012/eigs/laplacian" \
            --K 5
    """
    utils.make_output_dir(output_dir)
    kwargs = dict(K=K,
                 which_features=which_features,
                 normalize=normalize,
                 threshold_at_zero=threshold_at_zero,
                 images_root=images_root,
                 output_dir=output_dir,
                 image_downsample_factor=image_downsample_factor,
                 image_color_lambda=image_color_lambda,
                 lapnorm=lapnorm)
    print(kwargs)
    fn = partial(_extract_eig, **kwargs)
    inputs = list(enumerate(sorted(Path(features_dir).iterdir())))
    utils.parallel_process(inputs, fn, multiprocessing)


def get_des_idx(segmap, ori_img, n_clusters, ):
    ori_img = np.array(ori_img)
    segmap_np = np.array(segmap)
    ave = np.array([])
    for k in range(n_clusters):
        binary_image = np.where(segmap_np == k, 1, 0).astype(np.uint8)
        res = np.sum(binary_image*ori_img) / np.sum(binary_image)
        if np.sum(binary_image) >= 100:
            ave = np.append(ave, res)
        else:
            ave = np.append(ave, 0)
        # ave = np.append(ave, res)
    return np.argmax(ave)


def use_kmeans(n_clusters, data_dict, num_eigenvectors, 
               H_patch, W_patch):
    
    kmeans = KMeans(n_clusters=n_clusters)
    eigenvectors = data_dict['eigenvectors'][1:1+num_eigenvectors].numpy()  # take non-constant eigenvectors
    clusters = kmeans.fit_predict(eigenvectors.T)

    # Reshape
    if clusters.size == H_patch * W_patch:
        segmap = clusters.reshape(H_patch, W_patch)
    elif clusters.size == H_patch * W_patch * 4:
        segmap = clusters.reshape(H_patch * 2, W_patch * 2)
    else:
        raise ValueError()
    return segmap
    

def _extract_seg(
    inp: Tuple[int, Tuple[str, str]],
    n_clusters: int,
    output_dir: str,
    num_eigenvectors: int,
    input_path: str
):
    index, (feature_path, eigs_path) = inp

    # Load 
    data_dict = torch.load(feature_path, map_location='cpu')
    data_dict.update(torch.load(eigs_path, map_location='cpu'))

    # Output file
    id = Path(data_dict['id'])
    output_file = str(Path(output_dir) / f'{id}.png')
    if Path(output_file).is_file():
        print(f'Skipping existing file {str(output_file)}')
        return  # skip because already generated

    # Sizes
    B, C, H, W, P, H_patch, W_patch, H_pad, W_pad = utils.get_image_sizes(data_dict)

    print("*************n_clusters:")
    print(n_clusters)
    segmap = use_kmeans(n_clusters, data_dict, num_eigenvectors, H_patch, W_patch)
    
    ori_img_path = str(Path(input_path) / f'{id}.png')
    ori_img = Image.open(ori_img_path).convert('L')
    segmap_to_depend = Image.fromarray(segmap).resize(ori_img.size, Image.NEAREST)
    
    cells_idx = get_des_idx(segmap_to_depend, ori_img, n_clusters)
    
    binary_map = np.where(segmap==cells_idx, 255, 0).astype(np.uint8)
    
    segmap = np.where(segmap==cells_idx, 3, segmap)
    
    # Save dict
    Image.fromarray(binary_map).save(output_file)


def extract_seg(
    images_dir: str,
    features_dir: str,
    eigs_dir: str,
    output_dir: str,
    n_clusters: int = 2,
    num_eigenvectors: int = 1_000_000,
    multiprocessing: int = 0
):
    """
    Example:
    python extract.py extract_multi_region_segmentations \
        --features_dir "./data/VOC2012/features/dino_vits16" \
        --eigs_dir "./data/VOC2012/eigs/laplacian" \
        --output_dir "./data/VOC2012/multi_region_segmentation/fixed" \
    """
    utils.make_output_dir(output_dir)
    fn = partial(_extract_seg,
                 n_clusters=n_clusters,
                 num_eigenvectors=num_eigenvectors,
                 output_dir=output_dir,
                 input_path=images_dir)
    inputs = utils.get_paired_input_files(features_dir, eigs_dir)
    utils.parallel_process(inputs, fn, multiprocessing)




def _extract_crf_segmentations(
    inp: Tuple[int, Tuple[str, str]], 
    images_dir: str,
    num_classes: int,
    output_dir: str,
    crf_params: Tuple,
    downsample_factor: int = 1,
):
    index, (image_file, segmap_path) = inp

    # Output file
    id = Path(image_file).stem
    output_file = str(Path(output_dir) / f'{id}.png')
    if Path(output_file).is_file():
        print(f'Skipping existing file {str(output_file)}')
        return  # skip because already generated

    # Load image and segmap
    image_file = str(Path(images_dir) / f'{id}.png')
    image = np.array(Image.open(image_file).convert('RGB'))  # (H_patch, W_patch, 3)
    segmap = np.array(Image.open(segmap_path))  # (H_patch, W_patch)
     
    # Sizes
    P = downsample_factor
    H, W = image.shape[:2]
    H_patch, W_patch = H // P, W // P
    H_pad, W_pad = H_patch * P, W_patch * P

    # Resize and expand
    segmap_upscaled = cv2.resize(segmap, dsize=(W_pad, H_pad), interpolation=cv2.INTER_NEAREST)  # (H_pad, W_pad)
    segmap_orig_res = cv2.resize(segmap, dsize=(W, H), interpolation=cv2.INTER_NEAREST)  # (H, W)
    segmap_orig_res[:H_pad, :W_pad] = segmap_upscaled  # replace with the correctly upscaled version, just in case they are different

    # Convert binary
    if set(np.unique(segmap_orig_res).tolist()) == {0, 255}:
        segmap_orig_res[segmap_orig_res == 255] = 1

    print(segmap_orig_res.shape, image.shape)
    # CRF
    import denseCRF  # make sure you've installed SimpleCRF
    unary_potentials = F.one_hot(torch.from_numpy(segmap_orig_res).long(), num_classes=num_classes)
    segmap_crf = denseCRF.densecrf(image, unary_potentials, crf_params)  # (H_pad, W_pad)

    # Save
    segmap_crf = np.where(segmap_crf == 1, 255, 0).astype(np.uint8)
    Image.fromarray(segmap_crf).convert('L').save(output_file)


def extract_crf(
    images_dir: str,
    segmentations_dir: str,
    output_dir: str,
    num_classes: int = 2,
    downsample_factor: int = 16,
    multiprocessing: int = 0,
    # CRF parameters
    crf_params: dict = dict(w1=10, alpha=80, beta=13, w2=3, gamma=3, it=5.0),
):
    """
    w1    = 10,    # weight of bilateral term  # default: 10.0,
    alpha = 80,    # spatial std  # default: 80,  
    beta  = 13,    # rgb  std  # default: 13,  
    w2    = 3,     # weight of spatial term  # default: 3.0, 
    gamma = 3,     # spatial std  # default: 3,   
    it    = 5.0,   # iteration  # default: 5.0, 
    Applies a CRF to segmentations in order to sharpen them.

    Example:
        python extract.py extract_crf_segmentations \
            --images_list "./data/VOC2012/lists/images.txt" \
            --images_root "./data/VOC2012/images" \
            --segmentations_dir "./data/VOC2012/semantic_segmentations/patches/fixed/segmaps_e2_d5_pca_32" \
            --output_dir "./data/VOC2012/semantic_segmentations/crf/fixed/segmaps_e2_d5_pca_32" \
    """
    try:
        import denseCRF
    except:
        raise ImportError(
            'Please install SimpleCRF to compute CRF segmentations:\n'
            'pip3 install SimpleCRF'
        )

    utils.make_output_dir(output_dir)
    crf_params = (crf_params['w1'], crf_params['alpha'], crf_params['beta'], crf_params['w2'], crf_params['gamma'], crf_params['it'])
    fn = partial(_extract_crf_segmentations, images_dir=images_dir, num_classes=num_classes, output_dir=output_dir,
                 crf_params=crf_params, downsample_factor=downsample_factor)
    inputs = utils.get_paired_input_files(images_dir, segmentations_dir)
    print(f'Found {len(inputs)} images and segmaps')
    utils.parallel_process(inputs, fn, multiprocessing)


def vis_segmentations(
    images_list: str,
    images_root: str,
    segmentations_dir: str,
    bbox_file: Optional[str] = None,
):
    """
    Example:
        streamlit run extract.py vis_segmentations -- \
            --images_list "./data/VOC2012/lists/images.txt" \
            --images_root "./data/VOC2012/images" \
            --segmentations_dir "./data/VOC2012/multi_region_segmentation/fixed"
    or alternatively:
            --segmentations_dir "./data/VOC2012/semantic_segmentations/crf/fixed/segmaps_e2_d5_pca_32/"
    """
    # Streamlit setup
    import streamlit as st
    from matplotlib.cm import get_cmap
    from skimage.color import label2rgb
    st.set_page_config(layout='wide')

    # Inputs
    image_paths = []
    segmap_paths = []
    images_root = Path(images_root)
    segmentations_dir = Path(segmentations_dir)
    for image_file in Path(images_list).read_text().splitlines():
        segmap_file = f'{Path(image_file).stem}.png'
        image_paths.append(images_root / image_file)
        segmap_paths.append(segmentations_dir / segmap_file)
    print(f'Found {len(image_paths)} image and segmap paths')

    # Load optional bounding boxes
    if bbox_file is not None:
        bboxes_list = torch.load(bbox_file)

    # Colors
    colors = get_cmap('tab20', 21).colors[:, :3]

    # Which index
    which_index = st.number_input(label='Which index to view (0 for all)', value=0)

    # Load
    total = 0
    for i, (image_path, segmap_path) in enumerate(zip(image_paths, segmap_paths)):
        if total > 40: break
        image_id = image_path.stem
        
        # Streamlit
        cols = []
        
        # Load
        image = np.array(Image.open(image_path).convert('RGB'))
        segmap = np.array(Image.open(segmap_path))

        # Convert binary
        if set(np.unique(segmap).tolist()) == {0, 255}:
            segmap[segmap == 255] = 1

        # Resize
        segmap_fullres = cv2.resize(segmap, dsize=image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        # Only view images with a specific class
        if which_index not in np.unique(segmap):
            continue
        total += 1

        # Streamlit
        cols.append({'image': image, 'caption': image_id})

        # Load optional bounding boxes
        bboxes = None
        if bbox_file is not None:
            bboxes = torch.tensor(bboxes_list[i]['bboxes_original_resolution'])
            assert bboxes_list[i]['id'] == image_id, f"{bboxes_list[i]['id']=} but {image_id=}"
            image_torch = torch.from_numpy(image).permute(2, 0, 1)
            image_with_boxes_torch = draw_bounding_boxes(image_torch, bboxes)
            image_with_boxes = image_with_boxes_torch.permute(1, 2, 0).numpy()
            
            # Streamlit
            cols.append({'image': image_with_boxes})
            
        # Color
        segmap_label_indices, segmap_label_counts = np.unique(segmap, return_counts=True)
        blank_segmap_overlay = label2rgb(label=segmap_fullres, image=np.full_like(image, 128), 
            colors=colors[segmap_label_indices[segmap_label_indices != 0]], bg_label=0, alpha=1.0)
        image_segmap_overlay = label2rgb(label=segmap_fullres, image=image, 
            colors=colors[segmap_label_indices[segmap_label_indices != 0]], bg_label=0, alpha=0.45)
        segmap_caption = dict(zip(segmap_label_indices.tolist(), (segmap_label_counts).tolist()))

        # Streamlit
        cols.append({'image': blank_segmap_overlay, 'caption': segmap_caption})
        cols.append({'image': image_segmap_overlay, 'caption': segmap_caption})

        # Display
        for d, col in zip(cols, st.columns(len(cols))):
            col.image(**d)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    fire.Fire(dict(
        extract_features=extract_features,
        extract_eigs=extract_eigs,
        extract_multi_region_segmentations=extract_multi_region_segmentations,
        extract_crf_segmentations=extract_crf_segmentations,
        extract_single_region_segmentations=extract_single_region_segmentations,
        vis_segmentations=vis_segmentations,
    ))
