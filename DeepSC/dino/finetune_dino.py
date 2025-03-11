import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import vit_utils as utils
import vision_transformer as vits
from vision_transformer import DINOHead
import json

from data_augmentation_dino import DataAugmentationDINO
from cell_painting_dataset import CellPaintingDataset

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('--channel', type=int, required=True)
parser.add_argument('--config_path', type=str, required=True)
args_input = parser.parse_args()


class Arg:
    def __init__(self):
        pass


args = Arg()
if args_input.config_path:
    with open(args_input.config_path, 'r') as file:
        config = json.load(file)
        for key, value in config.items():
            setattr(args, key, value)
        
args.output_dir = os.path.join(args.output_dir, "C0" + str(args_input.channel))
         
utils.init_distributed_mode(args)
utils.fix_random_seeds(args.seed)
print("git:\n  {}\n".format(utils.get_sha()))
print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
cudnn.benchmark = True

    # ============ preparing data ... ============
transform = DataAugmentationDINO(
    args.global_crops_scale,
    args.local_crops_scale,
    args.local_crops_number,
    args.input_size,
    args_input.channel,
    args.gt_img_mean,
    args.gt_img_std,
)
dataset = CellPaintingDataset(
    args.data_path,
    args_input.channel,
    args.crop_num,
    args.input_size,
    transform,
    args.clip_percent
)
sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
data_loader = torch.utils.data.DataLoader(
    dataset,
    sampler=sampler,
    batch_size=args.batch_size_per_gpu,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True,
    )
print(f"Data loaded: there are {len(dataset)} images.")


# ============ building student and teacher networks ... ============
# we changed the name DeiT-S for ViT-S to avoid confusions
args.arch = args.arch.replace("deit", "vit")
# if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
if args.arch in vits.__dict__.keys():
    student = vits.__dict__[args.arch](
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path_rate,  # stochastic depth
    )
    teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
    embed_dim = student.embed_dim
    # if the network is a XCiT
elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
    student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                            pretrained=False, drop_path_rate=args.drop_path_rate)
    teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
    embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
elif args.arch in torchvision_models.__dict__.keys():
    student = torchvision_models.__dict__[args.arch]()
    teacher = torchvision_models.__dict__[args.arch]()
    embed_dim = student.fc.weight.shape[1]
else:
    print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
student = utils.MultiCropWrapper(student, DINOHead(
    embed_dim,
    args.out_dim,
    use_bn=args.use_bn_in_head,
    norm_last_layer=args.norm_last_layer,
))
teacher = utils.MultiCropWrapper(
    teacher,
    DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
)
    # move networks to gpu
student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
if utils.has_batchnorms(student):
    student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
    teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
    teacher_without_ddp = teacher.module
else:
    # teacher_without_ddp and teacher are the same thing
    teacher_without_ddp = teacher
student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
for p in teacher.parameters():
    p.requires_grad = False
print(f"Student and Teacher are built: they are both {args.arch} network.")

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

# ============ preparing loss ... ============
dino_loss = DINOLoss(
    args.out_dim,
    # total number of crops = 2 global crops + local_crops_number
    args.local_crops_number + 2,
    args.warmup_teacher_temp,
    args.teacher_temp,
    args.warmup_teacher_temp_epochs,
    args.epochs,
).cuda()

# ============ preparing optimizer ... ============
params_groups = utils.get_params_groups(student)
if args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(
        params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
elif args.optimizer == "lars":
    # to use with convnet and large batches
    optimizer = utils.LARS(params_groups)
# for mixed precision training
fp16_scaler = None
if args.use_fp16:
    fp16_scaler = torch.cuda.amp.GradScaler()

# ============ init schedulers ... ============
lr_schedule = utils.cosine_scheduler(
    args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
    args.min_lr,
    args.epochs, len(data_loader),
    warmup_epochs=args.warmup_epochs,
)
wd_schedule = utils.cosine_scheduler(
    args.weight_decay,
    args.weight_decay_end,
    args.epochs, len(data_loader),
)
    # momentum parameter is increased to 1. during training with a cosine schedule
momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                        args.epochs, len(data_loader))
print(f"Loss, optimizer and schedulers ready.")

# ============ optionally resume training ... ============
to_restore = {"epoch": 0}
utils.restart_from_checkpoint(
    args.pretrained_ckpt_path,  # the path to the checkpoint
    run_variables=to_restore,
    student=student,
    teacher=teacher,
    optimizer=optimizer,
    fp16_scaler=fp16_scaler,
    dino_loss=dino_loss,
)

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

start_epoch = to_restore["epoch"]

start_time = time.time()
torch.cuda.empty_cache()
print("Starting DINO training !")
for epoch in range(start_epoch, args.epochs):
    data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
    train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
        data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
        epoch, fp16_scaler, args)

    # ============ writing logs ... ============
    save_dict = {
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'args': args,
        'dino_loss': dino_loss.state_dict(),
    }
    if fp16_scaler is not None:
        save_dict['fp16_scaler'] = fp16_scaler.state_dict()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
    if args.saveckp_freq and epoch % args.saveckp_freq == 0:
       utils.save_ckpt_for_dsm(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}_for_dsm.pth'))
    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                'epoch': epoch}
    if utils.is_main_process():
        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
    torch.cuda.empty_cache()
total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))

