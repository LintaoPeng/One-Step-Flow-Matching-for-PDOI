# ==============================
# 鍩虹绯荤粺涓庡弬鏁扮浉鍏冲簱
# ==============================
import os
import sys
import argparse

# ==============================
# 妯″瀷涓庤缁冨櫒
# ==============================
from src.UnetRes_Meanflow import UnetRes
from src.meanflow import *
from src.trainer_loop import *

# ==============================
# PyTorch
# ==============================
import torch
import torch.nn as nn

# -*- coding: utf-8 -*-
# =============================================================================
# Single Dataset MeanFlow Training Script
# =============================================================================

import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image

from accelerate import Accelerator
from data.paired_image_dataset import PairedImageDataset


# -----------------------------------------------------------------------------
# 鎸囧畾浣跨敤鐨?GPU
# -----------------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


# =============================================================================
# 鍙傛暟瑙ｆ瀽
# =============================================================================
def parsr_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataroot",
        type=str,
        default="./data/training",
        help="dataset root"
    )

    # 褰撳墠闃舵
    parser.add_argument(
        "--phase",
        type=str,
        default="train",
        choices=["train", "test"]
    )

    # 鏈€澶у姞杞芥暟鎹噺
    parser.add_argument(
        "--max_dataset_size",
        type=int,
        default=float("inf")
    )

    # resize
    parser.add_argument(
        "--load_size",
        type=int,
        default=64
    )

    # crop
    parser.add_argument(
        "--crop_size",
        type=int,
        default=64
    )

    # 鍥惧儚鏂瑰悜
    parser.add_argument(
        "--direction",
        type=str,
        default="AtoB"
    )

    parser.add_argument(
        "--preprocess",
        type=str,
        default="crop"
    )

    # 鏄惁鍏抽棴 flip
    parser.add_argument(
        "--no_flip",
        action="store_true"
    )

    # Dataset 鍐呴儴 batch size锛堜竴鑸笉鐢級
    parser.add_argument(
        "--bsize",
        type=int,
        default=2
    )

    # ------------------------------------------------------------------
    # 鏂偣缁鍙傛暟
    # 鐢ㄦ硶绀轰緥锛?    #   python train_meanflow.py --resume latest      # 鑷姩鎵炬渶鏂?ckpt
    #   python train_meanflow.py --resume 30          # 鍔犺浇 model-30.pt
    #   python train_meanflow.py --resume 999999      # 鍔犺浇蹇収 ckpt
    # 涓嶄紶鍒欎粠澶村紑濮嬭缁冦€?    # ------------------------------------------------------------------
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Resume training from a checkpoint. "
            "Pass an integer milestone (e.g. 30) or 'latest' to auto-detect "
            "the most recent checkpoint in results_folder. "
            "Omit to train from scratch."
        )
    )

    return parser.parse_args()


# =============================================================================
# 鍩虹璁剧疆
# =============================================================================
sys.stdout.flush()

set_seed(10)


# =============================================================================
# 璁粌 & 閲囨牱瓒呭弬鏁?# =============================================================================
save_and_sample_every = 1000

# 閲囨牱姝ユ暟锛堢涓€涓綅缃弬鏁帮紝涓?--resume 涓嶅啿绐侊級
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    sampling_timesteps = int(sys.argv[1])
else:
    sampling_timesteps = 10

train_batch_size = 32
num_samples = 1
sum_scale = 0.01
image_size = 64
condition = True
data_channels = 9
target_channels = 1

# =============================================================================
# 瑙ｆ瀽鍙傛暟
# =============================================================================
opt = parsr_args()
opt.phase = "train"

results_folder = "./ckpt_single_dataset"

# ------------------------------------------------------------------
# 瑙ｆ瀽 resume milestone锛堝皢瀛楃涓?'30' 杞负 int锛屼繚鐣?'latest'锛?# ------------------------------------------------------------------
resume_milestone = None
resume_arg = opt.resume.strip() if opt.resume is not None else ""
if resume_arg:
    if resume_arg == "latest":
        resume_milestone = "latest"
    else:
        try:
            resume_milestone = int(resume_arg)
        except ValueError:
            raise ValueError(
                f"--resume must be an integer milestone or 'latest', got: {opt.resume!r}"
            )

# =============================================================================
# 鏋勫缓銆愬崟涓€銆戣嚜寤烘暟鎹泦
# =============================================================================
dataset = PairedImageDataset(
    dataroot=opt.dataroot,
    phase="train",
    image_size=image_size,
    augment_flip=True
)

# -------------------------------------------------
# Test dataset (low / high)
# -------------------------------------------------
sample_dataset = PairedImageDataset(
    dataroot=opt.dataroot,
    phase="test",
    image_size=image_size,
    augment_flip=False
)

# =============================================================================
# 璁粌閰嶇疆锛圫ingle Dataset锛?# =============================================================================
num_unet = 1
objective = "pred_res"          # residual learning
test_res_or_noise = "res"

train_num_steps = 600000
train_lr = 1e-4


# =============================================================================
# U-Net 涓诲共缃戠粶
# =============================================================================
base_model = UnetRes(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=target_channels,
    cond_channels=data_channels,
    num_unet=num_unet,
    condition=condition,
    objective=objective,
    test_res_or_noise=test_res_or_noise
)


# =============================================================================
# MeanFlow 涓绘ā鍨?# =============================================================================
meanflow = MeanFlow(
    base_model,
    channels=target_channels,
    image_size=image_size,
    flow_ratio=0.5,
    time_dist=['lognorm', -0.4, 1.0],
    cfg_ratio=0,
    cfg_scale=2.0,
    cfg_uncond='u',
    jvp_api="funtorch"
)


# =============================================================================
# Trainer
# =============================================================================
trainer = Trainer(
    meanflow,
    dataset,
    sample_dataset,
    opt,
    train_batch_size=train_batch_size,
    train_lr=train_lr,
    train_num_steps=train_num_steps,
    ema_decay=0.995,
    amp=False,
    fp16=False,
    results_folder=results_folder,
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    resume_milestone=resume_milestone,   # 鈫?浼犲叆鏂偣缂栧彿
)


# =============================================================================
# 寮€濮嬭缁?# =============================================================================
if __name__ == "__main__":
    trainer.train()
