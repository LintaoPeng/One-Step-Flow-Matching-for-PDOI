"""修改 z= (1 - t_) * x +t_*c#z是流路径"""
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.vision_transformer import Attention
import torch.nn.functional as F
from einops import repeat, pack, unpack,rearrange
from torch.cuda.amp import autocast
from functools import partial
import math
import os
import random
from functools import partial
from pathlib import Path
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from ema_pytorch import EMA
from PIL import Image
import time
from torch import einsum, nn
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm.auto import tqdm
from src.UnetRes_Meanflow import tensor2img,UnetRes
from src.model import *


# ==========================================
# 数据预处理工具
# ==========================================

class Normalizer:
    """
    数据归一化类。
    支持两种模式：
    1. 'minmax': 将数据归一化到 [-1, 1] (通常用于图像原始像素)。
    2. 'mean_std': 使用均值和标准差进行标准化 (通常用于 VAE 的 Latent 空间)。
    """
    # minmax for raw image, mean_std for vae latent
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            # 将 mean/std 转换为 (1, 1, 1) 的形状以便广播
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        """
        从列表配置创建实例。
        config: [mode, mean, std]
        """
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        """归一化操作"""
        if self.mode == 'minmax':
            return x * 2 - 1  # [0, 1] -> [-1, 1]
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        """反归一化操作"""
        if self.mode == 'minmax':
            return (x + 1) * 0.5 # [-1, 1] -> [0, 1]
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x):
    """
    停止梯度传播。
    在计算损失函数的目标值(target)时使用，防止梯度回传到目标值的生成过程。
    """
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    自适应 L2 损失函数 (Adaptive L2 Loss)。
    这是一种鲁棒损失函数，类似于 Charbonnier Loss 或 Huber Loss 的变体。
    
    公式: Loss = w * ||Δ||^2
    其中权重 w = 1 / (||Δ||^2 + c)^(1-γ)
    
    当误差很大时，w 会变小，从而降低异常值（outliers）对梯度的影响。
    
    Args:
        error: 预测误差张量 (B, C, W, H)
        gamma: 控制损失函数形状的参数。
        c: 防止除零的小常数。
    Returns:
        标量 Loss
    """
    # 计算每个样本的均方误差
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    # 计算自适应权重，误差越大权重越小
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    # stopgrad(w) 很重要，意味着我们不优化权重本身，只将其作为常数系数
    return (stopgrad(w) * loss).mean()


# ==========================================
# 核心模型：MeanFlow
# ==========================================

class MeanFlow(nn.Module):
    def __init__(
        self,
        base_model,
        channels=1,
        image_size=32,
        normalizer=['minmax', None, None],
        # mean flow 设置
        flow_ratio=0.50, # 控制 t 和 r 相等的样本比例
        # 时间分布参数 (log-normal 分布)
        time_dist=['lognorm', -0.4, 1.0], 
        cfg_ratio=0.10,  # Classifier-Free Guidance 的 dropout 比例
        cfg_scale=2.0,   # Guidance scale (虽然代码中初始化似乎没直接用到 self.w 赋值)
        # 实验性设置
        cfg_uncond='u',
        jvp_api='autograd', # 计算雅可比向量积的 API 后端  
    ):
        super().__init__()
        self.model = base_model # 基础去噪/流模型 (通常是 DiT 或 UNet)
        self.channels = channels
        self.image_size = image_size
        self.use_cond = True # 标记是否使用条件

        self.normer = Normalizer.from_list(normalizer)

        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        # self.w = cfg_scale # 注意：原代码这里注释掉了，如果需要引导，应当赋值
        self.w = None      # 当前设为 None，表示不使用额外的引导缩放

        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api

        # 选择计算 JVP (Jacobian-Vector Product) 的方式
        # 两个后端签名不同：
        #   torch.func.jvp(f, primals, tangents)                          — funtorch，无 create_graph
        #   torch.autograd.functional.jvp(f, inputs, v, create_graph=...) — autograd，支持高阶导数
        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        self.jvp_api = jvp_api
        if jvp_api == 'funtorch':
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.create_graph = True
            
        self.cond_coef = 0.1 # 条件系数：控制退化图像 c 在流中的混合比例
        self.noise_coef = 0.1 # 噪声系数：控制随机噪声 e 在流中的混合比例

    def condition_to_state(self, c, target_channels=None):
        if target_channels is None:
            target_channels = self.channels
        if c.shape[1] == target_channels:
            return c
        if target_channels == 1:
            return c.mean(dim=1, keepdim=True)
        raise ValueError(
            f"Cannot map condition with {c.shape[1]} channels to {target_channels} channels"
        )

    def sample_t_r(self, batch_size, device):
        mu, sigma = self.time_dist[-2], self.time_dist[-1]
        samples = torch.randn(batch_size, 2, device=device) * sigma + mu
        samples = torch.sigmoid(samples)
        t = torch.max(samples[:, 0], samples[:, 1])
        r = torch.min(samples[:, 0], samples[:, 1])
        num_selected = int(self.flow_ratio * batch_size)
        idx = torch.randperm(batch_size, device=device)[:num_selected]
        r[idx] = t[idx]
        return t, r

    def flow_loss(self, x, c):
        """
        训练步骤：计算损失。
        这是一个基于流匹配（Flow Matching）变体的损失函数，利用 JVP 来进行泰勒展开式的约束。
        
        Args:
            x: 真实图像 (Target, clean image) (B, C, H, W)
            c: 条件/退化图像 (Condition, degraded image) (B, C, H, W)
        """
        if c is None:
            raise ValueError("退化图像 c 不能为 None")
            
        batch_size = x.shape[0]
        device = x.device

        # 1. 采样时间步
        t, r = self.sample_t_r(batch_size, device)

        # 调整形状以便广播
        t_ = rearrange(t, "b -> b 1 1 1")
        r_ = rearrange(r, "b -> b 1 1 1")

        # 2. 数据准备与归一化
        e = torch.randn_like(x) # 随机高斯噪声
        x = self.normer.norm(x) # 归一化真实图像 [-1, 1]
        c = self.normer.norm(c) # 归一化退化图像 [-1, 1]
        c_state = self.condition_to_state(c, x.shape[1])
        
        # 3. 构建流的插值轨迹 z (Input) 和 速度场 v (Target Velocity)
        # z 定义了从 Clean Image (t=0) 到 Noisy Condition (t=1) 的路径
        # 公式: z_t = (1-t)x + (1-coeff)*t*c + coeff*t*e
        # 当 t=0 时, z = x (干净图像)
        # 当 t=1 时, z ≈ c (退化图像 + 少量噪声)
        z = (1 - t_) * x + (t_ - self.cond_coef * t_) * c_state + self.noise_coef * t_ * e
        
        # v 是 z 关于 t 的导数 (dz/dt)
        # v = -x + c - cond_coef*c + noise_coef*e
        v = (1 - self.cond_coef) * c_state - x + self.noise_coef * e 

        # 4. 可选：模型预测与引导 (Guidance)
        # 如果启用了 w (cfg_scale)，这里会混合有条件和无条件的预测
        if self.w is not None:
            uncond = torch.zeros_like(c)  # 无条件输入
            with torch.no_grad():
                # u_t 是模型在当前位置 z 预测的速度
                u_t = self.model(z, t, t, uncond)
            # v_hat 是引导后的目标速度
            v_hat = self.w * v + (1 - self.w) * u_t
        else:
            v_hat = v

        # 代码中被注释掉的 CFG Mask 逻辑...
        
        # 特殊处理：如果 cfg_uncond 模式为 'v'，根据 r 的值决定是否使用真实速度 v
        if self.cfg_uncond == 'v':
            cfg_mask_v = rearrange(r, "b -> b 1 1 1").bool()
            v_hat = torch.where(cfg_mask_v, v, v_hat)

        # 5. JVP (Jacobian-Vector Product) 计算
        # 我们需要计算模型输出 u 关于输入 (z, t, r) 的变化率
        # 这里的目标是利用泰勒展开：u(r) ≈ u(t) + du/dt * (r - t)
        
        # 封装模型，固定条件图像 y=c
        # 注意：两个后端对函数签名的要求不同，需要分别处理
        #   funtorch : f(*primals)         → f 接收散开的位置参数
        #   autograd : f(inputs_tuple)     → f 接收一个 tuple，内部再解包
        if self.jvp_api == 'funtorch':
            # torch.func.jvp(func, primals_tuple, tangents_tuple)
            # func 的签名必须是 f(*primals)，即 f(z, t, r)
            def fn_funtorch(z, t, r):
                return self.model(z, t, r, y=c)

            primals  = (z, t, r)
            tangents = (v_hat, torch.ones_like(t), torch.zeros_like(r))
            u, dudt  = torch.func.jvp(fn_funtorch, primals, tangents)

        else:  # autograd
            # torch.autograd.functional.jvp(func, inputs_tuple, v_tuple, create_graph=...)
            # func 的签名必须是 f(inputs_tuple)，即 f((z, t, r))，内部再解包
            def fn_autograd(inputs):
                z_, t_, r_ = inputs
                return self.model(z_, t_, r_, y=c)

            inputs   = (z, t, r)
            tangents = (v_hat, torch.ones_like(t), torch.zeros_like(r))
            u, dudt  = torch.autograd.functional.jvp(fn_autograd, inputs, tangents, create_graph=True)

        # 6. 计算一致性目标 u_tgt
        # 这里的逻辑是：如果我们知道 t 时刻的速度 u 和变化率 dudt，
        # 我们可以推断出 r 时刻应该有的速度，或者修正当前速度。
        # u_tgt = v_hat - (t - r) * dudt 看起来像是一个一阶泰勒修正，
        # 试图强制模型满足流的一致性方程。
        u_tgt = v_hat - (t_ - r_) * dudt

        # 7. 计算速度场损失
        # error: 模型预测 u 与 目标 u_tgt 之间的差异
        error = u - stopgrad(u_tgt)
        flow_loss = adaptive_l2_loss(error)

        # 8. 计算 L1 重建损失
        # 复用本 step 已归一化的 c，从 t=1 做一次单步推理，预测出 x_0，再与真值 x 做 L1。
        # 推理逻辑与 restore_images 一致：
        #   z_init = noise_coef * e_recon + (1 - cond_coef) * c
        #   u_recon = model(z_init, t=1, r=0)
        #   x_pred  = z_init - u_recon
        # 整个过程用 no_grad 包裹，不引入额外的高阶梯度，保证训练效率。
        
        #e_recon = torch.randn_like(x)
        #z_recon = self.noise_coef * e_recon + (1 - self.cond_coef) * c
        #t_ones  = torch.ones((batch_size,),  device=device)
        #r_zeros = torch.zeros((batch_size,), device=device)
        #u_recon = self.model(z_recon, t_ones, r_zeros, y=None)
        #x_pred  = z_recon - u_recon  # 预测的归一化干净图像

        # x 此时已是归一化真值 (step 2 中已 norm)，直接做 L1
        #recon_loss = F.l1_loss(x_pred, x)

        # 9. 合并损失
        #loss = flow_loss + self.recon_weight * recon_loss

        mse_val = (stopgrad(error) ** 2).mean() # 记录均方误差用于监控
        return flow_loss, mse_val


    def rec_loss(self, x, c, return_pred=False):
        # 8. 计算 L1 重建损失
        # 复用本 step 已归一化的 c，从 t=1 做一次单步推理，预测出 x_0，再与真值 x 做 L1。
        # 推理逻辑与 restore_images 一致：
        #   z_init = noise_coef * e_recon + (1 - cond_coef) * c
        #   u_recon = model(z_init, t=1, r=0)
        #   x_pred  = z_init - u_recon
        # 整个过程用 no_grad 包裹，不引入额外的高阶梯度，保证训练效率。

        # Bug 1 修复：定义 batch_size 和 device
        batch_size = x.shape[0]
        device = x.device

        # Bug 2 修复：对输入进行归一化，与 flow_loss 保持一致
        x = self.normer.norm(x)
        c = self.normer.norm(c)
        c_state = self.condition_to_state(c, x.shape[1])

        e_recon = torch.randn_like(x)
        z_recon = self.noise_coef * e_recon + (1 - self.cond_coef) * c_state
        t_ones  = torch.ones((batch_size,),  device=device)
        r_zeros = torch.zeros((batch_size,), device=device)
        u_recon = self.model(z_recon, t_ones, r_zeros, y=c)
        x_pred  = z_recon - u_recon  # 预测的归一化干净图像

        # x 此时已是归一化真值，直接做 L1
        recon_loss = F.l1_loss(x_pred, x)

        if return_pred:
            return recon_loss, self.normer.unnorm(x_pred.clip(-1, 1))

        return recon_loss










    @torch.no_grad()
    def restore_images(self, degraded_images, sample_steps=1, device='cuda'):
        """
        图像修复推理函数 (Inference)。
        使用训练好的模型，从退化图像一步或多步恢复出清晰图像。
        
        Args:
            degraded_images: 退化图像 (B, C, H, W)
            sample_steps: 采样步数 (当前代码逻辑主要是针对单步/少步生成优化的)
            device: 设备
        """
        self.model.eval()
        
        batch_size = degraded_images.shape[0]
        # 归一化输入图像
        degraded_images = self.normer.norm(degraded_images.to(device))
        H, W = degraded_images.shape[-2:]
        degraded_state = self.condition_to_state(degraded_images, self.channels)
        
        # 1. 初始化起始点 z
        # 推理时，我们通常从 t=1 (退化状态) 开始，向 t=0 (清晰状态) 移动
        # 构造初始 z，使其分布与训练时的 t=1 状态一致：
        # z_init = (1-cond_coef)*c + noise_coef*e
        z = torch.randn(batch_size, self.channels, H, W, device=device)
        z = self.noise_coef * z + (1 - self.cond_coef) * degraded_state
        
        # 设置时间步
        t = torch.ones((batch_size,), device=device)   # Start at t=1
        r = torch.zeros((batch_size,), device=device)  # End at t=0 (conceptual target)

        # 2. 单步去噪 (One-step generation / Euler step)
        # 模型预测的是从 t=1 到 t=0 的变化量 u
        # 注意：这里的 u 实际上代表的是速度 v * dt。
        # 由于我们从 t=1 走到 t=0，dt = -1。
        # 公式: x_0 = x_1 - v * 1
        u = self.model(z, t, r, y=degraded_images)
        z = z - u  # 更新状态

        # 3. 反归一化并截断
        restored = self.normer.unnorm(z.clip(-1, 1))
        
        return restored
