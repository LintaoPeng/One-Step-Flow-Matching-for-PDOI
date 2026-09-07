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


# ==========================================
# 辅助函数
# ==========================================

def modulate(x, scale, shift):
    """
    AdaLN (Adaptive Layer Norm) 的核心调制操作。
    对归一化后的特征 x 进行仿射变换。
    x: 输入特征 (Batch, Sequence_Length, Dim)
    scale: 缩放因子 (Batch, Dim)
    shift: 偏移因子 (Batch, Dim)
    """
    # unsqueeze(1) 是为了在序列长度维度上进行广播 (Broadcasting)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def set_seed(SEED):
    """
    设置随机种子以保证结果的可复现性。
    包括 CPU, GPU (CUDA), Numpy 和 Python 自带的 random。
    """
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

# ==========================================
# 嵌入模块 (Embedders)
# ==========================================

class TimestepEmbedder(nn.Module):
    """
    时间步嵌入模块。
    将标量时间步 t 转换为向量嵌入。
    结合了正弦位置编码 (Sinusoidal Positional Embeddings) 和一个两层的 MLP。
    """
    def __init__(self, dim, nfreq=256):
        super().__init__()
        # MLP 用于将正弦频率特征映射到隐藏维度
        self.mlp = nn.Sequential(
            nn.Linear(nfreq, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        生成正弦位置编码，类似于 Transformer 中的位置编码，但用于时间步。
        t: 时间步张量 (N,)
        dim: 嵌入维度
        """
        half_dim = dim // 2
        # 计算频率: exp(-log(10000) * (0, 1, ..., d/2) / d/2)
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        
        # 计算 args = t * freqs
        args = t[:, None].float() * freqs[None]
        # 拼接 cos 和 sin 部分
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # 如果维度是奇数，补一个零 (虽然通常 dim 都是偶数)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        # 1. 获取正弦编码 (Batch, nfreq)
        t_freq = self.timestep_embedding(t, self.nfreq)
        # 2. 通过 MLP 提取特征 (Batch, dim)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    类别标签嵌入模块。
    用于类条件生成 (Class-Conditional Generation)。
    通常 num_classes + 1 是为了包含一个用于无分类器引导 (CFG) 的空标签。
    """
    def __init__(self, num_classes, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, dim)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding(labels)
        return embeddings


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm)。
    比 LayerNorm 计算量更小且在 Transformer 中通常更稳定。
    公式: x / RMS(x) * scale
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1)) # 可学习的增益参数

    def forward(self, x):
        # F.normalize 默认做 L2 归一化，这里用于实现 RMSNorm
        return F.normalize(x, dim=-1) * self.scale * self.g


# ==========================================
# 核心 Transformer 模块
# ==========================================

class DiTBlock(nn.Module):
    """
    DiT (Diffusion Transformer) 模块。
    这是模型的主体层，包含 Self-Attention 和 MLP，并使用 AdaLN 进行条件控制。
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        
        # 注意：这里假设 Attention 类在外部已定义
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm)
        # 禁用 fused attention 以兼容某些操作 (如 JVP)
        self.attn.fused_attn = False
        
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        
        # 注意：这里假设 Mlp 类在外部已定义
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        
        # AdaLN Modulation: 从条件向量 c 回归出 6 个参数
        # 这里的 c 通常是 时间步嵌入 + 类别/图像条件嵌入
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        # 1. 预测调制参数 (Shift, Scale, Gate)
        # 分别用于 MSA (Multi-Head Self Attention) 和 MLP 部分
        # chunk(6, dim=-1) 将长向量切分为 6 份
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        
        # 2. Attention 块:
        # Norm -> Modulate -> Attention -> Scale by Gate -> Residual Add
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )
        
        # 3. MLP 块:
        # Norm -> Modulate -> MLP -> Scale by Gate -> Residual Add
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    DiT 的最后一层。
    将 Transformer 的隐变量投影回图像的 Patch 空间。
    """
    def __init__(self, dim, patch_size, out_dim):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        # 线性层输出维度: Patch_Size * Patch_Size * Channels
        self.linear = nn.Linear(dim, patch_size * patch_size * out_dim)
        # 最后一层也使用 AdaLN，只需要 shift 和 scale
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        # 获取调制参数
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        # 归一化并调制
        x = modulate(self.norm_final(x), shift, scale)
        # 线性投影
        x = self.linear(x)
        return x


class FeatureFusionConditioner(nn.Module):
    """
    图像条件处理器。
    用于从受损/参考图像中提取特征，作为生成的条件。
    包含 Patch Embedding, 简单的 CNN 融合网络 和 全局池化投影。
    """
    def __init__(self, input_size, patch_size, in_channels, dim):
        super().__init__()
        # 假设 PatchEmbed 在外部定义
        self.patch_embed = PatchEmbed(input_size, patch_size, in_channels, dim)
        
        # 特征融合网络 (简单的卷积层)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        
        # 全局特征提取: 将空间特征压缩为全局向量
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, degraded_images):
        # degraded_images: (B, C, H, W)
        B, C, H, W = degraded_images.shape
        
        # 1. Patch embedding -> (B, num_patches, dim)
        x = self.patch_embed(degraded_images)  
        
        # 2. 重新整理为空间特征图 (B, dim, h, w) 以进行卷积操作
        patch_size = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(B, h, w, -1).permute(0, 3, 1, 2) 
        
        # 3. 特征融合 (卷积)
        fused_features = self.fusion_conv(x)  # (B, dim, h, w)
        
        # 4. 获取全局特征 (用于 AdaLN)
        global_feat = self.global_pool(fused_features).squeeze(-1).squeeze(-1)  # (B, dim)
        global_feat = self.global_proj(global_feat)  # (B, dim)
        
        # 返回全局特征和空间特征 (MFDiT 目前似乎只用了全局特征)
        return global_feat, fused_features


# ==========================================
# 主模型 MFDiT
# ==========================================

class MFDiT(nn.Module):
    """
    MFDiT: Multi-Feature (or similar) Diffusion Transformer.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        use_image_condition=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # 1. 输入图像的 Patch Embedding
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, dim)
        
        # 2. 时间步嵌入 (t: 扩散时间步, r: 可能是退化程度或细化步骤)
        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)
        
        # 3. 图像条件处理器初始化
        self.use_image_condition = use_image_condition
        if use_image_condition:
            self.cond_processor = FeatureFusionConditioner(input_size, patch_size, in_channels, dim)
        else:
            self.cond_processor = None

        num_patches = self.x_embedder.num_patches
        
        # 4. 位置编码 (Positional Embedding)
        # 初始化为全零的可学习参数，后续会用 sin-cos 固定编码填充
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=True)

        # 5. 堆叠 DiT Block
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        
        # 6. 输出层
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        """
        权重初始化逻辑。
        DiT 的关键在于将最后的输出层初始化为 0，使得初始状态下模型像一个恒等映射，有助于训练稳定性。
        """
        # 初始化 Transformer 层 (Xavier Uniform)
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 初始化 (并冻结/复制) 位置编码为 2D sin-cos 形式
        # get_2d_sincos_pos_embed 需要在外部定义
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 初始化 x_embedder 的投影层
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 初始化条件处理器
        if self.cond_processor is not None:
            # Patch Embed 初始化
            w = self.cond_processor.patch_embed.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.cond_processor.patch_embed.proj.bias, 0)
            
            # 卷积层初始化
            for layer in self.cond_processor.fusion_conv:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
            
            # 全局投影层初始化
            for layer in self.cond_processor.global_proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # 初始化时间步嵌入 MLP (Normal Distribution)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 关键: 将 DiT Block 中的 adaLN 最后一个线性层初始化为 0
        # 这样初始时 scale=0, shift=0, gate=0，block 输出为恒等映射
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 关键: 将最终输出层的 adaLN 和 Linear 初始化为 0
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        将 Patch 序列还原为图像张量。
        x: (N, T, patch_size**2 * C)
        Returns: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        # 计算特征图的高和宽 (假设是正方形)
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        # 1. Reshape 为 (N, h, w, p, p, c)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        # 2. 维度重排: (N, c, h, p, w, q)
        x = torch.einsum('nhwpqc->nchpwq', x)
        # 3. 展平为图像: (N, c, H, W)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, r, y=None):
        """
        前向传播函数。
        x: 噪声输入/潜变量 (N, C, H, W)
        t: 扩散时间步 (N,)
        r: 另一个条件时间步 (N,) (例如退化程度)
        y: 条件图像 (N, C, H, W)
        """
        H, W = x.shape[-2:]

        # 1. 输入 Embedding 并加上位置编码
        # x 变为 (N, T, D), 其中 T 是 patch 数量
        x = self.x_embedder(x) + self.pos_embed 

        # 2. 处理时间步嵌入
        t = self.t_embedder(t) # (N, D)
        r = self.r_embedder(r) # (N, D)
        # 将两个时间步信息相加融合
        t = t + r

        # 3. 准备调节向量 c
        c = t
        if self.use_image_condition and y is not None:
            # 提取图像条件的全局特征和空间特征
            global_cond, spatial_cond = self.cond_processor(y)
            # 将全局图像特征加到时间步特征上
            # 注意：此处 spatial_cond 虽然被计算了，但在该代码段中未被使用 (可能在其他变体中使用 CrossAttn)
            c = c + global_cond  # (N, D)

        # 4. 通过 DiT Blocks
        for i, block in enumerate(self.blocks):
            # 将 x 和调节向量 c 传入 Block
            x = block(x, c)                      

        # 5. 最终层投影
        x = self.final_layer(x, c) # (N, T, patch_size**2 * out_channels)
        
        # 6. 还原为图像空间
        x = self.unpatchify(x)     # (N, out_channels, H, W)
        return x


# Positional embedding from:
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



