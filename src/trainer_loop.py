"""淇敼 z= (1 - t_) * x +t_*c#z鏄祦璺緞"""
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
from src.meanflow import *
from src.model import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.utils import save_image



from data.paired_image_dataset import PairedImageDataset


# ==========================================
# 杈呭姪鍑芥暟
# ==========================================

def exists(x):
    return x is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


# ==========================================
# Trainer
# ==========================================

class Trainer(object):
    def __init__(
        self,
        meanflow_model,
        dataset,
        sample_dataset,
        opts,
        *,
        train_batch_size=16,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        results_folder='./results',
        amp=False,
        fp16=False,
        split_batches=True,
        condition=True,
        recon_weight=0.1,
        resume_milestone=None,   # 鈫?鏂板锛氬惎鍔ㄦ椂鑷姩鍔犺浇鐨?milestone
    ):
        super().__init__()

        # -------------------------------------------------
        # Accelerator
        # -------------------------------------------------
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp

        self.device = self.accelerator.device
        self.model = meanflow_model
        self.condition = condition

        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.save_and_sample_every = save_and_sample_every
        self.sample_dataset = sample_dataset
        self.recon_weight = recon_weight  # L1 閲嶅缓鎹熷け鏉冮噸

        # -------------------------------------------------
        # DataLoader (train)
        # -------------------------------------------------
        self.dl = cycle(
            self.accelerator.prepare(
                DataLoader(
                    dataset,
                    batch_size=train_batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
            )
        )

        # -------------------------------------------------
        # Optimizer
        # -------------------------------------------------
        self.opt0 = Adam(
            self.model.parameters(),
            lr=train_lr,
            betas=adam_betas
        )

        # -------------------------------------------------
        # results_folder锛氭墍鏈夎繘绋嬮兘闇€瑕侊紝鐢ㄤ簬 load() 鎷艰矾寰?        # -------------------------------------------------
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # EMA & folders锛堜粎涓昏繘绋嬫寔鏈?EMA锛?        # -------------------------------------------------
        if self.accelerator.is_main_process:
            self.ema = EMA(
                meanflow_model,
                beta=ema_decay,
                update_every=ema_update_every
            )

        self.step = 0
        self.best_psnr = float("-inf")

        # prepare model & optimizer
        self.model, self.opt0 = self.accelerator.prepare(
            self.model, self.opt0
        )

        if self.accelerator.is_main_process:
            print(f"[Accelerate] Using {self.accelerator.num_processes} GPUs")

        # -------------------------------------------------
        # 鏂偣缁锛歘_init__ 缁撴潫鍓嶈嚜鍔ㄥ姞杞?        # -------------------------------------------------
        if resume_milestone is not None:
            self.load(resume_milestone)

    # ======================================
    # Save / Load
    # ======================================

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'best_psnr': self.best_psnr,
            'model': self.accelerator.get_state_dict(self.model),
            'opt0': self.opt0.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': (
                self.accelerator.scaler.state_dict()
                if exists(self.accelerator.scaler) else None
            )
        }

        torch.save(
            data,
            str(self.results_folder / f'model-{milestone}.pt')
        )
        print(f"[Trainer] Checkpoint saved 鈫?model-{milestone}.pt  (step={self.step})")

    def save_best(self):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'best_psnr': self.best_psnr,
            'model': self.accelerator.get_state_dict(self.model),
            'opt0': self.opt0.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': (
                self.accelerator.scaler.state_dict()
                if exists(self.accelerator.scaler) else None
            )
        }

        torch.save(
            data,
            str(self.results_folder / 'model-best.pt')
        )
        print(
            f"[Trainer] Best checkpoint saved -> model-best.pt "
            f"(best_psnr={self.best_psnr:.4f}, step={self.step})"
        )

    def load(self, milestone):
        """
        浠?results_folder/model-{milestone}.pt 鎭㈠璁粌鐘舵€併€?
        Parameters
        ----------
        milestone : int | str
            checkpoint 缂栧彿锛屼笌 save() 淇濇寔涓€鑷淬€?            浼犲叆 'latest' 鏃惰嚜鍔ㄦ煡鎵剧紪鍙锋渶澶х殑鏂囦欢銆?        """
        # ---- 鏀寔 'latest' 鍏抽敭瀛?----
        if milestone == 'latest':
            milestone = self._find_latest_milestone()
            if milestone is None:
                print("[Trainer] No checkpoint found, starting from scratch.")
                return
            print(f"[Trainer] 'latest' resolved to milestone={milestone}")

        path = self.results_folder / f'model-{milestone}.pt'
        if not path.exists():
            print(f"[Trainer] Checkpoint {path} not found, starting from scratch.")
            return

        # map_location 纭繚璺ㄨ澶囧姞杞?        data = torch.load(str(path), map_location=self.device)
        data = torch.load(str(path), map_location=self.device)

        # 鎭㈠妯″瀷鏉冮噸锛坲nwrap 鍐嶈祴鍊硷級
        raw_model = self.accelerator.unwrap_model(self.model)
        raw_model.load_state_dict(data['model'])

        # 鎭㈠鍏ㄥ眬 step
        self.step = data['step']
        self.best_psnr = data.get('best_psnr', self.best_psnr)

        # 鎭㈠浼樺寲鍣ㄧ姸鎬?        self.opt0.load_state_dict(data['opt0'])
        self.opt0.load_state_dict(data['opt0'])

        # 鎭㈠ EMA锛堜粎涓昏繘绋嬫寔鏈夛級
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        # 鎭㈠ AMP scaler锛堝鏋滃瓨鍦級
        if exists(self.accelerator.scaler) and exists(data.get('scaler')):
            self.accelerator.scaler.load_state_dict(data['scaler'])

        print(f"[Trainer] Checkpoint loaded: {path}  (resume from step={self.step})")

    def _find_latest_milestone(self):
        """Return the largest numbered checkpoint milestone, or None."""
        ckpt_files = list(self.results_folder.glob('model-*.pt'))
        if not ckpt_files:
            return None
        milestones = []
        for f in ckpt_files:
            stem = f.stem  # e.g. 'model-30'
            try:
                milestones.append(int(stem.split('-', 1)[1]))
            except (IndexError, ValueError):
                pass
        return max(milestones) if milestones else None

    # ======================================
    # Train
    # ======================================

    def train(self):
        accelerator = self.accelerator
        self.model.train()

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process
        ) as pbar:

            while self.step < self.train_num_steps:

                batch = next(self.dl)

                if self.condition:
                    gt = batch["high"].to(self.device)
                    cond = batch["low"].to(self.device)
                else:
                    gt = batch.to(self.device)
                    cond = None

                with accelerator.autocast():
                    if self.condition:
                        model = self.accelerator.unwrap_model(self.model)
                        flowloss, mse_val = model.flow_loss(gt, cond)
                        recloss, train_restored = model.rec_loss(gt, cond, return_pred=True)
                        loss = flowloss + self.recon_weight * recloss
                        with torch.no_grad():
                            train_mse = F.mse_loss(train_restored.clamp(0, 1), gt.clamp(0, 1))
                            train_psnr = -10.0 * torch.log10(train_mse + 1e-8)
                    else:
                        model = self.accelerator.unwrap_model(self.model)
                        loss, _ = model.flow_loss(gt)
                        flowloss = loss
                        # condition=False 鏃跺崰浣嶏紝閬垮厤涓嬫柟寮曠敤鏈畾涔夊彉閲?                        flowloss = loss
                        recloss = torch.zeros_like(loss)
                        train_psnr = torch.zeros_like(loss)

                    accelerator.backward(loss)

                accelerator.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )

                self.opt0.step()
                self.opt0.zero_grad()

                accelerator.wait_for_everyone()
                self.step += 1

                # EMA + save + test
                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()

                    # 姣?1000 姝ヨ皟鐢ㄤ竴娆?test
                    if self.step % 1000 == 0:
                        print(f"\nStep {self.step}: Running test...")
                        test_psnr = self.test(
                            self.sample_dataset,
                            save_folder=str(self.results_folder / f"test_step_{self.step}")
                        )
                        if test_psnr > self.best_psnr:
                            self.best_psnr = test_psnr
                            self.save_best()

                    # Save numbered checkpoint periodically.
                    if self.step % (self.save_and_sample_every * 10) == 0:
                        self.save(self.step // self.save_and_sample_every)

                    # Save latest snapshot periodically.
                    if self.step % self.save_and_sample_every == 0:
                        self.save(999999)
                pbar.set_description(
                    f'loss: {loss.item():.4f} | '
                    f'fm {flowloss.item():.4f} | '
                    f'rec {recloss.item():.4f} | '
                    f'train_psnr {train_psnr.item():.4f} | '
                    f'best_psnr {self.best_psnr:.4f}'
                )
                pbar.update(1)

        accelerator.print('training complete')

    # ======================================
    # Test
    # ======================================

    def test(self, dataset, save_folder="./results/test_output"):
        self.ema.to(self.device)
        self.ema.ema_model.eval()

        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        psnr_list = []

        print("Testing start...")

        with torch.no_grad():
            for i, item in enumerate(tqdm(loader)):
                low = item["low"].to(self.device)
                high = item["high"].to(self.device)

                # 鎭㈠鍥惧儚
                restored = self.ema.ema_model.restore_images(low, device=self.device)

                # 淇濆瓨缁撴灉
                file_name = f"sample_{i:05d}.png"
                save_path = save_folder / file_name
                save_image(restored, str(save_path))

                high_np = high.squeeze(0).permute(1, 2, 0).cpu().numpy()
                high_np = np.clip(high_np, 0, 1)

                restored_np = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
                restored_np = np.clip(restored_np, 0, 1)

                psnr_val = compare_psnr(high_np, restored_np, data_range=1.0)
                psnr_list.append(psnr_val)

        avg_psnr = np.mean(psnr_list)
        print(f"Test complete | Avg PSNR: {avg_psnr:.4f}")
        self.ema.ema_model.train()
        return avg_psnr

    # ======================================
    # Utils
    # ======================================

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        self.results_folder.mkdir(parents=True, exist_ok=True)
