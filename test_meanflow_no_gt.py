import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from ema_pytorch import EMA
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from src.UnetRes_Meanflow import UnetRes
from src.meanflow import MeanFlow


# =============================================================================
# Test config
# 直接在这里指定预训练权重、测试数据路径和输出路径。
# 运行方式: python test_meanflow_no_gt.py
# 如果命令行传入 --checkpoint / --test_path / --output_dir，会覆盖这里的设置。
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "ckpt_single_dataset" / "model-best.pt"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "test"
OUTPUT_DIR = PROJECT_ROOT / "results" / "no_gt_test"

IMAGE_SIZE = 64
BATCH_SIZE = 1
NUM_WORKERS = 0
MAT_KEY = "mea1"
DEVICE = ""  # "" 表示自动选择 cuda/cpu；也可以写 "cuda:0" 或 "cpu"
USE_EMA = True


def natural_key(path):
    name = Path(path).stem
    try:
        return int(name)
    except ValueError:
        return name


def minmax_norm(x):
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def load_mat_stack(path, key="mea1"):
    mat = sio.loadmat(path)
    if key in mat:
        arr = mat[key]
    else:
        candidates = [v for k, v in mat.items() if not k.startswith("__")]
        if not candidates:
            raise KeyError(f"No array found in mat file: {path}")
        arr = candidates[0]

    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D mat array, got shape {arr.shape}: {path}")

    if arr.shape[0] == 9:
        pass
    elif arr.shape[-1] == 9:
        arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Expected 9 channels in mat array, got shape {arr.shape}: {path}")

    return minmax_norm(arr)


class MatInputDataset(Dataset):
    def __init__(self, test_path, image_size=64, mat_key="mea1"):
        self.image_size = image_size
        self.mat_key = mat_key

        root = Path(test_path)
        input_dir = root / "input" if (root / "input").is_dir() else root
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        self.paths = sorted(input_dir.glob("*.mat"), key=natural_key)
        if not self.paths:
            raise RuntimeError(f"Found 0 .mat files in: {input_dir}")

    def __len__(self):
        return len(self.paths)

    def _center_crop(self, arr):
        _, h, w = arr.shape
        if h == self.image_size and w == self.image_size:
            return arr
        if h < self.image_size or w < self.image_size:
            raise ValueError(
                f"Image shape {(h, w)} is smaller than requested crop {self.image_size}"
            )
        top = (h - self.image_size) // 2
        left = (w - self.image_size) // 2
        return arr[:, top:top + self.image_size, left:left + self.image_size]

    def __getitem__(self, idx):
        path = self.paths[idx]
        low = load_mat_stack(str(path), self.mat_key)
        low = self._center_crop(low)
        return {
            "low": torch.from_numpy(low.astype(np.float32)),
            "name": path.stem,
            "path": str(path),
        }


def build_model(args):
    base_model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=args.target_channels,
        cond_channels=args.data_channels,
        num_unet=1,
        condition=True,
        objective="pred_res",
        test_res_or_noise="res",
    )

    return MeanFlow(
        base_model,
        channels=args.target_channels,
        image_size=args.image_size,
        flow_ratio=0.5,
        time_dist=["lognorm", -0.4, 1.0],
        cfg_ratio=0,
        cfg_scale=2.0,
        cfg_uncond="u",
        jvp_api="funtorch",
    )


def load_checkpoint(model, checkpoint_path, device, use_ema=True):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and use_ema and "ema" in checkpoint:
        ema = EMA(model, beta=0.995, update_every=10)
        ema.load_state_dict(checkpoint["ema"])
        ema.to(device)
        ema.ema_model.eval()
        step = checkpoint.get("step", "unknown")
        print(f"Loaded EMA weights from {checkpoint_path} (step={step})")
        return ema.ema_model

    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    step = checkpoint.get("step", "unknown") if isinstance(checkpoint, dict) else "unknown"
    print(f"Loaded model weights from {checkpoint_path} (step={step})")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="MeanFlow inference without GT.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH, help="Path to .pt/.pth checkpoint.")
    parser.add_argument(
        "--test_path",
        type=Path,
        default=TEST_DATA_PATH,
        help="Test data path. It can be a folder containing .mat files or a root with an input/ folder.",
    )
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="Folder to save PNG results.")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE, help="Center crop size used by training.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Inference batch size.")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="DataLoader workers.")
    parser.add_argument("--mat_key", type=str, default=MAT_KEY, help="Variable key in .mat files.")
    parser.add_argument("--data_channels", type=int, default=9, help="Input condition channels.")
    parser.add_argument("--target_channels", type=int, default=1, help="Output image channels.")
    parser.add_argument("--device", type=str, default=DEVICE, help="cuda, cuda:0 or cpu. Default: auto.")
    parser.add_argument(
        "--no_ema",
        action="store_true",
        default=not USE_EMA,
        help="Load raw model weights instead of EMA weights.",
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = MatInputDataset(args.test_path, image_size=args.image_size, mat_key=args.mat_key)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args).to(device)
    infer_model = load_checkpoint(model, args.checkpoint, device, use_ema=not args.no_ema)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generated images will be saved to: {output_dir.resolve()}", flush=True)

    saved_count = 0
    for batch in tqdm(loader, desc="Testing"):
        low = batch["low"].to(device)
        restored = infer_model.restore_images(low, device=device).clamp(0, 1)

        for image, name in zip(restored, batch["name"]):
            save_path = output_dir / f"{name}.png"
            # Move each generated image off the accelerator before writing it.
            # This also ensures that save_image receives an independent CHW tensor.
            save_image(image.detach().cpu(), save_path)
            if not save_path.is_file():
                raise RuntimeError(f"Failed to save generated image: {save_path}")
            saved_count += 1

    if saved_count != len(dataset):
        raise RuntimeError(
            f"Saved {saved_count} images, but the test dataset contains {len(dataset)} samples."
        )

    print(f"Done. Saved {saved_count} result images to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
