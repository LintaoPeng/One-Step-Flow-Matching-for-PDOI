import os

import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def _natural_key(path):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(name)
    except ValueError:
        return name


def _minmax_norm(x):
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def _load_mat_stack(path, key="mea1"):
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

    return _minmax_norm(arr)


def _load_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return _minmax_norm(img)[None, ...]


class PairedImageDataset(Dataset):
    """
    Paired optical stack dataset.

    - low:  9 x 64 x 64 normalized measurement stack from .mat files
    - high: 1 x 64 x 64 normalized grayscale ground truth image
    """

    def __init__(
        self,
        dataroot,
        phase="train",
        image_size=64,
        augment_flip=True,
        mat_key="mea1",
        convert_rgb=True,
    ):
        super().__init__()

        self.phase = phase
        self.image_size = image_size
        self.augment_flip = augment_flip and phase == "train"
        self.mat_key = mat_key

        low_dir = os.path.join(dataroot, phase, "input")
        high_dir = os.path.join(dataroot, phase, "gt")

        assert os.path.isdir(low_dir), f"Not found: {low_dir}"
        assert os.path.isdir(high_dir), f"Not found: {high_dir}"

        self.low_paths = sorted(
            [
                os.path.join(low_dir, f)
                for f in os.listdir(low_dir)
                if f.lower().endswith(".mat")
            ],
            key=_natural_key,
        )

        self.high_paths = sorted(
            [
                os.path.join(high_dir, f)
                for f in os.listdir(high_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            ],
            key=_natural_key,
        )

        assert len(self.low_paths) == len(self.high_paths), "input / gt file count mismatch"

    def __len__(self):
        return len(self.low_paths)

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
        low = _load_mat_stack(self.low_paths[idx], self.mat_key)
        high = _load_gray_image(self.high_paths[idx])

        low = self._center_crop(low)
        high = self._center_crop(high)

        if self.augment_flip:
            if torch.rand(()) < 0.5:
                low = np.flip(low, axis=2).copy()
                high = np.flip(high, axis=2).copy()
            if torch.rand(()) < 0.5:
                low = np.flip(low, axis=1).copy()
                high = np.flip(high, axis=1).copy()

        return {
            "low": torch.from_numpy(low.astype(np.float32)),
            "high": torch.from_numpy(high.astype(np.float32)),
            "path": self.low_paths[idx],
        }
