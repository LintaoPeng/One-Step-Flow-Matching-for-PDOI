import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from test_meanflow_no_gt import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DEVICE,
    IMAGE_SIZE,
    MAT_KEY,
    NUM_WORKERS,
    PROJECT_ROOT,
    MatInputDataset,
    build_model,
    load_checkpoint,
)


GT_DIR = PROJECT_ROOT / "data" / "test" / "gt"
OUTPUT_DIR = PROJECT_ROOT / "results" / "psnr_test"


class PairedMatDataset(MatInputDataset):
    """Load MAT conditions and match their ground-truth PNGs by file stem."""

    def __init__(self, test_path, gt_dir, image_size=64, mat_key="mea1"):
        super().__init__(test_path, image_size=image_size, mat_key=mat_key)
        self.gt_dir = Path(gt_dir)
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(f"Ground-truth directory not found: {self.gt_dir}")

        missing = [path.stem for path in self.paths if not (self.gt_dir / f"{path.stem}.png").is_file()]
        if missing:
            preview = ", ".join(missing[:10])
            suffix = " ..." if len(missing) > 10 else ""
            raise FileNotFoundError(
                f"Missing {len(missing)} ground-truth PNG files in {self.gt_dir}: "
                f"{preview}{suffix}"
            )

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        gt_path = self.gt_dir / f"{sample['name']}.png"

        with Image.open(gt_path) as image:
            gt = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

        h, w = gt.shape
        if h < self.image_size or w < self.image_size:
            raise ValueError(
                f"Ground-truth image shape {(h, w)} is smaller than "
                f"requested crop {self.image_size}: {gt_path}"
            )
        if h != self.image_size or w != self.image_size:
            top = (h - self.image_size) // 2
            left = (w - self.image_size) // 2
            gt = gt[top:top + self.image_size, left:left + self.image_size]

        sample["high"] = torch.from_numpy(gt.copy()).unsqueeze(0)
        sample["gt_path"] = str(gt_path)
        return sample


def calculate_batch_psnr(restored, target):
    """Return one PSNR value per image for tensors whose valid range is [0, 1]."""
    mse = (restored - target).square().flatten(1).mean(dim=1)
    return torch.where(
        mse == 0,
        torch.full_like(mse, float("inf")),
        -10.0 * torch.log10(mse),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="MeanFlow inference and PSNR evaluation.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--test_path", type=Path, default=PROJECT_ROOT / "data" / "test")
    parser.add_argument("--gt_dir", type=Path, default=GT_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--mat_key", type=str, default=MAT_KEY)
    parser.add_argument("--data_channels", type=int, default=9)
    parser.add_argument("--target_channels", type=int, default=1)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_ema", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset = PairedMatDataset(
        args.test_path,
        args.gt_dir,
        image_size=args.image_size,
        mat_key=args.mat_key,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args).to(device)
    infer_model = load_checkpoint(model, args.checkpoint, device, use_ema=not args.no_ema)

    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "psnr.csv"
    print(f"Reconstructed images will be saved to: {image_dir}", flush=True)

    rows = []
    for batch in tqdm(loader, desc="Testing PSNR"):
        low = batch["low"].to(device, non_blocking=True)
        target = batch["high"].to(device, non_blocking=True).clamp(0, 1)
        restored = infer_model.restore_images(low, device=device).clamp(0, 1)

        if restored.shape != target.shape:
            raise ValueError(
                f"Restored/GT shape mismatch: {tuple(restored.shape)} vs {tuple(target.shape)}"
            )

        psnr_values = calculate_batch_psnr(restored, target).cpu().tolist()
        for image, name, psnr in zip(restored, batch["name"], psnr_values):
            save_path = image_dir / f"{name}.png"
            save_image(image.detach().cpu(), save_path)
            if not save_path.is_file():
                raise RuntimeError(f"Failed to save reconstructed image: {save_path}")
            rows.append((name, psnr))

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "psnr_db"])
        writer.writerows((name, f"{psnr:.6f}") for name, psnr in rows)

    average_psnr = float(np.mean([psnr for _, psnr in rows]))
    print(f"Test complete | Images: {len(rows)} | Average PSNR: {average_psnr:.4f} dB")
    print(f"Per-image PSNR saved to: {csv_path}")


if __name__ == "__main__":
    main()
