"""Read the figshare brain-tumour .mat files, cache them at a fixed size, and split by patient."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def read_mat(path: str | Path) -> dict:
    """One slice: MATLAB v7.3 (HDF5) struct `cjdata` with image, tumorMask, label, PID."""
    import h5py

    with h5py.File(path, "r") as f:
        g = f["cjdata"]
        return {
            "image": np.asarray(g["image"]).T.astype(np.float32),  # MATLAB is column-major
            "mask": np.asarray(g["tumorMask"]).T.astype(np.uint8),
            "label": int(np.asarray(g["label"]).item()) - 1,
            "pid": "".join(chr(int(c)) for c in np.asarray(g["PID"]).ravel()).strip(),
        }


def normalise(img: np.ndarray) -> np.ndarray:
    """Clip to the 0.5-99.5th percentile and scale to [0, 1] (MRI intensities have no fixed unit)."""
    lo, hi = np.percentile(img, [0.5, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def resize_pair(img: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    it = torch.from_numpy(img)[None, None]
    mt = torch.from_numpy(mask.astype(np.float32))[None, None]
    img_r = F.interpolate(it, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    mask_r = F.interpolate(mt, size=(size, size), mode="nearest")
    return img_r[0, 0].numpy(), mask_r[0, 0].numpy().astype(np.uint8)


def build_cache(mat_dir: str | Path, out: str | Path, size: int, progress=None) -> dict:
    files = sorted(Path(mat_dir).glob("*.mat"), key=lambda p: int(p.stem))
    imgs, masks, labels, pids, ids = [], [], [], [], []
    for p in progress(files) if progress else files:
        rec = read_mat(p)
        img, mask = resize_pair(normalise(rec["image"]), rec["mask"], size)
        imgs.append((img * 255).round().astype(np.uint8))
        masks.append(mask)
        labels.append(rec["label"])
        pids.append(rec["pid"])
        ids.append(int(p.stem))
    data = {"images": np.stack(imgs), "masks": np.stack(masks), "labels": np.array(labels, dtype=np.int64),
            "pids": np.array(pids), "ids": np.array(ids)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    return data


def load_cache(path: str | Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def patient_split(pids: np.ndarray, labels: np.ndarray, val_frac: float, test_frac: float, seed: int) -> dict:
    """Assign whole patients to train/val/test, stratified by tumour type."""
    rng = np.random.default_rng(seed)
    patient_label = {}
    for p, y in zip(pids, labels, strict=True):
        patient_label.setdefault(p, y)
    split = {"train": [], "val": [], "test": []}
    for cls in sorted(set(patient_label.values())):
        pts = sorted(p for p, y in patient_label.items() if y == cls)
        pts = [pts[i] for i in rng.permutation(len(pts))]
        n_test, n_val = math.ceil(len(pts) * test_frac), math.ceil(len(pts) * val_frac)
        split["test"] += pts[:n_test]
        split["val"] += pts[n_test : n_test + n_val]
        split["train"] += pts[n_test + n_val :]
    return {k: sorted(v) for k, v in split.items()}


def indices_for(pids: np.ndarray, patients: list[str]) -> np.ndarray:
    return np.flatnonzero(np.isin(pids, patients))


class SliceDataset(Dataset):
    """Returns (image 1xHxW float, mask 1xHxW float, label). Augmentation keeps image and mask aligned."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.images, self.masks, self.labels, self.augment = images, masks, labels, augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = torch.from_numpy(self.images[i].astype(np.float32) / 255.0)[None]
        mask = torch.from_numpy(self.masks[i].astype(np.float32))[None]
        if self.augment:
            img, mask = random_affine(img, mask)
            img = intensity_jitter(img)
        return img, mask, int(self.labels[i])


def random_affine(img: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    angle = math.radians(np.random.uniform(-15, 15))
    scale = np.random.uniform(0.9, 1.1)
    tx, ty = np.random.uniform(-0.06, 0.06, size=2)
    flip = -1.0 if np.random.rand() < 0.5 else 1.0
    cos, sin = math.cos(angle) / scale, math.sin(angle) / scale
    theta = torch.tensor([[flip * cos, -sin, tx], [flip * sin, cos, ty]], dtype=torch.float32)[None]
    grid = F.affine_grid(theta, (1, 1, *img.shape[-2:]), align_corners=False)
    img = F.grid_sample(img[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    mask = F.grid_sample(mask[None], grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
    return img, mask


def intensity_jitter(img: torch.Tensor) -> torch.Tensor:
    gamma = np.random.uniform(0.8, 1.25)
    img = img.clamp(0, 1) ** gamma
    img = img * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)
    return (img + 0.01 * torch.randn_like(img)).clamp(0, 1)
