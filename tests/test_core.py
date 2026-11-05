import numpy as np
import torch

from oncoscope.data import SliceDataset, normalise, patient_split, random_affine
from oncoscope.evaluate import erase_tumour, reliability, tumour_only
from oncoscope.model import TumourNet, dice_loss, grad_cam
from oncoscope.train import dice_per_image


def test_patient_split_is_disjoint_and_stratified():
    pids = np.array([f"p{i // 4}" for i in range(120)])  # 30 patients x 4 slices
    labels = np.array([(i // 4) % 3 for i in range(120)])
    split = patient_split(pids, labels, val_frac=0.1, test_frac=0.2, seed=0)
    sets = [set(v) for v in split.values()]
    assert sum(map(len, sets)) == 30
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    test_classes = {labels[pids == p][0] for p in split["test"]}
    assert test_classes == {0, 1, 2}


def test_normalise_range():
    x = normalise(np.random.default_rng(0).normal(500, 200, (64, 64)).astype(np.float32))
    assert x.min() >= 0 and x.max() <= 1


def test_affine_keeps_mask_binary_and_aligned():
    img = torch.zeros(1, 64, 64)
    mask = torch.zeros(1, 64, 64)
    img[:, 20:40, 20:40] = 1
    mask[:, 20:40, 20:40] = 1
    np.random.seed(0)
    img2, mask2 = random_affine(img, mask)
    assert set(mask2.unique().tolist()) <= {0.0, 1.0}
    overlap = ((img2 > 0.5) & (mask2 > 0.5)).sum() / (mask2 > 0.5).sum()
    assert overlap > 0.9


def test_dice():
    t = torch.zeros(2, 1, 8, 8)
    t[:, :, 2:6, 2:6] = 1
    assert torch.allclose(dice_per_image(t, t), torch.ones(2))
    assert dice_loss(torch.full_like(t, 20.0) * (2 * t - 1), t) < 0.01


def test_model_shapes_and_gradcam():
    model = TumourNet(pretrained=False)
    x = torch.rand(2, 1, 64, 64)
    seg, logits = model(x)
    assert seg.shape == (2, 1, 64, 64) and logits.shape == (2, 3)
    cam, _ = grad_cam(model, x)
    assert cam.shape == (2, 64, 64) and float(cam.max()) <= 1.0 + 1e-6


def test_shortcut_transforms():
    rng = np.random.default_rng(1)
    img = rng.uniform(0, 50, (64, 64)).astype(np.float32)
    mask = np.zeros((64, 64), np.uint8)
    mask[25:35, 25:35] = 1
    img[mask > 0] = 255  # bright tumour
    erased = erase_tumour(img, mask)
    assert erased[mask > 0].mean() < 100  # tumour brightness gone
    only = tumour_only(img, mask, pad=2)
    assert only[0, 0] == 0 and only[30, 30] == 255


def test_dataset_and_reliability():
    ds = SliceDataset(np.zeros((3, 32, 32), np.uint8), np.zeros((3, 32, 32), np.uint8), np.array([0, 1, 2]), True)
    img, mask, y = ds[1]
    assert img.shape == (1, 32, 32) and y == 1
    probs = np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1]])
    table, ece = reliability(probs, np.array([0, 1]))
    assert 0 <= ece <= 1 and table.n.sum() == 2
