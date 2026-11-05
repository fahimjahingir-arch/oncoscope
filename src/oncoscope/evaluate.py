"""Held-out patient evaluation: segmentation, classification, calibration, Grad-CAM localisation, shortcut tests.

Shortcut tests ask whether the tumour-type prediction actually depends on the tumour:
  * tumour erased  - the tumour (plus a margin) is inpainted from surrounding tissue
  * tumour only    - everything outside a box around the tumour is blanked
If accuracy stays high with the tumour erased, the classifier is reading context (slice position, view, anatomy),
not the lesion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from oncoscope import CLASSES
from oncoscope.data import SliceDataset, indices_for, load_cache
from oncoscope.model import TumourNet, grad_cam
from oncoscope.train import predict


def load_model(path: str | Path) -> tuple[TumourNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = TumourNet(n_classes=len(CLASSES), **{**cfg["model"], "pretrained": False})
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg


def erase_tumour(img: np.ndarray, mask: np.ndarray, margin: int = 6, sigma: float = 12.0) -> np.ndarray:
    """Fill the (dilated) tumour with a normalised-convolution blur of the surrounding tissue."""
    hole = ndimage.binary_dilation(mask > 0, iterations=margin)
    keep = (~hole).astype(np.float32)
    num = ndimage.gaussian_filter(img * keep, sigma)
    den = ndimage.gaussian_filter(keep, sigma)
    filled = num / np.maximum(den, 1e-3)
    return np.where(hole, filled, img).astype(img.dtype)


def tumour_only(img: np.ndarray, mask: np.ndarray, pad: int = 12) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    out = np.zeros_like(img)
    if len(ys):
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
        out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    return out


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    conf, correct = probs.max(1), (probs.argmax(1) == y).astype(float)
    rows, ece = [], 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
            rows.append({"confidence": conf[m].mean(), "accuracy": correct[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows), float(ece)


def cls_scores(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "kappa": float(cohen_kappa_score(y, pred))}


def evaluate(run: str | Path, log=print) -> dict:
    run = Path(run)
    model, cfg = load_model(run / "best.pt")
    data = load_cache(cfg["data"]["cache"])
    split = json.loads((run / "split.json").read_text())
    idx = indices_for(data["pids"], split["test"])
    imgs, masks, y = data["images"][idx], data["masks"][idx], data["labels"][idx]
    pids, ids = data["pids"][idx], data["ids"][idx]
    ds = SliceDataset(imgs, masks, y)

    seg, probs, dice = predict(model, ds)
    pred = probs.argmax(1)
    seg_np = seg[:, 0].numpy()
    inter = (seg_np & (masks > 0)).sum((1, 2))
    union = (seg_np | (masks > 0)).sum((1, 2))
    metrics = {"n_patients": len(split["test"]), "n_slices": len(idx), **cls_scores(y, pred),
               "dice_mean": float(dice.mean()), "dice_median": float(np.median(dice)),
               "iou_mean": float(np.mean(inter / np.maximum(union, 1)))}

    # patient-level diagnosis: average slice probabilities per patient
    pdf = pd.DataFrame(probs, columns=CLASSES).assign(pid=pids, y=y)
    per_pt = pdf.groupby("pid").agg({**{c: "mean" for c in CLASSES}, "y": "first"})
    metrics["patient_accuracy"] = float(accuracy_score(per_pt.y, per_pt[CLASSES].to_numpy().argmax(1)))

    rel, ece = reliability(probs, y)
    metrics["ece"] = ece
    log(f"TEST  {metrics['n_patients']} patients / {metrics['n_slices']} slices | acc {metrics['accuracy']:.3f}  "
        f"F1 {metrics['macro_f1']:.3f}  patient-acc {metrics['patient_accuracy']:.3f} | Dice {metrics['dice_mean']:.3f} "
        f"(median {metrics['dice_median']:.3f}) | ECE {ece:.3f}")

    # Grad-CAM: how much of the classifier's attention lands on the tumour?
    cams = []
    for i in range(0, len(idx), 32):
        x = torch.from_numpy(imgs[i : i + 32].astype(np.float32) / 255)[:, None]
        cams.append(grad_cam(model, x)[0].numpy())
    cams = np.concatenate(cams)
    m = masks > 0
    near = np.stack([ndimage.binary_dilation(k, iterations=8) for k in m])
    cam_in = (cams * m).sum((1, 2)) / np.maximum(cams.sum((1, 2)), 1e-8)
    area = m.mean((1, 2))
    peak = cams.reshape(len(cams), -1).argmax(1)
    pointing = near.reshape(len(near), -1)[np.arange(len(near)), peak]
    metrics.update({"cam_energy_in_tumour": float(cam_in.mean()), "tumour_area_fraction": float(area.mean()),
                    "cam_pointing_game": float(pointing.mean())})
    log(f"Grad-CAM: {100 * cam_in.mean():.1f}% of attention inside a tumour covering {100 * area.mean():.1f}% of "
        f"the image | peak inside tumour (+8px) {100 * pointing.mean():.1f}%")

    # shortcut tests
    def accuracy_on(transform) -> tuple[float, np.ndarray]:
        alt = np.stack([transform(im.astype(np.float32), mk) for im, mk in zip(imgs, masks, strict=True)])
        _, p, _ = predict(model, SliceDataset(np.clip(alt, 0, 255).astype(np.uint8), masks, y))
        return float(accuracy_score(y, p.argmax(1))), p

    acc_erased, p_erased = accuracy_on(erase_tumour)
    acc_only, _ = accuracy_on(tumour_only)
    majority = float(np.bincount(y).max() / len(y))
    shortcut = pd.DataFrame([
        {"condition": "full image", "accuracy": metrics["accuracy"]},
        {"condition": "tumour only (context blanked)", "accuracy": acc_only},
        {"condition": "tumour erased (context only)", "accuracy": acc_erased},
        {"condition": "majority-class baseline", "accuracy": majority},
    ])
    metrics.update({"acc_tumour_erased": acc_erased, "acc_tumour_only": acc_only, "acc_majority": majority})
    log("shortcut test: " + " | ".join(f"{r.condition}: {r.accuracy:.3f}" for r in shortcut.itertuples()))

    out = run / "eval"
    out.mkdir(exist_ok=True)
    per_image = pd.DataFrame({
        "id": ids, "pid": pids, "true": [CLASSES[i] for i in y], "pred": [CLASSES[i] for i in pred],
        "confidence": probs.max(1), "dice": dice, "iou": inter / np.maximum(union, 1), "tumour_area": area,
        "cam_in_tumour": cam_in, "cam_pointing": pointing.astype(int),
        "pred_tumour_erased": [CLASSES[i] for i in p_erased.argmax(1)],
        **{f"p_{c}": probs[:, k] for k, c in enumerate(CLASSES)},
    })
    per_image.to_csv(out / "per_image.csv", index=False)
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    pd.DataFrame([{"true": CLASSES[i], "pred": CLASSES[j], "n": int(cm[i, j])}
                  for i in range(len(CLASSES)) for j in range(len(CLASSES))]).to_csv(out / "confusion.csv", index=False)
    rel.to_csv(out / "reliability.csv", index=False)
    shortcut.to_csv(out / "shortcut.csv", index=False)
    per_class = per_image.groupby("true").agg(slices=("dice", "size"), dice=("dice", "mean"),
                                               recall=("pred", lambda s: 0.0)).reset_index()
    per_class["recall"] = [float((per_image[per_image.true == c].pred == c).mean()) for c in per_class.true]
    per_class.to_csv(out / "per_class.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / "cams_sample.npz", cams=cams.astype(np.float16), seg=seg_np, ids=ids)
    log(f"wrote {out}")
    return metrics
