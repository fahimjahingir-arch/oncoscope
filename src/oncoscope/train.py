"""Joint training: BCE + Dice for the tumour mask, cross-entropy for tumour type. Model selection on validation patients."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from oncoscope import CLASSES
from oncoscope.data import SliceDataset, indices_for, load_cache, patient_split
from oncoscope.model import TumourNet, dice_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dice_per_image(prob: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    pred = (prob > thr).float()
    inter = (pred * target).sum((1, 2, 3))
    denom = pred.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return torch.where(denom > 0, 2 * inter / denom.clamp_min(1), torch.ones_like(denom))


@torch.no_grad()
def predict(model: TumourNet, ds: SliceDataset, batch_size: int = 32):
    model.eval()
    seg, cls, dice = [], [], []
    for img, mask, _ in DataLoader(ds, batch_size=batch_size):
        s, c = model(img)
        p = torch.sigmoid(s)
        seg.append((p > 0.5).to(torch.uint8))
        cls.append(F.softmax(c, 1))
        dice.append(dice_per_image(p, mask))
    return torch.cat(seg), torch.cat(cls).numpy(), torch.cat(dice).numpy()


def run_dir(cfg: dict) -> Path:
    d = Path("runs") / cfg["run_name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def train(cfg: dict, log=print) -> Path:
    tcfg = cfg["train"]
    seed_everything(tcfg["seed"])
    torch.set_num_threads(os.cpu_count() or 4)
    out = run_dir(cfg)

    data = load_cache(cfg["data"]["cache"])
    split = patient_split(data["pids"], data["labels"], cfg["split"]["val_frac"], cfg["split"]["test_frac"],
                          cfg["split"]["seed"])
    (out / "split.json").write_text(json.dumps(split, indent=1))
    idx = {k: indices_for(data["pids"], v) for k, v in split.items()}
    log(" | ".join(f"{k}: {len(split[k])} patients / {len(idx[k])} slices" for k in idx))

    def ds(k: str, augment: bool = False) -> SliceDataset:
        i = idx[k]
        return SliceDataset(data["images"][i], data["masks"][i], data["labels"][i], augment)

    train_ds, val_ds = ds("train", augment=True), ds("val")
    loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=tcfg["workers"])

    model = TumourNet(n_classes=len(CLASSES), **cfg["model"])
    enc_ids = {id(p) for p in model.encoder_parameters()}
    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if id(p) in enc_ids], "lr": tcfg["encoder_lr"]},
         {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": tcfg["lr"]}],
        weight_decay=tcfg["weight_decay"],
    )
    total = tcfg["epochs"] * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / len(loader)) * 0.5 * (1 + math.cos(math.pi * s / total)))
    counts = np.bincount(data["labels"][idx["train"]], minlength=len(CLASSES))
    cls_w = torch.tensor((counts.sum() / (len(CLASSES) * counts)) ** 0.5, dtype=torch.float32)
    log(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters | class weights {cls_w.numpy().round(2)}")

    history, best, stale = [], -1.0, 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, run_seg, run_cls, n = time.time(), 0.0, 0.0, 0
        for img, mask, y in loader:
            seg, logits = model(img)
            l_seg = F.binary_cross_entropy_with_logits(seg, mask) + dice_loss(seg, mask)
            l_cls = F.cross_entropy(logits, y, weight=cls_w)
            loss = tcfg["seg_weight"] * l_seg + tcfg["cls_weight"] * l_cls
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_seg += l_seg.item() * len(y)
            run_cls += l_cls.item() * len(y)
            n += len(y)

        _, probs, dice = predict(model, val_ds)
        y_val = val_ds.labels
        f1 = f1_score(y_val, probs.argmax(1), average="macro")
        score = 0.5 * (float(np.mean(dice)) + f1)
        row = {"epoch": epoch, "seg_loss": run_seg / n, "cls_loss": run_cls / n, "val_dice": float(np.mean(dice)),
               "val_accuracy": accuracy_score(y_val, probs.argmax(1)), "val_macro_f1": f1, "val_score": score,
               "seconds": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        improved = score > best
        log(f"epoch {epoch:2d}  seg {row['seg_loss']:.3f}  cls {row['cls_loss']:.3f}  val dice {row['val_dice']:.3f}  "
            f"acc {row['val_accuracy']:.3f}  F1 {f1:.3f}  ({row['seconds']:.0f}s)" + ("  *" if improved else ""))
        if improved:
            best, stale = score, 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, out / "best.pt")
        else:
            stale += 1
            if stale >= tcfg["patience"]:
                log(f"early stop after {stale} epochs without improvement")
                break
    return out
