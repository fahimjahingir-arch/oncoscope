"""MRI galleries for the README: segmentation + Grad-CAM per tumour type, and the tumour-erased shortcut test.

Usage: python scripts/gallery.py [run_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from oncoscope import CLASSES
from oncoscope.data import indices_for, load_cache
from oncoscope.evaluate import erase_tumour, load_model

BG, FG, MUTED = "#0E1117", "#E8EAF0", "#8B93A7"
TRUE_C, PRED_C = "#3ddc97", "#ff7a00"
CAM = LinearSegmentedColormap.from_list("ember", ["#00000000", "#7a0c0c", "#f77f00", "#ffd166", "#fff3c4"])
plt.rcParams.update({"font.family": "sans-serif", "text.color": FG, "axes.facecolor": BG, "figure.facecolor": BG})


def panel(ax, img, title="", sub="", sub_colour=MUTED):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=FG, fontsize=10, loc="left", pad=4, fontweight="bold")
    ax.text(0.02, 0.03, sub, transform=ax.transAxes, color=sub_colour, fontsize=8.5,
            bbox={"facecolor": BG, "alpha": 0.75, "edgecolor": "none", "pad": 2})


def main(run: Path) -> None:
    model, cfg = load_model(run / "best.pt")
    data = load_cache(cfg["data"]["cache"])
    split = json.loads((run / "split.json").read_text())
    idx = indices_for(data["pids"], split["test"])
    imgs, masks = data["images"][idx], data["masks"][idx]
    pi = pd.read_csv(run / "eval" / "per_image.csv")
    extra = np.load(run / "eval" / "cams_sample.npz")
    cams, segs = extra["cams"].astype(np.float32), extra["seg"]
    out = Path("figures")
    out.mkdir(exist_ok=True)

    # 1. two typical correct cases per tumour type: MRI with outlines, then Grad-CAM
    fig, axes = plt.subplots(3, 4, figsize=(13, 10.4))
    for r, cls in enumerate(CLASSES):
        ok = pi[(pi.true == cls) & (pi.pred == cls)].sort_values("dice")
        picks = [ok.index[int(len(ok) * q)] for q in (0.5, 0.8)]
        for k, i in enumerate(picks):
            a, b = axes[r, 2 * k], axes[r, 2 * k + 1]
            panel(a, imgs[i], cls.capitalize() if k == 0 else "",
                  f"Dice {pi.dice[i]:.2f}   expert ─  model ─")
            a.contour(masks[i], levels=[0.5], colors=TRUE_C, linewidths=1.4)
            a.contour(segs[i], levels=[0.5], colors=PRED_C, linewidths=1.4)
            panel(b, imgs[i], "", f"predicted {pi.pred[i]} ({pi.confidence[i]:.0%})", "#ffd166")
            b.imshow(cams[i], cmap=CAM, alpha=0.75, vmin=0, vmax=1)
            b.contour(masks[i], levels=[0.5], colors=TRUE_C, linewidths=0.9, linestyles="dotted")
    fig.suptitle("Held-out patients: tumour outline (green = expert, orange = model) and Grad-CAM attention",
                 color=FG, fontsize=13, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out / "gallery.png", dpi=150)
    plt.close(fig)

    # 2. shortcut test: same slice with the tumour erased
    fooled = pi[(pi.pred == pi.true) & (pi.pred_tumour_erased == pi.true)]
    picks = [fooled[fooled.true == c].index[0] for c in CLASSES if (fooled.true == c).any()][:3]
    fig, axes = plt.subplots(2, len(picks), figsize=(4.2 * len(picks), 8.6), squeeze=False)
    for k, i in enumerate(picks):
        panel(axes[0, k], imgs[i], f"{pi.true[i].capitalize()}: original", f"model says {pi.pred[i]}", "#ffd166")
        axes[0, k].contour(masks[i], levels=[0.5], colors=TRUE_C, linewidths=1.2)
        erased = erase_tumour(imgs[i].astype(np.float32), masks[i])
        panel(axes[1, k], erased, "tumour erased", f"model still says {pi.pred_tumour_erased[i]}", "#ff5d5d")
    fig.suptitle("Shortcut check: the classifier often names the tumour type after the tumour is removed",
                 color=FG, fontsize=12.5, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "shortcut_examples.png", dpi=150)
    plt.close(fig)
    print("wrote figures/gallery.png, figures/shortcut_examples.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/unet-resnet18"))
