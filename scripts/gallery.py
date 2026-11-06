"""MRI figures for the README in a journal style: segmentation + Grad-CAM per tumour type, and the shortcut test.

Usage: python scripts/gallery.py [run_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from oncoscope import CLASSES
from oncoscope.data import indices_for, load_cache
from oncoscope.evaluate import erase_tumour, load_model

EXPERT, MODEL = "#77AC30", "#D95319"  # MATLAB green / orange
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.linewidth": 0.6,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def show(ax, img):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def tag(ax, letter):
    ax.text(-0.04, 1.02, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")


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

    # 1. segmentation + Grad-CAM, two representative correct cases per class
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.6), gridspec_kw={"wspace": 0.04, "hspace": 0.12})
    letters = iter("abcdefghijkl")
    for r, cls in enumerate(CLASSES):
        ok = pi[(pi.true == cls) & (pi.pred == cls)].sort_values("dice")
        picks = [ok.index[int(len(ok) * q)] for q in (0.5, 0.8)]
        for k, i in enumerate(picks):
            a, b = axes[r, 2 * k], axes[r, 2 * k + 1]
            show(a, imgs[i])
            a.contour(masks[i], levels=[0.5], colors=EXPERT, linewidths=0.9)
            a.contour(segs[i], levels=[0.5], colors=MODEL, linewidths=0.9, linestyles="--")
            a.text(0.03, 0.04, f"Dice = {pi.dice[i]:.2f}", transform=a.transAxes, color="white", fontsize=7.5)
            show(b, imgs[i])
            b.imshow(cams[i], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            b.contour(masks[i], levels=[0.5], colors="white", linewidths=0.6)
            b.text(0.03, 0.04, f"p = {pi.confidence[i]:.2f}", transform=b.transAxes, color="white", fontsize=7.5)
            tag(a, next(letters))
        axes[r, 0].set_ylabel(cls.capitalize(), fontsize=9.5)
    for k, title in enumerate(["Segmentation", "Grad-CAM", "Segmentation", "Grad-CAM"]):
        axes[0, k].set_title(title, fontsize=9, pad=4)
    fig.legend(handles=[Line2D([], [], color=EXPERT, lw=1.2, label="Expert"),
                        Line2D([], [], color=MODEL, lw=1.2, ls="--", label="Model")],
               loc="lower center", ncol=2, frameon=True, edgecolor="black", fancybox=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.005))
    fig.savefig(out / "gallery.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. shortcut test examples: original vs tumour erased
    keep = pi[(pi.pred == pi.true) & (pi.pred_tumour_erased == pi.true)]
    picks = [keep[keep.true == c].index[0] for c in CLASSES if (keep.true == c).any()][:3]
    fig, axes = plt.subplots(2, len(picks), figsize=(2.4 * len(picks), 5.0), squeeze=False,
                             gridspec_kw={"wspace": 0.04, "hspace": 0.16})
    letters = iter("abcdef")
    for k, i in enumerate(picks):
        show(axes[0, k], imgs[i])
        axes[0, k].contour(masks[i], levels=[0.5], colors=EXPERT, linewidths=0.8)
        axes[0, k].set_title(pi.true[i].capitalize(), fontsize=9, pad=4)
        axes[0, k].text(0.03, 0.04, f"Predicted: {pi.pred[i]}", transform=axes[0, k].transAxes, color="white", fontsize=7.5)
        show(axes[1, k], erase_tumour(imgs[i].astype(np.float32), masks[i]))
        axes[1, k].text(0.03, 0.04, f"Predicted: {pi.pred_tumour_erased[i]}", transform=axes[1, k].transAxes,
                        color="white", fontsize=7.5)
        tag(axes[0, k], next(letters))
    axes[0, 0].set_ylabel("Original", fontsize=9.5)
    axes[1, 0].set_ylabel("Tumour erased", fontsize=9.5)
    fig.savefig(out / "shortcut_examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/gallery.png, figures/shortcut_examples.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/unet-resnet18"))
