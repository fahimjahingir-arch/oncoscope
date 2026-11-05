"""`oncoscope prepare | train | evaluate | predict`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
import yaml
from rich.console import Console
from tqdm import tqdm

from oncoscope import CLASSES

app = typer.Typer(add_completion=False, help="Brain-tumour MRI segmentation + classification.")
console = Console()


def _cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@app.command()
def prepare(config: Path = typer.Option(Path("configs/default.yaml"))):
    """Read the figshare .mat files and cache images + masks at a fixed size."""
    from oncoscope.data import build_cache

    cfg = _cfg(config)["data"]
    d = build_cache(cfg["mat_dir"], cfg["cache"], cfg["size"], progress=lambda it: tqdm(list(it), desc="slices"))
    counts = dict(zip(CLASSES, np.bincount(d["labels"]).tolist(), strict=True))
    console.print(f"[green]{len(d['labels'])} slices from {len(set(d['pids']))} patients[/green] {counts}")


@app.command()
def train(config: Path = typer.Option(Path("configs/default.yaml"))):
    """Train the joint segmentation + classification network."""
    from oncoscope.train import train as run

    out = run(_cfg(config), log=console.print)
    console.print(f"[green]best checkpoint: {out / 'best.pt'}[/green]")


@app.command()
def evaluate(run: Path = typer.Argument(Path("runs/unet-resnet18"))):
    """Evaluate on held-out patients, including Grad-CAM and shortcut tests."""
    from oncoscope.evaluate import evaluate as run_eval

    run_eval(run, log=console.print)


@app.command()
def predict(
    image: Path = typer.Argument(..., help="A .mat slice from the dataset, or a PNG/JPG MRI slice"),
    checkpoint: Path = typer.Option(Path("runs/unet-resnet18/best.pt")),
    out: Path = typer.Option(Path("prediction.png")),
):
    """Segment and classify one MRI slice; writes an overlay image."""
    import matplotlib.pyplot as plt
    import torch

    from oncoscope.data import normalise, read_mat, resize_pair
    from oncoscope.evaluate import load_model

    model, cfg = load_model(checkpoint)
    size = cfg["data"]["size"]
    if image.suffix == ".mat":
        raw = read_mat(image)["image"]
    else:
        raw = plt.imread(image).astype(np.float32)
        raw = raw.mean(-1) if raw.ndim == 3 else raw
    img, _ = resize_pair(normalise(raw), np.zeros_like(raw, dtype=np.uint8), size)
    with torch.no_grad():
        seg, logits = model(torch.from_numpy(img)[None, None])
    probs = torch.softmax(logits, 1)[0].numpy()
    mask = torch.sigmoid(seg)[0, 0].numpy() > 0.5
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
    ax.imshow(img, cmap="gray")
    ax.contour(mask, levels=[0.5], colors="#ff7a00", linewidths=1.5)
    ax.set_title(f"{CLASSES[probs.argmax()]}  ({probs.max():.0%})", color="white")
    ax.axis("off")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    console.print({c: round(float(p), 3) for c, p in zip(CLASSES, probs, strict=True)}, f"-> {out}")


if __name__ == "__main__":
    app()
