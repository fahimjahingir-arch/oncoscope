"""U-Net with an ImageNet ResNet-18 encoder and a tumour-type head on the bottleneck."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Up(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in + c_skip, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.conv(x)


class TumourNet(nn.Module):
    def __init__(self, n_classes: int = 3, pretrained: bool = True, dropout: float = 0.3, **_):
        super().__init__()
        enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # 1/2, 64
        self.pool = enc.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.up4 = Up(512, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up1 = Up(64, 64, 32)
        self.up0 = Up(32, 0, 16)
        self.seg_head = nn.Conv2d(16, 1, 1)
        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(512, n_classes))
        self.feature_hook: torch.Tensor | None = None  # last encoder map, kept for Grad-CAM

    def encoder_parameters(self):
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from m.parameters()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (x.repeat(1, 3, 1, 1) - IMAGENET_MEAN.to(x)) / IMAGENET_STD.to(x)
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        self.feature_hook = s4
        d = self.up4(s4, s3)
        d = self.up3(d, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        d = self.up0(d, None)
        return self.seg_head(d), self.cls_head(s4)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


def grad_cam(model: TumourNet, x: torch.Tensor, target: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Grad-CAM on the encoder's last block for the predicted (or given) class -> (B, H, W) in [0, 1]."""
    model.eval()
    x = x.requires_grad_(False)
    _, logits = model(x)
    feats = model.feature_hook
    cls = logits.argmax(1) if target is None else target
    score = logits.gather(1, cls[:, None]).sum()
    grads = torch.autograd.grad(score, feats)[0]
    cam = F.relu((grads.mean((2, 3), keepdim=True) * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / cam.flatten(1).amax(1).clamp_min(1e-8)[:, None, None]
    return cam.detach(), logits.detach()
