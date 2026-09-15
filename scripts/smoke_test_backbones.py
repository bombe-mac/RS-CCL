"""Smoke test for VGG16 / MobileViT_S backbones: raw forward + through MoCo.

Run from the repo root: python scripts/smoke_test_backbones.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.models import VGG16, MobileViT_S
from src.moco.builder import MoCo

BATCH, NUM_CLASSES, RES = 10, 21, 256  # UCM pipeline uses 256x256 (src/utils.py)


def test_raw_backbone(cls, name):
    model = cls(name=name, num_classes=NUM_CLASSES)
    model.eval()
    x = torch.randn(BATCH, 3, RES, RES)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES), f"{name} raw forward: got {tuple(out.shape)}"
    print(f"[OK] {name} raw backbone -> {tuple(out.shape)}")


def test_through_moco(cls, name):
    # mirrors config.yaml's cl_config values for MoCo construction
    model = MoCo(cls, name, dim=2048, K=50, T=0.05, num_classes=NUM_CLASSES)
    model.train()
    im_q = torch.randn(BATCH, 3, RES, RES)
    im_k = torch.randn(BATCH, 3, RES, RES)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))
    features, target, logits = model(im_q=im_q, im_k=im_k, labels=labels)
    print(f"[OK] {name} via MoCo -> features {tuple(features.shape)}, "
          f"target {tuple(target.shape)}, logits {tuple(logits.shape)}")
    assert logits.shape == (2 * BATCH, NUM_CLASSES), f"{name} MoCo logits: got {tuple(logits.shape)}"


if __name__ == "__main__":
    for cls, name in [(VGG16, "VGG16"), (MobileViT_S, "MobileViT_S")]:
        test_raw_backbone(cls, name)
        test_through_moco(cls, name)
    print("All backbone smoke tests passed.")
