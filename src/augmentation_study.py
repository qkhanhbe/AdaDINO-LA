"""Augmentation study: the common improved protocol (v2) with nnU-Net-style 2D data augmentation.

Pre-specified on 2026-11-10, before any augmented run, in response to a mock review: the
no-pretraining control and the DINOv2 models were trained without augmentation, whereas the
nnU-Net reference uses its own augmentation. This study is post hoc (the test set has been seen
by earlier analyses) and changes ONE factor relative to the label-efficiency study: stochastic
augmentation of the training slices. Everything else (v2 recipe, nested subsets, 2376 batches
per epoch, validation, early stopping, test set, evaluator) is unchanged.

Cells: LoRA (DINOv2 ViT-L/14, v2 recipe) and the randomly initialized ViT-L/14 with full
fine-tuning (encoder learning rate 1e-4), each with 108 and 11 training volumes and seeds 42,
2026 and 3407 (12 runs). Each run is evaluated on test once, after training completes.

The transforms follow the default nnU-Net 2D pipeline in kind, order and probability. nnU-Net
works on z-scored intensities; here intensities are in [0, 1], so additive noise is scaled by the
standard deviation of each slice, and multiplicative or gamma transforms are scale-free.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .label_efficiency_study import (BATCH_SIZE, CELLS as LE_CELLS, EXPECTED_TRAINABLE as LE_TRAINABLE,
                                     PROTOCOL as LE_PROTOCOL, SEEDS, STEPS_PER_EPOCH, encoder_lr)

CELLS = {"V_LO": LE_CELLS["V_LO"], "V_SC": LE_CELLS["V_SC"]}
EXPECTED_TRAINABLE = {cell: LE_TRAINABLE[cell] for cell in CELLS}
FRACTIONS = {10: 11, 100: 108}
RUNS = tuple((f, c, s) for c in CELLS for f in (100, 10) for s in SEEDS)
PILOT_FRACTION = 10
AUG_STREAM = 50000
MIN_FREE_VRAM_AT_START = 10 * 1024**3

AUGMENTATION = {
    "domain": "grayscale slice in [0, 1] before ImageNet normalization; mask resampled with nearest neighbour",
    "order": ["rotation_scaling", "gaussian_noise", "gaussian_blur", "brightness", "contrast",
              "low_resolution", "inverted_gamma", "gamma", "mirroring"],
    "rotation_scaling": {"p_rotation": 0.2, "rotation_degrees": [-180.0, 180.0],
                         "p_scaling": 0.2, "coordinate_scale": [0.7, 1.4],
                         "image": "bilinear", "mask": "nearest", "padding": "zeros"},
    "gaussian_noise": {"p": 0.1, "variance": [0.0, 0.1], "units": "variance relative to the slice variance"},
    "gaussian_blur": {"p": 0.2, "sigma_pixels": [0.5, 1.0]},
    "brightness": {"p": 0.15, "multiplier": [0.75, 1.25]},
    "contrast": {"p": 0.15, "factor": [0.75, 1.25], "preserve_range": True},
    "low_resolution": {"p": 0.25, "zoom": [0.5, 1.0], "down": "nearest", "up": "bicubic"},
    "inverted_gamma": {"p": 0.1, "gamma": [0.7, 1.5], "retain_stats": True},
    "gamma": {"p": 0.3, "gamma": [0.7, 1.5], "retain_stats": True},
    "mirroring": {"p_per_axis": 0.5, "axes": ["horizontal", "vertical"]},
    "final_clip": [0.0, 1.0],
}

PROTOCOL = {
    **LE_PROTOCOL,
    "name": "augmentation_study_v2",
    "status": "pre-specified 2026-11-10 before any augmented run; post hoc (test previously seen)",
    "cells": CELLS, "fractions_percent_to_volumes": FRACTIONS,
    "augmentation": AUGMENTATION, "augmentation_stream": "torch.Generator seeded seed + 50000 + epoch + 1000 * sweep",
    "runs": [list(run) for run in RUNS],
    "min_free_vram_at_start_gib": MIN_FREE_VRAM_AT_START / 1024**3,
}

_MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def _uniform(generator, low, high):
    return float(low + (high - low) * torch.rand((), generator=generator))


def _hit(generator, p):
    return bool(torch.rand((), generator=generator) < p)


def _gaussian_blur(x, sigma):
    radius = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), kernel.view(1, 1, -1, 1))


def _gamma(x, gamma, invert):
    if invert:
        x = 1 - x
    mean, std = x.mean(), x.std()
    lo, hi = x.min(), x.max()
    span = (hi - lo).clamp_min(1e-7)
    y = ((x - lo) / span) ** gamma * span + lo
    y = (y - y.mean()) / y.std().clamp_min(1e-7) * std + mean  # retain mean and std
    return 1 - y if invert else y


def augment_sample(gray, mask, generator, params=AUGMENTATION):
    """gray (1, 1, H, W) in [0, 1], mask (1, 1, H, W) in {0, 1}; returns augmented copies."""
    h, w = gray.shape[-2:]
    rs = params["rotation_scaling"]
    rotate, scale = _hit(generator, rs["p_rotation"]), _hit(generator, rs["p_scaling"])
    if rotate or scale:
        angle = math.radians(_uniform(generator, *rs["rotation_degrees"])) if rotate else 0.0
        factor = _uniform(generator, *rs["coordinate_scale"]) if scale else 1.0
        theta = torch.tensor([[math.cos(angle) * factor, -math.sin(angle) * factor, 0.0],
                              [math.sin(angle) * factor, math.cos(angle) * factor, 0.0]], dtype=gray.dtype, device=gray.device)
        grid = F.affine_grid(theta.unsqueeze(0), (1, 1, h, w), align_corners=False)
        gray = F.grid_sample(gray, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        mask = F.grid_sample(mask, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    if _hit(generator, params["gaussian_noise"]["p"]):
        variance = _uniform(generator, *params["gaussian_noise"]["variance"])
        noise = torch.randn(gray.shape, generator=generator, dtype=gray.dtype).to(gray.device)
        gray = gray + noise * math.sqrt(variance) * gray.std()
    if _hit(generator, params["gaussian_blur"]["p"]):
        gray = _gaussian_blur(gray, _uniform(generator, *params["gaussian_blur"]["sigma_pixels"]))
    if _hit(generator, params["brightness"]["p"]):
        gray = gray * _uniform(generator, *params["brightness"]["multiplier"])
    if _hit(generator, params["contrast"]["p"]):
        factor = _uniform(generator, *params["contrast"]["factor"])
        lo, hi, mean = gray.min(), gray.max(), gray.mean()
        gray = ((gray - mean) * factor + mean).clamp(lo, hi)
    if _hit(generator, params["low_resolution"]["p"]):
        zoom = _uniform(generator, *params["low_resolution"]["zoom"])
        small = F.interpolate(gray, size=(max(1, round(h * zoom)), max(1, round(w * zoom))), mode="nearest")
        gray = F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False)
    if _hit(generator, params["inverted_gamma"]["p"]):
        gray = _gamma(gray.clamp(0, 1), _uniform(generator, *params["inverted_gamma"]["gamma"]), invert=True)
    if _hit(generator, params["gamma"]["p"]):
        gray = _gamma(gray.clamp(0, 1), _uniform(generator, *params["gamma"]["gamma"]), invert=False)
    p_mirror = params["mirroring"]["p_per_axis"]
    if _hit(generator, p_mirror):
        gray, mask = gray.flip(-1), mask.flip(-1)
    if _hit(generator, p_mirror):
        gray, mask = gray.flip(-2), mask.flip(-2)
    return gray.clamp(*params["final_clip"]), (mask > 0.5).to(mask.dtype)


def augment_batch(images, masks, generator):
    """images (B, 3, H, W) ImageNet-normalized replicated grayscale; masks (B, 1, H, W).

    Tensors may live on the GPU; all random draws come from the CPU generator, so the draws do not
    depend on the device."""
    mean, std = _MEAN.to(images.device), _STD.to(images.device)
    gray = images[:, :1] * std[:, :1] + mean[:, :1]
    out_images, out_masks = [], []
    for i in range(images.shape[0]):
        g, m = augment_sample(gray[i:i + 1], masks[i:i + 1], generator)
        out_images.append(((g.expand(-1, 3, -1, -1) - mean) / std))
        out_masks.append(m)
    return torch.cat(out_images), torch.cat(out_masks)


__all__ = ["AUGMENTATION", "BATCH_SIZE", "CELLS", "EXPECTED_TRAINABLE", "FRACTIONS", "MIN_FREE_VRAM_AT_START",
           "PILOT_FRACTION", "PROTOCOL", "RUNS", "SEEDS", "STEPS_PER_EPOCH", "augment_batch", "encoder_lr"]
