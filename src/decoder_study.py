"""Decoder study: the v2 recipe with a UNETR-style hierarchical decoder instead of the lightweight one.

Pre-specified on 2026-11-01, before any decoder-study run; proposed after an internal mock review,
so it is post hoc with respect to the main and v2 results. Everything except the decoder follows
the label-efficiency runner: v2 recipe, same readout layers (5/11/17/23), same nested subsets
and batch order (100% reproduces the v2 order), same evaluator. The lightweight-decoder
counterparts are the archived v2 runs (100%) and the label-efficiency runs (10%).

Planned contrasts (paired over the 31 test volumes, bootstrap as in v2, labelled post hoc):
hierarchical minus lightweight decoder per strategy and fraction; gap of each hierarchical cell
to nnU-Net 3d_fullres; whether the strategy ranking at 100% changes.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .label_efficiency_study import FRACTIONS, PROTOCOL as LE_PROTOCOL, SEEDS
from .model import DINOv2Segmenter
from .protocol_v2_study import LR_ADAPTER, LR_PRETRAINED
from .readout_study import LAYERS, ReadoutEncoder, cpu_rng

CELLS = {"D_FR": "frozen", "D_PA": "partial", "D_LO": "lora", "D_FU": "full"}
LIGHT_COUNTERPART = {"D_FR": "V_FR", "D_PA": "V_PA", "D_LO": "V_LO", "D_FU": "V_FU"}
# Encoder-side trainable parameters of the v2 cells (v2 totals minus the 655,329-parameter decoder).
ENCODER_TRAINABLE = {"D_FR": 0, "D_PA": 50_393_088, "D_LO": 589_824, "D_FU": 304_368_640}
DECODER_PARAMS = 3_121_057
EXPECTED_TRAINABLE = {cell: n + DECODER_PARAMS for cell, n in ENCODER_TRAINABLE.items()}
# All four strategies at full data; the two strongest (LoRA, Full FT) also at 10%.
RUNS = tuple([(100, c, s) for s in SEEDS for c in CELLS] + [(10, c, s) for s in SEEDS for c in ("D_LO", "D_FU")])
PILOT_FRACTION = 100
# Decoder resolutions: 32x32 ViT tokens (patch 14 at 448 px) up to the 448 px input.
SIZES = (32, 64, 128, 224, 448)
PROTOCOL = {
    **LE_PROTOCOL,
    "name": "decoder_study_v2", "status": "pre-specified 2026-11-01 before any decoder-study run; post hoc "
                                           "with respect to the main and v2 results",
    "cells": CELLS, "light_counterpart": LIGHT_COUNTERPART,
    "decoder": {"type": "UNETR-style hierarchical decoder (2D)", "readout_layers_zero_based": list(LAYERS),
                "skip_resolutions": {"layer23": 32, "layer17": 64, "layer11": 128, "layer5": 224, "image_stem": 448},
                "upsampling": "bilinear resize + 3x3 conv + BN + ReLU", "fusion": "concatenate + two 3x3 conv blocks",
                "parameters": DECODER_PARAMS},
    "fractions_run": sorted({f for f, _, _ in RUNS}),
    "runs": [list(run) for run in RUNS],
    "contrasts": ["hierarchical - lightweight per strategy and fraction", "hierarchical - nnU-Net 3d_fullres",
                  "strategy ranking at 100%"],
}
del PROTOCOL["scratch"]


def encoder_lr(cell):
    return LR_ADAPTER if CELLS[cell] == "lora" else LR_PRETRAINED


def conv_bn_relu(in_ch, out_ch, kernel=3):
    return [nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]


class Up(nn.Module):
    """Bilinear resize to a fixed size, then 3x3 conv + BN + ReLU (448/32 = 14 is not a power of two)."""

    def __init__(self, in_ch, out_ch, size):
        super().__init__()
        self.size = size
        self.conv = nn.Sequential(*conv_bn_relu(in_ch, out_ch))

    def forward(self, x):
        return self.conv(F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False))


class HierarchicalDecoder(nn.Module):
    """UNETR-style decoder: deeper readout layers decode coarser stages, shallower ones finer stages.

    Each readout map is projected to 256 channels; layer 23 starts the top-down path at 32 px and
    layers 17/11/5 join it as skips at 64/128/224 px after their own upsampling chains (UNETR's
    deconvolution branches). A convolutional stem on the input image supplies the 448 px skip.
    """

    def __init__(self, embed_dim, target_size=448):
        super().__init__()
        if target_size != SIZES[-1]:
            raise ValueError(f"Decoder is built for {SIZES[-1]} px inputs")
        _, s64, s128, s224, s448 = SIZES
        self.proj = nn.ModuleList([nn.Sequential(*conv_bn_relu(embed_dim, 256, kernel=1)) for _ in LAYERS])
        self.skip17 = Up(256, 128, s64)
        self.skip11 = nn.Sequential(Up(256, 128, s64), Up(128, 64, s128))
        self.skip5 = nn.Sequential(Up(256, 128, s64), Up(128, 64, s128), Up(64, 32, s224))
        self.stem = nn.Sequential(*conv_bn_relu(3, 32), *conv_bn_relu(32, 32))
        self.up64, self.fuse64 = Up(256, 128, s64), nn.Sequential(*conv_bn_relu(256, 128), *conv_bn_relu(128, 128))
        self.up128, self.fuse128 = Up(128, 64, s128), nn.Sequential(*conv_bn_relu(128, 64), *conv_bn_relu(64, 64))
        self.up224, self.fuse224 = Up(64, 32, s224), nn.Sequential(*conv_bn_relu(64, 32), *conv_bn_relu(32, 32))
        self.up448, self.fuse448 = Up(32, 32, s448), nn.Sequential(*conv_bn_relu(64, 32), *conv_bn_relu(32, 32))
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, features, images):
        if len(features) != len(LAYERS):
            raise ValueError("Exactly four readout maps are required")
        z5, z11, z17, z23 = (proj(f) for proj, f in zip(self.proj, features))
        x = self.fuse64(torch.cat([self.up64(z23), self.skip17(z17)], dim=1))
        x = self.fuse128(torch.cat([self.up128(x), self.skip11(z11)], dim=1))
        x = self.fuse224(torch.cat([self.up224(x), self.skip5(z5)], dim=1))
        x = self.fuse448(torch.cat([self.up448(x), self.stem(images)], dim=1))
        return self.head(x)


class DecoderStudySegmenter(DINOv2Segmenter):
    def __init__(self, backbone, cell, seed, target_size=448):
        nn.Module.__init__(self)
        if cell not in CELLS or seed not in SEEDS:
            raise ValueError(f"Use {tuple(CELLS)} and seeds {SEEDS}")
        self.encoder = ReadoutEncoder(backbone, "quarter_concat")
        with cpu_rng(seed + 10000):
            self.decoder = HierarchicalDecoder(self.encoder.embed_dim, target_size)
        adaptation = CELLS[cell]
        # Same adapter initialization stream as v2, so LoRA starts from identical adapters.
        if adaptation == "lora":
            with cpu_rng(seed + 20000):
                self.configure_lora(rank=4, alpha=4, targets=("qkv", "proj"))
        elif adaptation == "partial":
            self.configure_partial_finetuning(num_blocks=4)
        elif adaptation == "full":
            self.configure_full_finetuning()

    def forward(self, x):
        return self.decoder(self.encoder(x), x)


assert set(FRACTIONS) >= {f for f, _, _ in RUNS}
