"""Common improved protocol (v2): one recipe applied to all four ViT-Large strategies.

Post hoc (defined after the main test results were known): four-layer readout, gradient-norm
clipping at 1.0, and a rule-based learning rate in which pretrained weights keep the main-protocol
rate while newly initialized parameters use a higher one (decoder 1e-3, zero-initialized LoRA
adapters 1e-4). The main protocol (v1), E0--E7 and the readout study are not changed.
"""
from __future__ import annotations

from torch import nn

from .model import DINOv2Segmenter, SegmentationDecoder
from .readout_study import LAYERS, ReadoutEncoder, SplitProjection, cpu_rng

CELLS = {"V_FR": "frozen", "V_PA": "partial", "V_LO": "lora", "V_FU": "full"}
SEEDS = (42, 2026, 3407)
READOUT = "quarter_concat"
LR_PRETRAINED, LR_ADAPTER, LR_DECODER = 1e-5, 1e-4, 1e-3
GRAD_CLIP_NORM = 1.0
# Same counts as the main-protocol cells: the split projection keeps the decoder at 655,329.
EXPECTED_TRAINABLE = {"V_FR": 655_329, "V_PA": 51_048_417, "V_LO": 1_245_153, "V_FU": 305_023_969}
PROTOCOL = {
    "name": "common_improved_protocol_v2", "status": "post hoc; test previously seen",
    "backbone": "dinov2_vitl14", "cells": CELLS, "seeds": list(SEEDS),
    "readout": READOUT, "layers_zero_based": list(LAYERS),
    "batch_size": 4, "accumulation": 1, "workers": 0, "image_size": 448,
    "max_epochs": 50, "patience": 10,
    "lr_rule": {"pretrained_weights": LR_PRETRAINED, "lora_adapters": LR_ADAPTER, "decoder": LR_DECODER},
    "grad_clip_norm": GRAD_CLIP_NORM, "weight_decay": 0.01, "betas": [0.9, 0.999], "epsilon": 1e-8,
    "partial_blocks": [20, 21, 22, 23], "rank": 4, "alpha": 4, "bce_weight": 0.5, "dice_weight": 0.5,
    "smooth": 1e-6, "amp": "float16", "loss_dtype": "float32", "threshold": 0.5,
    "augmentation": "none", "scheduler": "none",
    "bootstrap_count": 20000, "bootstrap_seed": 20260926,
    "pilot_batches": 100, "max_consecutive_skips": 20,
    "inference": "native_volume_existing_evaluator", "patient_independence": "unverified",
}


def encoder_lr(cell):
    return LR_ADAPTER if CELLS[cell] == "lora" else LR_PRETRAINED


class ProtocolV2Segmenter(DINOv2Segmenter):
    def __init__(self, backbone, cell, seed, target_size=448):
        nn.Module.__init__(self)
        if cell not in CELLS or seed not in SEEDS:
            raise ValueError(f"Use {tuple(CELLS)} and seeds {SEEDS}")
        self.encoder = ReadoutEncoder(backbone, READOUT)
        # Same decoder and adapter initialization streams as the readout study.
        with cpu_rng(seed + 10000):
            self.decoder = SegmentationDecoder(self.encoder.embed_dim, target_size)
            self.decoder.reduce[0] = SplitProjection(self.decoder.reduce[0])
        adaptation = CELLS[cell]
        if adaptation == "lora":
            with cpu_rng(seed + 20000):
                self.configure_lora(rank=4, alpha=4, targets=("qkv", "proj"))
        elif adaptation == "partial":
            self.configure_partial_finetuning(num_blocks=4)
        elif adaptation == "full":
            self.configure_full_finetuning()
