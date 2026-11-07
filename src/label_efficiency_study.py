"""Label-efficiency study: the common improved protocol (v2) trained on nested subsets of volumes.

Pre-specified on 2026-10-27, before any reduced-data run. The v2 recipe is unchanged. Each
training epoch draws the same number of batches as one full-data epoch (2376) by cycling
through the subset, so every fraction has the same update budget, validation schedule and
early-stopping rule. A fifth cell trains the same ViT-Large from DINOv2's random
initialization (no pretraining) with full fine-tuning.
"""
from __future__ import annotations

import numpy as np
import torch

from .protocol_v2_study import (CELLS as V2_CELLS, EXPECTED_TRAINABLE as V2_TRAINABLE, LR_ADAPTER,
                                LR_PRETRAINED, PROTOCOL as V2_PROTOCOL, SEEDS)
from .readout_study import cpu_rng

CELLS = {**V2_CELLS, "V_SC": "scratch"}
EXPECTED_TRAINABLE = {**V2_TRAINABLE, "V_SC": V2_TRAINABLE["V_FU"]}
# Same rule as v2: newly initialized parameters get the higher rate. 1e-4 with clipping at 1.0
# is also the tuned no-pretraining recipe already reported in the paper.
LR_SCRATCH = 1e-4
SLICES_PER_VOLUME = 88
BATCH_SIZE = 4
STEPS_PER_EPOCH = 108 * SLICES_PER_VOLUME // BATCH_SIZE
FRACTIONS = {10: 11, 25: 27, 50: 54, 100: 108}
SUBSET_STREAM = 20261027
# Reduced fractions for every cell; full data only where no v2 run exists (scratch), plus one
# parity run that should reproduce the archived v2 LoRA seed-42 result.
RUNS = tuple(
    [(f, c, s) for f in (10, 25, 50) for c in CELLS for s in SEEDS]
    + [(100, "V_SC", s) for s in SEEDS]
    + [(100, "V_LO", 42)]
)
PILOT_FRACTION = 10
PROTOCOL = {
    **V2_PROTOCOL,
    "name": "label_efficiency_v2", "status": "pre-specified 2026-10-27 before any reduced-data run",
    "cells": CELLS, "fractions_percent_to_volumes": FRACTIONS,
    "subset_rule": "per seed, one permutation of the sorted 108 training volumes (numpy PCG64, "
                   "seed [20261027, training seed]); each fraction takes a prefix, so subsets are nested",
    "steps_per_epoch": STEPS_PER_EPOCH,
    "epoch_rule": "cycle through the subset (new shuffle per sweep) until 2376 batches; "
                  "validation, checkpoint selection and patience as in v2",
    "scratch": {"init": "DINOv2 ViT-L/14 hub architecture with its own random initialization",
                "adaptation": "full", "encoder_lr": LR_SCRATCH},
    "validation": "fixed 15 volumes", "test": "fixed 31 volumes, evaluated once per completed run",
    "runs": [list(run) for run in RUNS],
}


def nested_subsets(train_ids, seed):
    """Nested volume subsets for one seed: {fraction_percent: sorted case IDs}."""
    ordered = sorted(train_ids)
    if len(ordered) != 108 or len(set(ordered)) != 108:
        raise ValueError("Expected the 108 unique training volumes")
    order = np.random.default_rng([SUBSET_STREAM, seed]).permutation(len(ordered))
    return {f: sorted(ordered[i] for i in order[:n]) for f, n in FRACTIONS.items()}


def all_subsets(train_ids):
    return {str(seed): {str(f): ids for f, ids in nested_subsets(train_ids, seed).items()} for seed in SEEDS}


def encoder_lr(cell):
    if cell == "V_SC":
        return LR_SCRATCH
    return LR_ADAPTER if CELLS[cell] == "lora" else LR_PRETRAINED


def random_backbone(repo, seed):
    """Same architecture as the pretrained encoder, DINOv2's own initialization, no weights loaded."""
    with cpu_rng(seed + 60000):
        backbone = torch.hub.load(str(repo), "dinov2_vitl14", source="local", pretrained=False)
    if backbone.embed_dim != 1024 or len(backbone.blocks) != 24:
        raise ValueError("Expected DINOv2 ViT-Large/14")
    return backbone
