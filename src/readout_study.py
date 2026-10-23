"""Isolated, parameter-matched readout x LoRA study; no E0--E7 defaults change."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from torch import nn

from .model import DINOv2Encoder, DINOv2Segmenter, SegmentationDecoder

CELLS = {
    "F0": ("frozen", "last"), "F1": ("frozen", "quarter_concat"),
    "L0": ("lora", "last"), "L1": ("lora", "quarter_concat"),
}
SEEDS = (42, 2026)
LAYERS = (5, 11, 17, 23)
PROTOCOL = {
    "name": "readout_lora_v1", "seeds": list(SEEDS), "layers_zero_based": list(LAYERS),
    "batch_size": 4, "accumulation": 1, "workers": 0, "image_size": 448,
    "max_epochs": 50, "patience": 10, "lr_decoder": 1e-3, "lr_adapter": 1e-5,
    "weight_decay": 0.01, "betas": [0.9, 0.999], "epsilon": 1e-8,
    "rank": 4, "alpha": 4, "bce_weight": 0.5, "dice_weight": 0.5,
    "smooth": 1e-6, "amp": "float16", "loss_dtype": "float32", "threshold": 0.5,
    "augmentation": "none", "scheduler": "none", "feature_cache": False,
    "bootstrap_count": 20000, "bootstrap_seed": 20260926,
    "pilot_batches": 100, "max_consecutive_skips": 20,
    "inference": "native_volume_existing_evaluator", "patient_independence": "unverified",
}


@contextmanager
def cpu_rng(seed):
    """Initialize CPU modules independently, without consuming the sampler RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        yield


class SplitProjection(nn.Module):
    """Row partition of the original 1024->256 projection, not four full copies."""

    def __init__(self, original: nn.Conv2d):
        super().__init__()
        if original.out_channels != 256 or original.kernel_size != (1, 1):
            raise ValueError("Expected the original 1x1, 256-channel projection")
        self.branches = nn.ModuleList([
            nn.Conv2d(original.in_channels, 64, 1, bias=original.bias is not None)
            for _ in range(4)
        ])
        with torch.no_grad():
            for i, branch in enumerate(self.branches):
                branch.weight.copy_(original.weight[i * 64:(i + 1) * 64])
                if original.bias is not None:
                    branch.bias.copy_(original.bias[i * 64:(i + 1) * 64])

    def forward(self, features):
        if len(features) != 4:
            raise ValueError("Exactly four feature maps are required")
        return torch.cat([p(x) for p, x in zip(self.branches, features)], dim=1)


class ReadoutEncoder(DINOv2Encoder):
    def __init__(self, backbone, mode):
        nn.Module.__init__(self)
        if mode not in {"last", "quarter_concat"}:
            raise ValueError(mode)
        if len(backbone.blocks) != 24 or getattr(backbone, "chunked_blocks", False):
            raise ValueError("Study requires an unchunked 24-block backbone")
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim
        self.mode = mode
        self._trainable_block_indices = None
        self._lora_enabled = False
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def forward(self, images):
        indices = (23,) if self.mode == "last" else LAYERS
        context = nullcontext() if self.has_trainable_backbone_parameters() else torch.no_grad()
        with context:
            # DINO's API applies final norm once and removes CLS/register tokens.
            tokens = self.backbone.get_intermediate_layers(
                images, n=indices, reshape=False, return_class_token=False, norm=True,
            )
        if len(tokens) != len(indices):
            raise RuntimeError("Backbone returned the wrong number of layers")
        h, w = images.shape[-2] // 14, images.shape[-1] // 14
        if images.shape[-2] % 14 or images.shape[-1] % 14:
            raise ValueError("Input dimensions must be multiples of 14")
        for value in tokens:
            if value.shape != (images.shape[0], h * w, self.embed_dim):
                raise RuntimeError(f"Invalid patch-token shape: {tuple(value.shape)}")
        maps = tuple(t.transpose(1, 2).reshape(images.shape[0], self.embed_dim, h, w) for t in tokens)
        return maps * 4 if self.mode == "last" else maps


class ReadoutSegmenter(DINOv2Segmenter):
    def __init__(self, backbone, cell, seed, target_size=448):
        nn.Module.__init__(self)
        if cell not in CELLS or seed not in SEEDS:
            raise ValueError("Use F0/F1/L0/L1 and seeds 42/2026")
        adaptation, mode = CELLS[cell]
        self.encoder = ReadoutEncoder(backbone, mode)
        with cpu_rng(seed + 10000):
            self.decoder = SegmentationDecoder(self.encoder.embed_dim, target_size)
            self.decoder.reduce[0] = SplitProjection(self.decoder.reduce[0])
        if adaptation == "lora":
            with cpu_rng(seed + 20000):
                self.configure_lora(rank=4, alpha=4, targets=("qkv", "proj"))


def load_backbone(repo: Path, weights: Path):
    """Explicit local source/weights: never silently fetch a different hub revision."""
    if not (repo / "hubconf.py").is_file() or not weights.is_file():
        raise FileNotFoundError("Local DINO repository and pretrained weights required")
    backbone = torch.hub.load(str(repo), "dinov2_vitl14", source="local", pretrained=False)
    backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    if backbone.embed_dim != 1024 or len(backbone.blocks) != 24:
        raise ValueError("Expected DINOv2 ViT-Large/14")
    return backbone


class EarlyStop:
    def __init__(self, best=-1.0, stale=0, best_epoch=0):
        self.best, self.stale, self.best_epoch = best, stale, best_epoch

    def update(self, score, epoch):
        import math
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite validation Dice")
        improved = score > self.best
        if improved:
            self.best, self.stale, self.best_epoch = score, 0, epoch
        else:
            self.stale += 1
        return improved, self.stale >= PROTOCOL["patience"]
