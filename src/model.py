"""DINOv2 encoder and the basic decoder shared by experiments E0--E6.

Encoder: DINOv2 ViT (frozen by default; adaptation is configured by the trainer)
Decoder: Conv2d + upsampling layers that restore the input resolution

Theo bài báo:
  - DINOv2 splits a 448×448 image into 32×32 = 1024 patches (patch_size=14)
  - Encoder output: (B, 1024, embed_dim) → reshape → (B, embed_dim, 32, 32)
  - The decoder upsamples 32×32 → 448×448 with conv + upsample layers
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Low-rank adapter around a frozen linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 4.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):
        base_output = self.base(inputs)
        update = F.linear(F.linear(inputs, self.lora_A), self.lora_B)
        return base_output + self.scaling * update


class DINOv2Encoder(nn.Module):
    """
    DINOv2 encoder (frozen by default).
    Loads the pretrained model from torch.hub and freezes its weights.
    """

    # Model name → (hub name, embed_dim)
    MODEL_CONFIGS = {
        "vit_small": ("dinov2_vits14", 384),
        "vit_base": ("dinov2_vitb14", 768),
        "vit_large": ("dinov2_vitl14", 1024),
        "vit_giant": ("dinov2_vitg14", 1536),
    }

    def __init__(self, model_name="vit_large", pretrained=True):
        super().__init__()

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Model '{model_name}' is not supported. "
                f"Choose from: {list(self.MODEL_CONFIGS.keys())}"
            )

        hub_name, self.embed_dim = self.MODEL_CONFIGS[model_name]

        # Load DINOv2 from facebookresearch
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            hub_name,
            # pretrained=False builds the same architecture with DINOv2's own random
            # initialisation (no-pretraining control, 19/10 revision baseline)
            pretrained=pretrained,
        )

        # Freeze all encoder weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        # None means the whole backbone follows the regular train/eval mode
        # (E2).  E3 stores the exact block indices here so frozen blocks stay
        # in eval mode while only the selected tail blocks enter train mode.
        self._trainable_block_indices = None
        self._lora_enabled = False
        self.backbone.eval()

    def has_trainable_backbone_parameters(self):
        """Whether full fine-tuning (E2) requires autograd through the backbone."""
        return any(param.requires_grad for param in self.backbone.parameters())

    @property
    def trainable_block_indices(self):
        """Tuple of E3 trainable block indices, or None outside partial FT."""
        return self._trainable_block_indices

    def unfreeze_last_blocks(self, num_blocks=4):
        """Freeze the backbone, then unfreeze exactly its last ``num_blocks``.

        Patch embedding, positional/register tokens, the final encoder norm and
        all earlier Transformer blocks remain frozen, matching experiment E3.
        """
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise AttributeError("DINOv2 backbone does not expose a 'blocks' sequence.")

        total_blocks = len(blocks)
        if not 1 <= num_blocks <= total_blocks:
            raise ValueError(
                f"num_blocks must be in [1, {total_blocks}], got {num_blocks}."
            )

        for param in self.backbone.parameters():
            param.requires_grad = False

        first_index = total_blocks - num_blocks
        indices = tuple(range(first_index, total_blocks))
        for index in indices:
            for param in blocks[index].parameters():
                param.requires_grad = True

        self._trainable_block_indices = indices
        return indices

    def train(self, mode=True):
        """Apply the correct backbone mode for frozen, full and partial FT."""
        super().train(mode)
        if not mode or not self.has_trainable_backbone_parameters():
            self.backbone.eval()
        elif getattr(self, "_lora_enabled", False):
            # LoRA does not require train mode in the frozen base backbone.
            # Autograd remains enabled and reaches lora_A/lora_B in eval mode.
            self.backbone.eval()
        elif self._trainable_block_indices is not None:
            # Avoid stochastic depth/dropout in the frozen prefix.  Calling
            # train() only on the four E3 blocks activates regular training
            # behavior exactly where gradients are enabled.
            self.backbone.eval()
            for index in self._trainable_block_indices:
                self.backbone.blocks[index].train(True)
        else:
            self.backbone.train(True)
        return self

    def enable_lora(self, rank=4, alpha=4.0, targets=("qkv", "proj")):
        """Inject LoRA into every attention qkv/proj layer in the backbone."""
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise AttributeError("DINOv2 backbone does not expose a 'blocks' sequence.")

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        replaced = []
        for block_index, block in enumerate(blocks):
            attention = getattr(block, "attn", None)
            if attention is None:
                raise AttributeError(f"Transformer block {block_index} has no attention module.")
            for target in targets:
                layer = getattr(attention, target, None)
                if isinstance(layer, LoRALinear):
                    raise RuntimeError(
                        f"LoRA was already injected into block {block_index} attention.{target}."
                    )
                if not isinstance(layer, nn.Linear):
                    raise TypeError(
                        f"Expected nn.Linear at block {block_index} attention.{target}, "
                        f"got {type(layer).__name__}."
                    )
                setattr(attention, target, LoRALinear(layer, rank=rank, alpha=alpha))
                replaced.append(f"blocks.{block_index}.attn.{target}")

        expected = len(blocks) * len(targets)
        if len(replaced) != expected:
            raise RuntimeError(f"Injected {len(replaced)} LoRA layers; expected {expected}.")
        self._trainable_block_indices = None
        self._lora_enabled = True
        self.backbone.eval()
        return tuple(replaced)

    def forward(self, x):
        """
        Args:
            x: tensor (B, 3, 448, 448)
        Returns:
            features: tensor (B, embed_dim, 32, 32)
        """
        # Frozen E0/E1 can skip autograd.  E2 must not: otherwise
        # their encoder parameters receive no gradient and the experiment is invalid.
        if self.has_trainable_backbone_parameters():
            output = self.backbone.forward_features(x)
        else:
            with torch.no_grad():
                output = self.backbone.forward_features(x)

        # Patch tokens (class token removed)
        # output["x_norm_patchtokens"]: (B, num_patches, embed_dim)
        patch_tokens = output["x_norm_patchtokens"]

        B, N, D = patch_tokens.shape
        # N = 32*32 = 1024 patches for a 448×448 image
        h, w = x.shape[-2] // 14, x.shape[-1] // 14
        if N != h * w:
            raise RuntimeError(
                f"Expected {h * w} patch tokens for input {tuple(x.shape)}, got {N}."
            )

        # Reshape: (B, N, D) → (B, D, h, w)
        features = patch_tokens.permute(0, 2, 1).reshape(B, D, h, w)

        return features


class SegmentationDecoder(nn.Module):
    """
    Decoder that upsamples the 32×32 feature map to 448×448.

    Architecture:
      1. 1×1 conv to reduce channels (LinearClassifier)
      2. A sequence of UpBlocks: Upsample + Conv + BN + ReLU
      3. Final 1×1 conv to one output channel (binary segmentation)
    """

    def __init__(self, in_channels, target_size=448):
        super().__init__()
        self.target_size = target_size

        # 1×1 conv to reduce feature dimension
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Upsampling path: 32 → 64 → 128 → 256 → 448
        self.up1 = self._up_block(256, 128)   # 32  → 64
        self.up2 = self._up_block(128, 64)    # 64  → 128
        self.up3 = self._up_block(64, 32)     # 128 → 256

        # Final layer: 1×1 conv to one channel
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    @staticmethod
    def _up_block(in_ch, out_ch):
        """Upsample ×2 + Conv + BN + ReLU block."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: tensor (B, embed_dim, 32, 32)
        Returns:
            logits: tensor (B, 1, 448, 448)
        """
        x = self.reduce(x)   # (B, 256, 32, 32)
        x = self.up1(x)      # (B, 128, 64, 64)
        x = self.up2(x)      # (B, 64, 128, 128)
        x = self.up3(x)      # (B, 32, 256, 256)

        # Interpolate to exactly 448×448
        x = F.interpolate(x, size=(self.target_size, self.target_size),
                          mode="bilinear", align_corners=False)

        logits = self.head(x)  # (B, 1, 448, 448)
        return logits


class DINOv2Segmenter(nn.Module):
    """
    Full model: DINOv2 encoder + trainable decoder.
    """

    def __init__(self, model_name="vit_large", pretrained=True):
        super().__init__()
        self.encoder = DINOv2Encoder(model_name=model_name, pretrained=pretrained)
        self.decoder = SegmentationDecoder(
            in_channels=self.encoder.embed_dim,
            target_size=448,
        )

    def forward(self, x):
        """
        Args:
            x: tensor (B, 3, 448, 448)
        Returns:
            logits: tensor (B, 1, 448, 448) — before sigmoid
        """
        features = self.encoder(x)  # (B, embed_dim, 32, 32)
        logits = self.decoder(features)  # (B, 1, 448, 448)
        return logits

    def get_trainable_params(self):
        """Return every parameter enabled by the selected experiment."""
        return (param for param in self.parameters() if param.requires_grad)

    def configure_partial_finetuning(self, num_blocks=4):
        """Configure E3 and return the trainable Transformer block indices."""
        for param in self.decoder.parameters():
            param.requires_grad = True
        return self.encoder.unfreeze_last_blocks(num_blocks=num_blocks)

    def configure_full_finetuning(self):
        """Configure E2 so every encoder and decoder parameter is trainable."""
        self.encoder.backbone.requires_grad_(True)
        self.decoder.requires_grad_(True)
        self.encoder._trainable_block_indices = None
        self.encoder._lora_enabled = False

    def configure_lora(self, rank=4, alpha=4.0, targets=("qkv", "proj")):
        """Configure LoRA plus the shared decoder for E4/E6."""
        for parameter in self.decoder.parameters():
            parameter.requires_grad = True
        return self.encoder.enable_lora(rank=rank, alpha=alpha, targets=targets)

    def lora_named_parameters(self):
        return [
            (name, parameter)
            for name, parameter in self.encoder.backbone.named_parameters()
            if name.endswith("lora_A") or name.endswith("lora_B")
        ]

    def num_trainable_params(self):
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self):
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("Checking model architecture...")

    model = DINOv2Segmenter(model_name="vit_large")
    print(f"\nEncoder: DINOv2 ViT-Large (embed_dim={model.encoder.embed_dim})")
    print(f"Total parameters:      {model.num_total_params():>12,}")
    print(f"Trainable parameters:  {model.num_trainable_params():>12,}")

    dummy = torch.randn(2, 3, 448, 448)
    with torch.no_grad():
        output = model(dummy)
    print(f"\nInput  shape: {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Forward pass succeeded.")
