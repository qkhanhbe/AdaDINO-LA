import unittest

import numpy as np
import torch
import torch.nn as nn

from src.model import DINOv2Encoder, DINOv2Segmenter, LoRALinear
from src.metrics import DiceBCELoss
from src.prepare_utah_run2 import normalize_volume
from src.train_e5_e6 import (
    EXPERIMENTS,
    _audit_successful_gradients,
    load_trainable_state_dict,
    trainable_state_dict,
)
from src.volume_metrics import hd95_mm


class FakeAttention(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)

    def forward(self, inputs):
        return self.proj(self.qkv(inputs)[..., : inputs.shape[-1]])


class FakeBlock(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.attn = FakeAttention(width)

    def forward(self, inputs):
        return inputs + self.attn(inputs)


class FakeBackbone(nn.Module):
    def __init__(self, depth=3, width=4):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(width) for _ in range(depth)])

    def forward_features(self, inputs):
        tokens = inputs.flatten(2).transpose(1, 2)[..., :4]
        for block in self.blocks:
            tokens = block(tokens)
        return {"x_norm_patchtokens": tokens}


def fake_segmenter():
    encoder = DINOv2Encoder.__new__(DINOv2Encoder)
    nn.Module.__init__(encoder)
    encoder.backbone = FakeBackbone()
    encoder.embed_dim = 4
    encoder._trainable_block_indices = None
    encoder._lora_enabled = False
    encoder.backbone.requires_grad_(False)

    model = DINOv2Segmenter.__new__(DINOv2Segmenter)
    nn.Module.__init__(model)
    model.encoder = encoder
    model.decoder = nn.Linear(4, 1)
    return model


class E5E6Tests(unittest.TestCase):
    def test_e0_registry_matches_protocol(self):
        self.assertEqual(
            EXPERIMENTS["e0"],
            {
                "method": "E0_Large_Frozen_BCE",
                "model": "vit_large",
                "adaptation": "frozen",
                "loss": "bce",
            },
        )

    def test_e1_registry_matches_protocol(self):
        self.assertEqual(
            EXPERIMENTS["e1"],
            {
                "method": "E1_Large_Frozen_BCEDice",
                "model": "vit_large",
                "adaptation": "frozen",
                "loss": "bce_dice",
            },
        )

    def test_e2_registry_matches_protocol(self):
        self.assertEqual(
            EXPERIMENTS["e2"],
            {
                "method": "E2_Large_Full_FT_BCEDice",
                "model": "vit_large",
                "adaptation": "full",
                "loss": "bce_dice",
            },
        )

    def test_e3_registry_matches_protocol(self):
        self.assertEqual(
            EXPERIMENTS["e3"],
            {
                "method": "E3_Large_Partial_FT_BCEDice",
                "model": "vit_large",
                "adaptation": "partial",
                "loss": "bce_dice",
            },
        )

    def test_e4_registry_matches_protocol(self):
        self.assertEqual(
            EXPERIMENTS["e4"],
            {
                "method": "E4_Giant_LoRA",
                "model": "vit_giant",
                "adaptation": "lora",
                "loss": "bce_dice",
            },
        )

    def test_dice_bce_loss_reductions_are_float32_and_finite(self):
        logits = torch.full((1, 1, 448, 448), 10.0, dtype=torch.float16)
        targets = torch.ones_like(logits)
        loss = DiceBCELoss(0.5, 0.5)(logits, targets)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))

    def test_full_finetuning_enables_entire_model(self):
        model = fake_segmenter()
        model.configure_full_finetuning()
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        model.train()
        self.assertTrue(model.encoder.backbone.training)

    def test_partial_finetuning_opens_only_last_four_blocks(self):
        model = fake_segmenter()
        model.encoder.backbone = FakeBackbone(depth=24)
        model.encoder.backbone.requires_grad_(False)

        indices = model.configure_partial_finetuning(num_blocks=4)

        self.assertEqual(indices, (20, 21, 22, 23))
        self.assertTrue(all(parameter.requires_grad for parameter in model.decoder.parameters()))
        for index, block in enumerate(model.encoder.backbone.blocks):
            expected = index in indices
            self.assertTrue(
                all(parameter.requires_grad is expected for parameter in block.parameters())
            )

        model.train()
        self.assertTrue(all(not block.training for block in model.encoder.backbone.blocks[:20]))
        self.assertTrue(all(block.training for block in model.encoder.backbone.blocks[20:]))

    def test_full_checkpoint_includes_encoder_buffers(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
                self.decoder = nn.Linear(2, 1)

        model = ToyModel()
        saved = trainable_state_dict(model)
        self.assertIn("encoder.1.running_mean", saved)
        self.assertEqual(set(saved), set(model.state_dict()))

    def test_gradient_audit_allows_unused_dinov2_mask_token(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mask_token = nn.Parameter(torch.zeros(1))
                self.weight = nn.Parameter(torch.ones(1))

        model = ToyModel()
        model.weight.grad = torch.ones_like(model.weight)
        _audit_successful_gradients(model)

    def test_compact_checkpoint_restores_decoder_batchnorm_buffers(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(2, 2)
                self.encoder.requires_grad_(False)
                self.decoder = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))

        model = ToyModel()
        model.decoder[1].running_mean.copy_(torch.tensor([3.0, 7.0]))
        saved = trainable_state_dict(model)
        self.assertIn("decoder.1.running_mean", saved)
        self.assertIn("decoder.1.running_var", saved)
        self.assertNotIn("encoder.weight", saved)

        expected_weight = saved["decoder.0.weight"].clone()
        model.decoder[0].weight.data.zero_()
        model.decoder[1].running_mean.zero_()
        load_trainable_state_dict(model, saved)

        self.assertTrue(torch.equal(model.decoder[0].weight, expected_weight))
        self.assertTrue(
            torch.equal(model.decoder[1].running_mean, torch.tensor([3.0, 7.0]))
        )

    def test_lora_is_initially_a_no_op(self):
        base = nn.Linear(5, 7)
        inputs = torch.randn(2, 3, 5)
        expected = base(inputs).detach()
        layer = LoRALinear(base, rank=4, alpha=4)
        self.assertTrue(torch.allclose(layer(inputs), expected))

    def test_e6_injects_every_attention_projection(self):
        model = fake_segmenter()
        replaced = model.configure_lora(rank=4, alpha=4)
        self.assertEqual(len(replaced), 6)
        self.assertTrue(model.encoder._lora_enabled)
        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(any(name.endswith("lora_A") for name in trainable_names))
        self.assertTrue(any(name.endswith("lora_B") for name in trainable_names))
        self.assertTrue(any(name.startswith("decoder.") for name in trainable_names))
        self.assertTrue(
            all(
                "lora_" in name or name.startswith("decoder.")
                for name in trainable_names
            )
        )

    def test_lora_gradients_exist_while_base_stays_frozen(self):
        model = fake_segmenter()
        model.configure_lora(rank=2, alpha=2)
        inputs = torch.randn(2, 4, requires_grad=True)
        output = inputs
        for block in model.encoder.backbone.blocks:
            output = block(output)
        output.sum().backward()
        lora = model.lora_named_parameters()
        self.assertTrue(lora)
        self.assertTrue(all(parameter.grad is not None for _, parameter in lora))
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.encoder.backbone.named_parameters()
                if "lora_" not in name
            )
        )

    def test_volume_normalization_uses_one_patient_level_range(self):
        volume = np.zeros((2, 2, 2), dtype=np.float32)
        volume[:, :, 0] = np.array([[1, 2], [3, 4]])
        volume[:, :, 1] = np.array([[10, 20], [30, 40]])
        normalized, low, high, excluded_zero = normalize_volume(volume, 0, 100)
        self.assertEqual((low, high), (1.0, 40.0))
        self.assertFalse(excluded_zero)
        self.assertLess(normalized[:, :, 0].max(), normalized[:, :, 1].max())

    def test_volume_normalization_excludes_mixed_zero_background(self):
        volume = np.array([0.0, 2.0, 4.0, 8.0], dtype=np.float32).reshape(2, 2, 1)
        normalized, low, high, excluded_zero = normalize_volume(volume, 0, 100)
        self.assertEqual((low, high), (2.0, 8.0))
        self.assertTrue(excluded_zero)
        self.assertEqual(int(normalized[0, 0, 0]), 0)

    def test_hd95_uses_max_of_directional_percentiles(self):
        target = np.zeros((8, 8, 8), dtype=bool)
        prediction = np.zeros_like(target)
        target[1:4, 1:4, 1:4] = True
        prediction[1:4, 1:4, 1:6] = True
        value = hd95_mm(prediction, target, (1.0, 1.0, 1.0))
        self.assertGreaterEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
