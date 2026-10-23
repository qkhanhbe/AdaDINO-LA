import unittest

import torch
import torch.nn as nn

from src.model import DINOv2Encoder, DINOv2Segmenter


class FakeBackbone(nn.Module):
    def __init__(self, block_count=8):
        super().__init__()
        self.patch_embed = nn.Linear(3 * 14 * 14, 4)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1)) for _ in range(block_count)]
        )
        self.norm = nn.LayerNorm(4)

    def forward_features(self, images):
        tokens = self.patch_embed(images.flatten(1)).unsqueeze(1)
        for block in self.blocks:
            tokens = block(tokens)
        return {"x_norm_patchtokens": self.norm(tokens)}


def make_segmenter(block_count=8):
    encoder = DINOv2Encoder.__new__(DINOv2Encoder)
    nn.Module.__init__(encoder)
    encoder.backbone = FakeBackbone(block_count)
    encoder.embed_dim = 4
    encoder._trainable_block_indices = None
    encoder.backbone.requires_grad_(False)

    model = DINOv2Segmenter.__new__(DINOv2Segmenter)
    nn.Module.__init__(model)
    model.encoder = encoder
    model.decoder = nn.Linear(4, 1)
    return model


class PartialFineTuningTests(unittest.TestCase):
    def test_only_last_four_blocks_and_decoder_are_trainable(self):
        model = make_segmenter(block_count=8)
        indices = model.configure_partial_finetuning(num_blocks=4)

        self.assertEqual(indices, (4, 5, 6, 7))
        self.assertTrue(all(p.requires_grad for p in model.decoder.parameters()))
        self.assertTrue(
            all(
                not p.requires_grad
                for block in model.encoder.backbone.blocks[:4]
                for p in block.parameters()
            )
        )
        self.assertTrue(
            all(
                p.requires_grad
                for block in model.encoder.backbone.blocks[4:]
                for p in block.parameters()
            )
        )
        self.assertTrue(
            all(not p.requires_grad for p in model.encoder.backbone.patch_embed.parameters())
        )
        self.assertTrue(
            all(not p.requires_grad for p in model.encoder.backbone.norm.parameters())
        )

    def test_train_mode_keeps_frozen_prefix_in_eval_mode(self):
        model = make_segmenter(block_count=8)
        model.configure_partial_finetuning(num_blocks=4)
        model.train()

        self.assertTrue(model.decoder.training)
        self.assertTrue(all(not block.training for block in model.encoder.backbone.blocks[:4]))
        self.assertTrue(all(block.training for block in model.encoder.backbone.blocks[4:]))
        self.assertFalse(model.encoder.backbone.patch_embed.training)
        self.assertFalse(model.encoder.backbone.norm.training)

    def test_partial_forward_backpropagates_only_through_tail_blocks(self):
        model = make_segmenter(block_count=8)
        model.configure_partial_finetuning(num_blocks=4)
        features = model.encoder(torch.randn(2, 3, 14, 14))
        features.sum().backward()

        frozen_gradients = [
            parameter.grad
            for block in model.encoder.backbone.blocks[:4]
            for parameter in block.parameters()
        ]
        tail_gradients = [
            parameter.grad
            for block in model.encoder.backbone.blocks[4:]
            for parameter in block.parameters()
        ]
        self.assertTrue(all(gradient is None for gradient in frozen_gradients))
        self.assertTrue(any(gradient is not None for gradient in tail_gradients))

    def test_invalid_number_of_blocks_fails_fast(self):
        model = make_segmenter(block_count=8)
        with self.assertRaises(ValueError):
            model.configure_partial_finetuning(num_blocks=0)
        with self.assertRaises(ValueError):
            model.configure_partial_finetuning(num_blocks=9)


if __name__ == "__main__":
    unittest.main()
