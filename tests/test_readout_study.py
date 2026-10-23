"""CPU regression tests; synthetic tensors are not research results."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from src.model import SegmentationDecoder
from src.readout_study import ReadoutSegmenter, SplitProjection, EarlyStop, cpu_rng, LAYERS
from src.train_e5_e6 import trainable_state_dict, load_trainable_state_dict, optimizer_groups
from src.train_readout_study import check_pilots, atomic_json, verify_completion


class Attention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.proj((q + k + v) / 3)


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = Attention(width)

    def forward(self, x):
        return x + self.attn(x).tanh() * 0.01


class Backbone(nn.Module):
    """24 inexpensive blocks implementing the real intermediate-layer API contract."""
    def __init__(self, width=8):
        super().__init__()
        self.embed_dim = width
        self.patch_embed = nn.Conv2d(3, width, 14, 14)
        self.blocks = nn.ModuleList([Block(width) for _ in range(24)])
        self.norm = nn.LayerNorm(width)
        self.norm_calls = 0
        self.last_requested = None

    def get_intermediate_layers(self, x, n, reshape, return_class_token, norm):
        assert not reshape and not return_class_token and norm
        self.last_requested = tuple(n)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in n:
                outputs.append(self.norm(x))
                self.norm_calls += 1
        return tuple(outputs)


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        with cpu_rng(999):
            self.backbone = Backbone()
            self.x = torch.randn(2, 3, 28, 28)

    def make(self, cell, seed=42):
        return ReadoutSegmenter(copy.deepcopy(self.backbone), cell, seed, target_size=28)

    def test_projection_parameters_forward_and_gradient(self):
        original = nn.Conv2d(8, 256, 1)
        split = SplitProjection(original)
        self.assertEqual(sum(p.numel() for p in original.parameters()), sum(p.numel() for p in split.parameters()))
        x = torch.randn(2, 8, 3, 3, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        torch.testing.assert_close(original(x), split((y,) * 4))
        original(x).square().mean().backward()
        split((y,) * 4).square().mean().backward()
        torch.testing.assert_close(x.grad, y.grad)
        torch.testing.assert_close(original.weight.grad, torch.cat([b.weight.grad for b in split.branches]))
        with torch.no_grad():
            original.weight.add_(original.weight.grad, alpha=-0.01)
            for branch in split.branches:
                branch.weight.add_(branch.weight.grad, alpha=-0.01)
        torch.testing.assert_close(original(x), split((y,) * 4))

    def test_original_decoder_mapping(self):
        model = self.make("F0").eval()
        with cpu_rng(10042):
            old = SegmentationDecoder(8, 28).eval()
        feature = model.encoder(self.x)[0]
        torch.testing.assert_close(model(self.x), old(feature), rtol=1e-4, atol=1e-5)

    def test_four_cells_pairing_and_zero_lora_identity(self):
        cells = {cell: self.make(cell).eval() for cell in ("F0", "F1", "L0", "L1")}
        base = cells["F0"].decoder.state_dict()
        for model in cells.values():
            for name, value in model.decoder.state_dict().items():
                torch.testing.assert_close(value, base[name])
        torch.testing.assert_close(cells["F0"](self.x), cells["L0"](self.x))
        torch.testing.assert_close(cells["F1"](self.x), cells["L1"](self.x))
        for (a, x), (b, y) in zip(cells["L0"].lora_named_parameters(), cells["L1"].lora_named_parameters()):
            self.assertEqual(a, b)
            torch.testing.assert_close(x, y)

    def test_layer_selection_norm_once_and_feature_shapes(self):
        model = self.make("F1")
        features = model.encoder(self.x)
        self.assertEqual(model.encoder.backbone.last_requested, LAYERS)
        self.assertEqual(model.encoder.backbone.norm_calls, 4)
        self.assertEqual([tuple(f.shape) for f in features], [(2, 8, 2, 2)] * 4)
        self.assertFalse(torch.equal(features[0], features[-1]))
        self.assertTrue(all(not f.requires_grad for f in features))

    def test_gradients_and_modes(self):
        for cell in ("F0", "F1", "L0", "L1"):
            model = self.make(cell).train()
            self.assertFalse(model.encoder.backbone.training)
            self.assertTrue(model.decoder.reduce[1].training)
            model(self.x).square().mean().backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertIsNone(parameter.grad, name)
            if cell.startswith("L"):
                self.assertTrue(any(p.grad.abs().sum() > 0 for n, p in model.lora_named_parameters() if n.endswith("lora_B")))
                self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for n, p in model.lora_named_parameters() if n.endswith("lora_A")))

    def test_counts_and_optimizer(self):
        # Exact Large decoder count without allocating a 300M backbone.
        decoder = SegmentationDecoder(1024)
        decoder.reduce[0] = SplitProjection(decoder.reduce[0])
        self.assertEqual(sum(p.numel() for p in decoder.parameters()), 655329)
        self.assertEqual(24 * 4 * (1024 + 3072 + 1024 + 1024), 589824)
        f, l = self.make("F1"), self.make("L1")
        self.assertEqual(l.num_trainable_params() - f.num_trainable_params(), 24 * 6 * 8 * 4)
        groups, names = optimizer_groups(l, 1e-5, 1e-3, 0.01)
        self.assertEqual(sum(p.numel() for g in groups for p in g["params"]), l.num_trainable_params())
        for label, members in names.items():
            for name in members:
                if name.endswith("bias") or "norm" in name:
                    self.assertTrue(label.endswith("wd=0"))

    def test_rng_isolation(self):
        torch.manual_seed(777)
        before = torch.get_rng_state().clone()
        self.make("L1")
        torch.testing.assert_close(torch.get_rng_state(), before)

    def test_save_reload_including_bn(self):
        model = self.make("L1").train()
        model(self.x).sum().backward()
        optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=1e-3)
        optimizer.step()
        model.eval()
        expected = model(self.x).detach()
        state = trainable_state_dict(model)
        self.assertTrue(any("running_mean" in n for n in state))
        restored = self.make("L1").eval()
        load_trainable_state_dict(restored, state)
        torch.testing.assert_close(restored(self.x), expected)

    def test_early_stop_strict_ties_and_resume(self):
        stop = EarlyStop()
        self.assertEqual(stop.update(0.8, 1), (True, False))
        for epoch in range(2, 11):
            self.assertEqual(stop.update(0.8, epoch), (False, False))
        stop = EarlyStop(**vars(stop))
        self.assertEqual(stop.update(0.8, 11), (False, True))
        with self.assertRaises(FloatingPointError):
            stop.update(float("nan"), 12)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            self.make("F2")
        with self.assertRaises(ValueError):
            self.make("F0", 3407)

    def test_all_four_pilots_required_and_fingerprint_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                check_pilots(root, "abc")
            for cell in ("F0", "F1", "L0", "L1"):
                path = root / "pilot" / cell / "seed42"
                path.mkdir(parents=True)
                atomic_json(path / "pilot.json", {"status": "PASS", "fingerprint": "abc"})
            check_pilots(root, "abc")
            with self.assertRaises(RuntimeError):
                check_pilots(root, "changed")


if __name__ == "__main__":
    unittest.main()
