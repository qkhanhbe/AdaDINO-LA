"""CPU tests for the revision-run options; synthetic tensors only."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import LeftAtriumDataset  # noqa: E402
from metrics import DiceBCELoss, LegacyDiceBCELoss  # noqa: E402
import train_e5_e6 as trainer  # noqa: E402


class LossImplementationTest(unittest.TestCase):
    def test_legacy_matches_fp32_definition_in_fp32(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 1, 16, 16)
        targets = (torch.rand(3, 1, 16, 16) > 0.7).float()
        legacy = LegacyDiceBCELoss()(logits, targets)
        current = DiceBCELoss()(logits, targets)
        self.assertTrue(torch.allclose(legacy, current, atol=1e-7))

    def test_legacy_keeps_input_dtype_while_current_casts(self):
        logits = torch.randn(2, 1, 8, 8).half()
        targets = (torch.rand(2, 1, 8, 8) > 0.5).half()
        self.assertEqual(DiceBCELoss()(logits, targets).dtype, torch.float32)
        probs = torch.sigmoid(logits)
        self.assertEqual(probs.dtype, torch.float16)  # legacy path reduces in input dtype


class TrainSubsetTest(unittest.TestCase):
    def _make(self, root, cases, slices=2):
        for sub in ("images", "masks"):
            (root / sub).mkdir(parents=True)
        for case in cases:
            for k in range(slices):
                name = f"{case}_slice{k:04d}.png"
                Image.fromarray(np.zeros((4, 4), np.uint8)).save(root / "images" / name)
                Image.fromarray(np.zeros((4, 4), np.uint8)).save(root / "masks" / name)

    def test_subset_keeps_whole_cases_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make(root, ["AAA", "BBB", "CCC"])
            ds = LeftAtriumDataset(str(root), case_ids=["AAA", "CCC"])
            self.assertEqual(len(ds), 4)
            self.assertTrue(all(f.startswith(("AAA_", "CCC_")) for f in ds.filenames))

    def test_missing_case_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make(root, ["AAA"])
            with self.assertRaises(ValueError):
                LeftAtriumDataset(str(root), case_ids=["AAA", "ZZZ"])

    def test_default_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make(root, ["AAA", "BBB"])
            self.assertEqual(len(LeftAtriumDataset(str(root))), 4)


class BestTransactionTest(unittest.TestCase):
    def test_pending_promoted_only_when_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            trainer.atomic_save({"epoch": 5}, out / "best.pth.pending")
            self.assertEqual(trainer.reconcile_pending_best(out, 5), "promoted")
            self.assertTrue((out / "best.pth").exists())
            self.assertFalse((out / "best.pth.pending").exists())

    def test_uncommitted_pending_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            trainer.atomic_save({"epoch": 3}, out / "best.pth")
            trainer.atomic_save({"epoch": 4}, out / "best.pth.pending")
            self.assertEqual(trainer.reconcile_pending_best(out, 3), "discarded")
            self.assertEqual(torch.load(out / "best.pth", weights_only=False)["epoch"], 3)

    def test_no_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(trainer.reconcile_pending_best(Path(tmp), 1), "none")

    def test_atomic_save_leaves_no_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "last.pth"
            trainer.atomic_save({"x": 1}, target)
            self.assertTrue(target.exists())
            self.assertFalse(target.with_suffix(".pth.tmp").exists())


class ClippingTest(unittest.TestCase):
    def test_clipping_bounds_gradient_norm(self):
        torch.manual_seed(0)
        layer = torch.nn.Linear(10, 1)
        loss = (layer(torch.randn(8, 10) * 100) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
        norm = torch.sqrt(sum((p.grad ** 2).sum() for p in layer.parameters()))
        self.assertLessEqual(float(norm), 1.0 + 1e-5)

    def test_cli_defaults_preserve_v1(self):
        argv = sys.argv
        try:
            sys.argv = ["x", "--experiment", "e6", "--seed", "42", "--raw_root", "r", "--output_dir", "o"]
            args = trainer.parse_args()
        finally:
            sys.argv = argv
        self.assertEqual(args.loss_impl, "fp32")
        self.assertEqual(args.grad_clip_norm, 0.0)
        self.assertIsNone(args.train_cases)


if __name__ == "__main__":
    unittest.main()
