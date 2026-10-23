"""CPU unit tests; these do not certify Giant GPU memory consumption."""
import unittest
import argparse
import csv
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
from tests.test_partial_finetuning import make_segmenter
from src.train_e5_e6 import (EXPERIMENTS, build_model, optimizer_groups,
                            trainable_state_dict, load_trainable_state_dict,
                            _audit_successful_gradients)


class E7Tests(unittest.TestCase):
    def model(self, depth=40):
        model = make_segmenter(depth)
        model.encoder._lora_enabled = False
        with patch("src.train_e5_e6.DINOv2Segmenter", return_value=model) as factory:
            result, lora, blocks = build_model("e7", 4, 4)
            factory.assert_called_once_with(model_name="vit_giant", pretrained=True)
        return result, lora, blocks

    def test_registry_and_tail(self):
        model, lora, blocks = self.model()
        self.assertEqual(EXPERIMENTS["e7"]["loss"], "bce_dice")
        self.assertEqual(blocks, (36, 37, 38, 39))
        self.assertEqual(lora, ())
        for name, value in model.named_parameters():
            expected = name.startswith("decoder.") or any(name.startswith(f"encoder.backbone.blocks.{i}.") for i in blocks)
            self.assertEqual(value.requires_grad, expected, name)
        model.train()
        self.assertFalse(model.encoder.backbone.norm.training)
        self.assertTrue(all(not block.training for block in model.encoder.backbone.blocks[:36]))
        self.assertTrue(all(block.training for block in model.encoder.backbone.blocks[36:]))

    def test_wrong_depth_rejected(self):
        with self.assertRaises(ValueError):
            self.model(24)

    def test_step_and_compact_checkpoint(self):
        model, _, _ = self.model()
        groups, _ = optimizer_groups(model, 1e-5, 1e-3, 0.01)
        optimizer = torch.optim.AdamW(groups)
        frozen = {n: p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        tokens = model.encoder(torch.randn(4, 3, 14, 14))
        loss = model.decoder(tokens.squeeze(-1).squeeze(-1)).square().mean()
        loss.backward()
        _audit_successful_gradients(model)
        optimizer.step()
        for n,p in model.named_parameters():
            if n in frozen:
                self.assertTrue(torch.equal(p, frozen[n]), n)
        state = trainable_state_dict(model)
        self.assertTrue(all(n.startswith("decoder.") or any(n.startswith(f"encoder.backbone.blocks.{i}.") for i in range(36,40)) for n in state))
        fresh, _, _ = self.model()
        load_trainable_state_dict(fresh, state)
        self.assertTrue(all(torch.equal(state[k], trainable_state_dict(fresh)[k]) for k in state))

    def test_report_preserves_case_level_limitations(self):
        from src.report_e5_e6 import build
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (42, 2026, 3407):
                directory = root / f"seed{seed}"
                directory.mkdir()
                efficiency = {"method": EXPERIMENTS["e7"]["method"], "seed": seed,
                    "best_epoch": 2, "best_val_dice_3d_patient_mean": 0.8,
                    "total_parameters": 100, "trainable_parameters": 10,
                    "physical_batch_size": 4, "gradient_accumulation_steps": 1,
                    "effective_batch_size": 4, "skipped_optimizer_updates": 0,
                    "peak_vram_gb": 1, "training_seconds": 60, "test_inference_ms_per_slice": 1}
                summary = {"seed": seed, "num_patients": 31,
                    "dice_3d": {"mean": .8}, "iou_3d": {"mean": .7}, "hd95_header_units": {"mean": 2}}
                config = {"seed": seed, "experiment": "e7", "protocol_extension": "test E7"}
                provenance = {"git_dirty": False, "split_sha256": {"train": "test-hash"}}
                for name, content in (("efficiency", efficiency), ("test_summary", summary), ("config", config), ("provenance", provenance)):
                    (directory / f"{name}.json").write_text(json.dumps(content), encoding="utf-8")
                for name in ("best.pth", "last.pth"):
                    (directory / name).write_bytes(b"unit-test fixture; not model weights")
                with (directory / "test_patient_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["patient_id", "dice_3d", "iou_3d", "hd95_header_units"])
                    for i in range(31):
                        writer.writerow([f"case{i}", .8, .7, 2])
            with patch("builtins.print"):
                build(argparse.Namespace(experiment_dir=str(root), output_dir=str(root)))
            aggregate = json.loads((root / "summary.json").read_text())
            self.assertEqual(aggregate["run_status"], "COMPLETE_CASE_LEVEL_WITH_LIMITATIONS")
            self.assertFalse(aggregate["patient_independence_verified"])
            self.assertEqual(aggregate["num_case_seed_records"], 93)


if __name__ == "__main__":
    unittest.main()
