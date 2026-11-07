"""CPU-only checks of the decoder-study matrix, decoder and model wiring; synthetic tensors only."""
import copy
import unittest

import torch

from src import decoder_study as study
from src import train_decoder_study as runner
from src.protocol_v2_study import EXPECTED_TRAINABLE as V2_TRAINABLE, ProtocolV2Segmenter
from src.readout_study import cpu_rng
from src.train_e5_e6 import load_trainable_state_dict, optimizer_groups, trainable_state_dict
from tests.test_readout_study import Backbone

LIGHT_DECODER = 655_329


def model_args(mode, cell, seed, fraction):
    paths = [x for n in ("data-root", "raw-root", "dino-repo", "weights", "subsets") for x in (f"--{n}", "unused")]
    return runner.parse_args(["--mode", mode, "--cell", cell, "--seed", str(seed), "--fraction", str(fraction),
                              *paths, "--output-root", "/tmp/decoder_unused"])


class MatrixTests(unittest.TestCase):
    def test_run_matrix(self):
        self.assertEqual(len(study.RUNS), 4 * 3 + 2 * 3)
        self.assertEqual(len(set(study.RUNS)), len(study.RUNS))
        self.assertEqual({(f, c) for f, c, _ in study.RUNS if f == 100}, {(100, c) for c in study.CELLS})
        self.assertEqual({(f, c) for f, c, _ in study.RUNS if f == 10}, {(10, "D_LO"), (10, "D_FU")})

    def test_parameter_counts_follow_v2(self):
        self.assertEqual(sum(p.numel() for p in study.HierarchicalDecoder(1024).parameters()), study.DECODER_PARAMS)
        for cell, light in study.LIGHT_COUNTERPART.items():
            self.assertEqual(study.ENCODER_TRAINABLE[cell], V2_TRAINABLE[light] - LIGHT_DECODER)
        self.assertEqual(study.encoder_lr("D_LO"), 1e-4)
        self.assertEqual(study.encoder_lr("D_FU"), 1e-5)

    def test_cli_rejects_runs_outside_matrix(self):
        for mode, cell, fraction in (("train", "D_FR", 10), ("pilot", "D_FR", 10), ("train", "D_LO", 25)):
            with self.assertRaises(ValueError):
                runner.run(model_args(mode, cell, 42, fraction))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        with cpu_rng(999):
            cls.backbone = Backbone()
            cls.x = torch.randn(1, 3, 448, 448)

    def model(self, cell, seed=42):
        return study.DecoderStudySegmenter(copy.deepcopy(self.backbone), cell, seed)

    def test_output_size_and_gradients_reach_every_decoder_part(self):
        model = self.model("D_FR").train()
        logits = model(self.x)
        self.assertEqual(tuple(logits.shape), (1, 1, 448, 448))
        logits.mean().backward()
        for name, parameter in model.decoder.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))

    def test_adaptation_matches_v2(self):
        for cell, light in study.LIGHT_COUNTERPART.items():
            ours = self.model(cell)
            v2 = ProtocolV2Segmenter(copy.deepcopy(self.backbone), light, 42)
            ours_encoder = {n: p for n, p in ours.encoder.named_parameters() if p.requires_grad}
            v2_encoder = {n: p for n, p in v2.encoder.named_parameters() if p.requires_grad}
            self.assertEqual(sorted(ours_encoder), sorted(v2_encoder), cell)
            for name, parameter in ours_encoder.items():
                # Same adapter initialization stream: LoRA starts from the v2 adapters.
                torch.testing.assert_close(parameter, v2_encoder[name])

    def test_decoder_uses_decoder_learning_rate(self):
        model = self.model("D_LO")
        _, names = optimizer_groups(model, study.encoder_lr("D_LO"), 1e-3, 0.01)
        decoder = [n for key, group in names.items() if key.startswith("lr=0.001") for n in group]
        self.assertEqual(sorted(decoder), sorted(n for n, _ in model.decoder.named_parameters(prefix="decoder")))

    def test_save_reload_restores_outputs(self):
        model = self.model("D_LO").train()
        model(self.x)  # update BatchNorm statistics once
        model.eval()
        with torch.no_grad():
            before = model(self.x)
            state = trainable_state_dict(model)
            for parameter in model.decoder.parameters():
                parameter.zero_()
            load_trainable_state_dict(model, state)
            torch.testing.assert_close(model(self.x), before)


if __name__ == "__main__":
    unittest.main()
