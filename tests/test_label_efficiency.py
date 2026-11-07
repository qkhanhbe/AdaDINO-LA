"""CPU-only checks of the label-efficiency subsets, run matrix and batch cycling."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from src import label_efficiency_study as study
from src import train_label_efficiency as runner

TRAIN = [f"case{i:03d}" for i in range(108)]


class Names(Dataset):
    """Stands in for LeftAtriumDataset: whole cases of 88 named slices."""

    def __init__(self, data_dir, augment=False, case_ids=None):
        cases = sorted(case_ids) if case_ids is not None else TRAIN
        self.names = [f"{c}_slice{k:04d}.png" for c in cases for k in range(88)]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return torch.zeros(1), torch.zeros(1), self.names[i]


def names_of(batches):
    return [list(batch[2]) for batch in batches]


class SubsetTests(unittest.TestCase):
    def test_sizes_nesting_and_membership(self):
        for seed in study.SEEDS:
            subsets = study.nested_subsets(TRAIN, seed)
            self.assertEqual({f: len(v) for f, v in subsets.items()}, {10: 11, 25: 27, 50: 54, 100: 108})
            self.assertTrue(set(subsets[10]) < set(subsets[25]) < set(subsets[50]) < set(subsets[100]))
            self.assertEqual(set(subsets[100]), set(TRAIN))

    def test_deterministic_order_independent_and_seed_specific(self):
        self.assertEqual(study.nested_subsets(TRAIN, 42), study.nested_subsets(list(reversed(TRAIN)), 42))
        self.assertNotEqual(study.nested_subsets(TRAIN, 42)[25], study.nested_subsets(TRAIN, 2026)[25])

    def test_rejects_wrong_training_split(self):
        with self.assertRaises(ValueError):
            study.nested_subsets(TRAIN[:-1], 42)

    def test_stored_subsets_must_match_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subsets.json"
            subsets = study.all_subsets(TRAIN)
            path.write_text(json.dumps({"subsets": subsets}))
            args = argparse.Namespace(subsets=str(path), seed=2026, fraction=25)
            self.assertEqual(runner.load_subset(args, TRAIN), subsets["2026"]["25"])
            subsets["2026"]["25"][0] = "other"
            path.write_text(json.dumps({"subsets": subsets}))
            with self.assertRaises(RuntimeError):
                runner.load_subset(args, TRAIN)


class MatrixTests(unittest.TestCase):
    def test_run_matrix(self):
        self.assertEqual(len(study.RUNS), 5 * 3 * 3 + 3 + 1)
        self.assertEqual(len(set(study.RUNS)), len(study.RUNS))
        reduced = [run for run in study.RUNS if run[0] < 100]
        self.assertEqual({(f, c) for f, c, _ in reduced}, {(f, c) for f in (10, 25, 50) for c in study.CELLS})
        self.assertEqual(sorted(r for r in study.RUNS if r[0] == 100),
                         sorted([(100, "V_SC", s) for s in study.SEEDS] + [(100, "V_LO", 42)]))

    def test_learning_rates_and_parameter_counts(self):
        self.assertEqual(study.encoder_lr("V_SC"), 1e-4)
        self.assertEqual(study.encoder_lr("V_LO"), 1e-4)
        self.assertEqual(study.encoder_lr("V_FU"), 1e-5)
        self.assertEqual(study.EXPECTED_TRAINABLE["V_SC"], study.EXPECTED_TRAINABLE["V_FU"])

    def test_cli_rejects_runs_outside_matrix(self):
        args = runner.parse_args(["--mode", "train", "--cell", "V_FU", "--seed", "42", "--fraction", "100",
                                  *[x for n in ("data-root", "raw-root", "dino-repo", "weights", "subsets")
                                    for x in (f"--{n}", "unused")], "--output-root", "/tmp/le_unused"])
        with self.assertRaises(ValueError):
            runner.run(args)


class BatchCyclingTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(runner, "LeftAtriumDataset", Names)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_every_epoch_has_the_full_data_number_of_batches(self):
        cases = study.nested_subsets(TRAIN, 42)[10]
        batches = names_of(runner.subset_batches(Path("unused"), cases, 42, epoch=3))
        self.assertEqual(len(batches), study.STEPS_PER_EPOCH)
        per_sweep = len(cases) * 88 // 4
        for start in range(0, len(batches) - per_sweep + 1, per_sweep):
            sweep = [name for batch in batches[start:start + per_sweep] for name in batch]
            self.assertEqual(sorted(sweep), sorted(Names(None, case_ids=cases).names))
        self.assertNotEqual(batches[:per_sweep], batches[per_sweep:2 * per_sweep])

    def test_full_data_reproduces_the_v2_batch_order(self):
        seed, epoch = 2026, 7
        observed = names_of(runner.subset_batches(Path("unused"), TRAIN, seed, epoch))
        generator = torch.Generator().manual_seed(seed + 30000 + epoch)
        expected = names_of(DataLoader(Names(None), batch_size=4, shuffle=True, drop_last=True, generator=generator))
        self.assertEqual(observed, expected)


class NnUNetFoldTests(unittest.TestCase):
    def test_reference_fold_then_one_fold_per_fraction_and_seed(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "le_nnunet", Path(__file__).resolve().parents[1] / "scripts/run_nnunet_label_efficiency.py")
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        conversion = {"cases": [{"case": c, "split": "train"} for c in TRAIN]
                      + [{"case": f"val{i}", "split": "val"} for i in range(15)]
                      + [{"case": f"test{i}", "split": "test"} for i in range(31)]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subsets.json"
            path.write_text(json.dumps({"subsets": study.all_subsets(TRAIN)}))
            folds = wrapper.expected_splits(conversion, path)
        self.assertEqual(len(folds), 10)
        self.assertEqual(folds[0]["train"], TRAIN)
        self.assertTrue(all(f["val"] == [f"val{i}" for i in range(15)] for f in folds))
        for (fraction, seed), index in wrapper.FOLDS.items():
            self.assertEqual(folds[index]["train"], study.nested_subsets(TRAIN, seed)[fraction])


if __name__ == "__main__":
    unittest.main()
