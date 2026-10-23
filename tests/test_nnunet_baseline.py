import copy
from pathlib import Path
import tempfile
import unittest

from scripts.run_nnunet_baseline import validate_split, select_resume_checkpoint, validate_resume
from scripts.test_nnunet_completed_run import validate_completed_run


class SplitGuardTests(unittest.TestCase):
    def setUp(self):
        self.split = [{'train': [f't{i}' for i in range(108)],
                       'val': [f'v{i}' for i in range(15)]}]
        self.conversion = {'cases': [
            {'case': case, 'split': split}
            for split, cases in [('train', self.split[0]['train']),
                                 ('val', self.split[0]['val']),
                                 ('test', [f'x{i}' for i in range(31)])]
            for case in cases]}

    def test_fixed_split_accepted(self):
        validate_split(self.split, self.conversion)

    def test_test_case_in_training_rejected(self):
        split = copy.deepcopy(self.split)
        split[0]['train'][0] = 'x0'
        with self.assertRaises(ValueError):
            validate_split(split, self.conversion)

    def test_automatic_folds_rejected(self):
        with self.assertRaises(ValueError):
            validate_split(self.split * 5, self.conversion)

    def test_duplicate_id_rejected(self):
        split = copy.deepcopy(self.split)
        split[0]['val'][0] = split[0]['val'][1]
        with self.assertRaises(ValueError):
            validate_split(split, self.conversion)

    def test_pilot_cannot_enter_test_stage(self):
        with self.assertRaises(ValueError):
            validate_completed_run({'mode': 'pilot', 'status': 'PILOT_COMPLETE_NOT_A_BASELINE_RESULT'})

    def test_completed_standard_training_accepted(self):
        validate_completed_run({'mode': 'train', 'status': 'TRAIN_COMPLETE_TEST_PENDING',
                                'epochs_schedule': 1000, 'executed_epoch_budget': 1000,
                                'plans': 'nnUNetResEncUNetMPlans'})

    def test_resume_checkpoint_priority_and_missing_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                select_resume_checkpoint(root)
            (root / 'checkpoint_best.pth').touch()
            self.assertEqual(select_resume_checkpoint(root).name, 'checkpoint_best.pth')
            (root / 'checkpoint_latest.pth').touch()
            self.assertEqual(select_resume_checkpoint(root).name, 'checkpoint_latest.pth')
            (root / 'checkpoint_final.pth').touch()
            self.assertEqual(select_resume_checkpoint(root).name, 'checkpoint_final.pth')

    def test_resume_rejects_completed_run(self):
        with self.assertRaises(ValueError):
            validate_resume({'mode': 'train', 'status': 'TRAIN_COMPLETE_TEST_PENDING'}, {})

    def test_resume_rejects_changed_seed(self):
        with self.assertRaisesRegex(ValueError, 'seed'):
            validate_resume({'mode': 'train', 'status': 'RUNNING', 'seed': 42},
                            {'mode': 'train', 'seed': 2026})


if __name__ == '__main__':
    unittest.main()
