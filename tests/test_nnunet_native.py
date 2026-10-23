from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
from scripts.evaluate_nnunet_native import load_aligned_binary
from src.volume_metrics import overlap_metrics


class NativeGeometryTests(unittest.TestCase):
    def test_complete_31_case_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / 'data/evaluation_labels'
            predictions = root / 'predictions'
            labels.mkdir(parents=True)
            predictions.mkdir()
            cases = []
            mask = np.zeros((4, 5, 6), dtype=np.uint8)
            mask[1:3, 1:4, 1:5] = 1
            for index in range(31):
                identity = f'case{index:02d}'
                label_path = labels / f'{identity}.nii.gz'
                nib.save(nib.Nifti1Image(mask, np.eye(4)), label_path)
                (predictions / label_path.name).write_bytes(label_path.read_bytes())
                cases.append({'case': identity, 'split': 'test',
                              'converted_mask_sha256': hashlib.sha256(label_path.read_bytes()).hexdigest()})
            (root / 'data/conversion_provenance.json').write_text(json.dumps({
                'cases': cases, 'split_sha256': {'test': 'unit-test-fixture'}}))
            repository = Path(__file__).resolve().parents[1]
            subprocess.run([sys.executable, str(repository / 'scripts/evaluate_nnunet_native.py'),
                            '--data', str(root / 'data'), '--predictions', str(predictions),
                            '--metric-source', str(repository / 'src/volume_metrics.py'),
                            '--split', 'test', '--seed', '42', '--output', str(root / 'evaluation'),
                            '--hd95'], check=True, capture_output=True)
            summary = json.loads((root / 'evaluation/test_summary.json').read_text())
            self.assertEqual(summary['num_cases'], 31)
            self.assertEqual(summary['dice_3d']['mean'], 1.)
            self.assertEqual(summary['hd95_header_units']['mean'], 0.)
            self.assertFalse(summary['patient_independence_verified'])
            self.assertFalse(summary['physical_spacing_verified'])

    def test_identical_native_binary_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'mask.nii.gz'
            mask = np.zeros((4, 5, 6), dtype=np.uint8)
            mask[1:3, 2:4, 2:4] = 1
            nib.save(nib.Nifti1Image(mask, np.diag([-1., -2., 3., 1.])), path)
            prediction, target, spacing = load_aligned_binary(path, path)
            self.assertEqual(spacing, (1., 2., 3.))
            self.assertEqual(overlap_metrics(prediction, target), (1., 1.))

    def test_misaligned_prediction_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask = np.zeros((4, 5, 6), dtype=np.uint8)
            nib.save(nib.Nifti1Image(mask, np.eye(4)), root / 'target.nii.gz')
            shifted = np.eye(4)
            shifted[0, 3] = 1
            nib.save(nib.Nifti1Image(mask, shifted), root / 'pred.nii.gz')
            with self.assertRaises(ValueError):
                load_aligned_binary(root / 'pred.nii.gz', root / 'target.nii.gz')

    def test_probability_map_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'prob.nii.gz'
            nib.save(nib.Nifti1Image(np.full((3, 3, 3), .5, dtype=np.float32), np.eye(4)), path)
            with self.assertRaises(ValueError):
                load_aligned_binary(path, path)


if __name__ == '__main__':
    unittest.main()
