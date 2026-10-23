"""Evaluate nnU-Net binary NIfTI predictions using the DINO study's 3D metric code.

No resampling, patient-identity inference, threshold tuning or test-driven selection.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def load_aligned_binary(prediction_path, label_path):
    prediction = nib.load(str(prediction_path))
    target = nib.load(str(label_path))
    if prediction.shape != target.shape or len(target.shape) != 3:
        raise ValueError('Prediction must have the exact native 3D target shape')
    if not np.allclose(prediction.affine, target.affine, atol=1e-5, rtol=1e-6):
        raise ValueError('Prediction and target affine geometry differ')
    basis = target.affine[:3, :3]
    gram = basis.T @ basis
    if not np.allclose(gram, np.diag(np.diag(gram)), atol=1e-5):
        raise ValueError('Surface-distance implementation requires orthogonal axes')
    spacing = np.sqrt(np.diag(gram))
    if not np.isfinite(spacing).all() or not (spacing > 0).all():
        raise ValueError('Invalid header spacing')
    arrays = [np.asanyarray(image.dataobj) for image in (prediction, target)]
    for array in arrays:
        if not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
            raise ValueError('Expected finite binary labels, not probabilities or multiclass labels')
    return arrays[0].astype(bool), arrays[1].astype(bool), tuple(spacing)


def describe(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite metric result')
    return {'mean': float(values.mean()), 'std_across_cases': float(values.std(ddof=1)),
            'min': float(values.min()), 'max': float(values.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--metric-source', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hd95', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous evaluation; choose a new version directory')
    conversion_path = args.data / 'conversion_provenance.json'
    conversion = json.loads(conversion_path.read_text())
    cases = [case for case in conversion['cases'] if case['split'] == args.split]
    expected = {case['case'] for case in cases}
    if len(cases) != {'val': 15, 'test': 31}[args.split] or len(expected) != len(cases):
        raise ValueError('Unexpected number of evaluation cases')
    actual = {path.name[:-7] for path in args.predictions.glob('*.nii.gz')}
    if actual != expected:
        raise ValueError(f'Prediction case mismatch: missing={expected-actual}, extra={actual-expected}')
    spec = importlib.util.spec_from_file_location('verified_study_volume_metrics', args.metric_source)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    labels = (args.data / 'evaluation_labels' if args.split == 'test'
              else args.data / 'nnUNet_raw/Dataset701_UtahLA/labelsTr')
    rows = []
    for case in cases:
        identity = case['case']
        prediction_path = args.predictions / f'{identity}.nii.gz'
        label_path = labels / f'{identity}.nii.gz'
        if digest(label_path) != case['converted_mask_sha256']:
            raise ValueError(f'Reference label differs from conversion provenance: {identity}')
        prediction, target, spacing = load_aligned_binary(prediction_path, label_path)
        dice, iou = metrics.overlap_metrics(prediction, target)
        row = {'case_id': identity, 'seed': args.seed, 'dice_3d': dice, 'iou_3d': iou,
               'native_shape': list(target.shape), 'header_spacing': list(spacing),
               'prediction_sha256': digest(prediction_path), 'reference_sha256': digest(label_path)}
        if args.hd95:
            # The existing function name is historical; header units are not verified millimetres.
            row['hd95_header_units'] = metrics.hd95_mm(prediction, target, spacing)
        rows.append(row)
        print(f'{identity}: Dice={dice:.6f}', flush=True)
    keys = ['dice_3d', 'iou_3d'] + (['hd95_header_units'] if args.hd95 else [])
    summary = {
        'method': 'nnUNet_ResEncM_3d_fullres', 'seed': args.seed, 'split': args.split,
        'num_cases': len(rows), 'aggregation': 'equal weight per native 3D case',
        'patient_independence_verified': False, 'physical_spacing_verified': False,
        'std_scope': 'cases within one fitted model; not retraining variability',
        'postprocessing': 'none in this evaluator',
        'prediction_rule': 'binary labels supplied by nnU-Net; no evaluator threshold selection',
        'metric_source_sha256': digest(args.metric_source),
        'conversion_provenance_sha256': digest(conversion_path),
        'source_split_sha256': conversion['split_sha256'],
        **{key: describe([row[key] for row in rows]) for key in keys},
    }
    args.output.mkdir(parents=True)
    (args.output / f'{args.split}_case_metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    (args.output / f'{args.split}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
