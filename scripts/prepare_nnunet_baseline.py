"""Lossless NRRD-to-NIfTI conversion for a fixed-split nnU-Net baseline.

Uses raw uint8 MRI, not DINO resized PNGs. Preserves header geometry, whose
physical units remain unverified. Test labels are outside nnU-Net imagesTr.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import nibabel as nib
import nrrd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.prepare_utah_run2 import EXPECTED_SPLIT_SHA256, sha256_file


def affine_from_header(header):
    if header.get('space') != 'left-posterior-superior':
        raise ValueError('Only verified LPS input is supported')
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(header['space directions'], dtype=float).T
    affine[:3, 3] = np.asarray(header['space origin'], dtype=float)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-10:
        raise ValueError('Invalid spatial transform')
    return np.diag([-1., -1., 1., 1.]) @ affine  # LPS to NIfTI RAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve existing dataset; choose empty output')
    dataset = output / 'nnUNet_raw/Dataset701_UtahLA'
    for name in ('imagesTr', 'labelsTr', 'imagesTs', 'imagesVal'):
        (dataset / name).mkdir(parents=True, exist_ok=True)
    (output / 'evaluation_labels').mkdir(parents=True, exist_ok=True)
    splits, records = {}, []
    for split, count in (('train', 108), ('val', 15), ('test', 31)):
        source = ROOT / f'data/utah2018_run2/{split}_patients.txt'
        if sha256_file(source) != EXPECTED_SPLIT_SHA256[split]:
            raise ValueError('Unexpected split hash')
        splits[split] = source.read_text().split()
        if len(splits[split]) != count or len(set(splits[split])) != count:
            raise ValueError('Unexpected case list')
    if sum(map(len, splits.values())) != len(set().union(*map(set, splits.values()))):
        raise ValueError('Case overlap')
    for split, ids in splits.items():
        for case in ids:
            image_path = ROOT / f'data/utah2018_raw/{case}/lgemri.nrrd'
            mask_path = image_path.with_name('laendo.nrrd')
            image, ih = nrrd.read(str(image_path))
            mask, mh = nrrd.read(str(mask_path))
            affine = affine_from_header(ih)
            if image.shape != mask.shape or not np.allclose(affine, affine_from_header(mh)):
                raise ValueError(f'Image/mask geometry mismatch: {case}')
            if not np.isfinite(image).all():
                raise ValueError(f'Nonfinite image: {case}')
            mask = (mask > 0).astype(np.uint8)
            # 123 development cases support explicit 108/15 training split.
            dest = dataset / ('imagesTr' if split != 'test' else 'imagesTs') / f'{case}_0000.nii.gz'
            label = (dataset / 'labelsTr' if split != 'test' else output / 'evaluation_labels') / f'{case}.nii.gz'
            for array, path in ((image, dest), (mask, label)):
                nii = nib.Nifti1Image(array, affine)
                nii.header.set_xyzt_units('unknown')
                nib.save(nii, str(path))
                restored = nib.load(str(path))
                if not np.array_equal(np.asanyarray(restored.dataobj), array) or not np.allclose(restored.affine, affine):
                    raise ValueError(f'Conversion changed data: {path}')
            if split == 'val':
                # Relative symlinks may not be permitted on Windows; a small validated copy is portable.
                import shutil
                shutil.copy2(dest, dataset / 'imagesVal' / dest.name)
            records.append({'case': case, 'split': split, 'shape': list(image.shape),
                            'raw_dtype': str(image.dtype), 'header_spacing': np.linalg.norm(affine[:3, :3], axis=0).tolist(),
                            'image_sha256': sha256_file(image_path), 'mask_sha256': sha256_file(mask_path),
                            'converted_image_sha256': sha256_file(dest), 'converted_mask_sha256': sha256_file(label)})
            print(f'{split} {len(records)}/154: converted and round-trip verified', flush=True)
    specification = {'channel_names': {'0': 'MRI'}, 'labels': {'background': 0, 'LA': 1},
                     'numTraining': 123, 'file_ending': '.nii.gz', 'overwrite_image_reader_writer': 'SimpleITKIO'}
    (dataset / 'dataset.json').write_text(json.dumps(specification, indent=2)+'\n')
    (output / 'splits_final.json').write_text(json.dumps([{'train': splits['train'], 'val': splits['val']}], indent=2)+'\n')
    (output / 'conversion_provenance.json').write_text(json.dumps({
        'split_sha256': EXPECTED_SPLIT_SHA256, 'cases': records,
        'patient_independence_verified': False, 'physical_spacing_verified': False,
        'planning_scope': '123 development scans, no test scans; weights trained on 108, validation on 15',
        'recipe': 'nnU-Net native-volume normalization/planning/augmentation; not identical DINO preprocessing',
        'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
