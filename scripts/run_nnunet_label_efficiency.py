"""nnU-Net ResEncM 3d_fullres on the label-efficiency subsets (same recipe as the reference baseline).

Modes: `splits` installs one predefined fold per (fraction, seed) next to the reference fold 0;
`train` trains one fold with the official 1000-epoch recipe and selects checkpoint_best on the
fixed 15 validation volumes; `test` predicts the fixed 31 test volumes once and scores them with
the DINO study's native-volume metrics. Plans stay those of the reference (123 development
scans); only the training cases are reduced. Wall-clock timing is not used in this study.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.label_efficiency_study import SEEDS, all_subsets  # noqa: E402

DATASET = 'Dataset701_UtahLA'
PLANS = 'nnUNetResEncUNetMPlans'
CONFIG = '3d_fullres'
FOLDS = {(f, s): 1 + i for i, (f, s) in enumerate((f, s) for f in (10, 25, 50) for s in SEEDS)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def expected_splits(conversion, subsets):
    by_split = {s: [c['case'] for c in conversion['cases'] if c['split'] == s] for s in ('train', 'val', 'test')}
    if [len(by_split[s]) for s in ('train', 'val', 'test')] != [108, 15, 31]:
        raise ValueError('Unexpected conversion split counts')
    stored = json.loads(Path(subsets).read_text())['subsets']
    if stored != all_subsets(by_split['train']):
        raise ValueError('Subset file differs from the pre-specified subset rule')
    folds = [{'train': by_split['train'], 'val': by_split['val']}]
    for (fraction, seed), index in sorted(FOLDS.items(), key=lambda item: item[1]):
        assert index == len(folds)
        folds.append({'train': stored[str(seed)][str(fraction)], 'val': by_split['val']})
    return folds


def install_splits(root, subsets):
    conversion = json.loads((root / 'conversion_provenance.json').read_text())
    folds = expected_splits(conversion, subsets)
    reference = json.loads((root / 'splits_final.json').read_text())
    if reference != folds[:1]:
        raise ValueError('Reference fold 0 differs from the converted 108/15 split')
    target = root / 'nnUNet_preprocessed' / DATASET / 'splits_final.json'
    if target.exists() and json.loads(target.read_text()) != folds:
        raise FileExistsError('A different splits_final.json is installed; refusing to replace it')
    target.write_text(json.dumps(folds, indent=2) + '\n')
    print(f'installed {len(folds)} folds -> {target}')


def run_folder(root, fraction, seed):
    return root / f'le_f{fraction:03d}_seed{seed}'


def model_folder(results):
    return results / DATASET / f'nnUNetTrainer__{PLANS}__{CONFIG}'


def check_free_vram(minimum_gib):
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                   text=True).split()
    if len(free) != 1 or int(free[0]) < minimum_gib * 1024:
        raise RuntimeError(f'Need {minimum_gib} GiB free GPU memory; other jobs are left untouched')


def train(args, root):
    fold = FOLDS[(args.fraction, args.seed)]
    conversion = json.loads((root / 'conversion_provenance.json').read_text())
    split_path = root / 'nnUNet_preprocessed' / DATASET / 'splits_final.json'
    folds = json.loads(split_path.read_text())
    if folds != expected_splits(conversion, args.subsets):
        raise ValueError('Installed folds differ from the pre-specified subsets')
    results = run_folder(root, args.fraction, args.seed)
    if results.exists() and not args.resume:
        raise FileExistsError('Existing run preserved; use --resume for an interrupted run')
    if args.resume and not results.is_dir():
        raise FileNotFoundError('Cannot resume a missing run')
    check_free_vram(12)
    if shutil.disk_usage(root).free < 12 * 1024**3:
        raise RuntimeError('At least 12 GiB of free space required')
    results.mkdir(exist_ok=args.resume)
    import fcntl
    run_lock = (results / 'run.lock').open('a')
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((results / 'run_provenance.json').read_text()) if args.resume else None
    os.environ.update({
        'nnUNet_raw': str(root / 'nnUNet_raw'), 'nnUNet_preprocessed': str(root / 'nnUNet_preprocessed'),
        'nnUNet_results': str(results), 'nnUNet_n_proc_DA': '2', 'nnUNet_compile': 'false',
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
    })
    import numpy as np
    import torch
    from nnunetv2.run.run_training import get_trainer_from_args
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    metadata = {
        'status': 'RUNNING', 'study': 'label_efficiency_v2', 'mode': 'train', 'seed': args.seed,
        'fraction_percent': args.fraction, 'fold': fold, 'train_volumes': len(folds[fold]['train']),
        'nnunetv2': importlib.metadata.version('nnunetv2'), 'plans': PLANS, 'configuration': CONFIG,
        'epochs_schedule': 1000, 'train_iterations_per_epoch': 250, 'validation_iterations_per_epoch': 50,
        'checkpoint_selection': 'official EMA validation-patch foreground Dice; not native-volume Dice',
        'planning_scope': 'reference plans from 123 development scans; training cases reduced only',
        'augmentation_workers': 2, 'torch_compile': False, 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'seed_scope': 'Python/NumPy/PyTorch main-process seeds; nondeterministic augmentation, no bitwise guarantee',
        'gpu': torch.cuda.get_device_name(0), 'split_sha256': sha256(split_path),
        'subsets_sha256': sha256(args.subsets), 'source_split_sha256': conversion['split_sha256'],
        'plans_sha256': sha256(root / 'nnUNet_preprocessed' / DATASET / (PLANS + '.json')),
        'trainer_source_sha256': sha256(inspect.getfile(nnUNetTrainer)), 'wrapper_source_sha256': sha256(__file__),
        'timing_note': 'Shared GPU; wall-clock timing is not used for reported results',
    }
    checkpoint_path = None
    if args.resume:
        for key in ('seed', 'fraction_percent', 'fold', 'nnunetv2', 'plans', 'torch', 'split_sha256',
                    'plans_sha256', 'trainer_source_sha256', 'wrapper_source_sha256'):
            if previous.get(key) != metadata[key]:
                raise ValueError(f'Resume provenance mismatch: {key}')
        folder = model_folder(results) / f'fold_{fold}'
        candidates = [folder / n for n in ('checkpoint_final.pth', 'checkpoint_latest.pth', 'checkpoint_best.pth')]
        checkpoint_path = next((c for c in candidates if c.is_file()), None)
        if checkpoint_path is None:
            raise FileNotFoundError('No checkpoint exists; refuse to silently restart from scratch')
        metadata['attempt'] = int(previous.get('attempt', 1)) + 1
        metadata['resumed_from'] = {'checkpoint': str(checkpoint_path), 'sha256': sha256(checkpoint_path)}
    else:
        metadata['attempt'] = 1
    trainer = get_trainer_from_args(DATASET, CONFIG, fold, plans_identifier=PLANS,
                                    continue_training=args.resume, device=torch.device('cuda'))
    assert trainer.num_epochs == 1000 and trainer.num_iterations_per_epoch == 250
    assert trainer.num_val_iterations_per_epoch == 50
    if checkpoint_path is not None:
        trainer.load_checkpoint(str(checkpoint_path))
    metadata_path = results / 'run_provenance.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    started = time.perf_counter()
    try:
        trainer.run_training()
        trainer.load_checkpoint(str(Path(trainer.output_folder) / 'checkpoint_best.pth'))
        trainer.perform_actual_validation(False)
        metadata['status'] = 'TRAIN_COMPLETE_TEST_PENDING'
    except BaseException as exc:
        metadata['status'] = 'FAILED_OR_INTERRUPTED'
        metadata['error'] = repr(exc)
        raise
    finally:
        metadata['elapsed_seconds'] = time.perf_counter() - started
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        run_lock.close()


def test(args, root):
    results = run_folder(root, args.fraction, args.seed)
    metadata = json.loads((results / 'run_provenance.json').read_text())
    if metadata.get('status') != 'TRAIN_COMPLETE_TEST_PENDING' or metadata.get('fraction_percent') != args.fraction:
        raise ValueError('Only a completed training run of this fraction can be tested')
    fold = metadata['fold']
    output = results / 'heldout_test'
    if output.exists():
        raise FileExistsError('Existing test artifacts preserved; no silent repeat or overwrite')
    conversion = json.loads((root / 'conversion_provenance.json').read_text())
    expected = {case['case'] for case in conversion['cases'] if case['split'] == 'test'}
    images = root / 'nnUNet_raw' / DATASET / 'imagesTs'
    if len(expected) != 31 or {p.name[:-12] for p in images.glob('*_0000.nii.gz')} != expected:
        raise ValueError('The test inputs must be exactly the fixed 31-case set')
    model = model_folder(results)
    checkpoint = model / f'fold_{fold}' / 'checkpoint_best.pth'
    if not (model / f'fold_{fold}' / 'checkpoint_final.pth').is_file():
        raise FileNotFoundError('Final training checkpoint missing')
    locked = sha256(checkpoint)
    check_free_vram(8)
    os.environ.update({'nnUNet_compile': 'false', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.set_num_threads(2)
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=torch.device('cuda'),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model), use_folds=(fold,), checkpoint_name='checkpoint_best.pth')
    output.mkdir()
    audit = {'status': 'TEST_RUNNING', 'seed': args.seed, 'fraction_percent': args.fraction, 'fold': fold,
             'num_cases': 31, 'checkpoint_sha256': locked, 'selection': metadata['checkpoint_selection'],
             'test_time_mirroring': True, 'postprocessing': 'none', 'wrapper_sha256': sha256(__file__)}
    audit_path = output / 'test_provenance.json'
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    try:
        predictor.predict_from_files(str(images), str(output / 'predictions'), save_probabilities=False,
                                     overwrite=False, num_processes_preprocessing=2,
                                     num_processes_segmentation_export=2)
        if sha256(checkpoint) != locked:
            raise RuntimeError('Selected checkpoint changed during testing')
        subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate_nnunet_native.py')),
                        '--data', str(root), '--predictions', str(output / 'predictions'),
                        '--metric-source', str(REPO / 'src/volume_metrics.py'), '--split', 'test',
                        '--seed', str(args.seed), '--output', str(output / 'metrics'), '--hd95'], check=True)
        audit['status'] = 'TEST_COMPLETE_NATIVE_CASE_EVALUATION'
    except BaseException as exc:
        audit['status'] = 'TEST_FAILED_OR_INTERRUPTED'
        audit['error'] = repr(exc)
        raise
    finally:
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('splits', 'train', 'test'), required=True)
    parser.add_argument('--root', type=Path, required=True, help='nnU-Net data folder from prepare_nnunet_baseline.py')
    parser.add_argument('--subsets', type=Path, required=True)
    parser.add_argument('--fraction', type=int, choices=(10, 25, 50))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.mode == 'splits':
        install_splits(root, args.subsets)
        return
    if args.fraction is None or args.seed is None:
        raise ValueError('--fraction and --seed are required for train/test')
    if importlib.metadata.version('nnunetv2') != '2.8.1':
        raise RuntimeError('This wrapper was checked against nnunetv2 2.8.1')
    (train if args.mode == 'train' else test)(args, root)


if __name__ == '__main__':
    main()
