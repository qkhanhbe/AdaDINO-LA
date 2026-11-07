"""nnU-Net ResEncM 2d on the reference split, as a slice-wise counterpart to the 3d_fullres reference.

Pre-specified on 2026-11-01 with the decoder study (post hoc with respect to the main results).
Same plans, fold 0 (108 training / 15 validation volumes), official 1000-epoch recipe and
checkpoint_best selection as the 3d_fullres reference; only the configuration differs. Modes:
`preprocess` builds the 2d data once; `train` trains one seed; `test` predicts the fixed 31 test
volumes once and scores them with the DINO study's native-volume metrics. Wall-clock timing is
not used in this study.
"""
import argparse
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
sys.path.insert(0, str(REPO / 'scripts'))
from run_nnunet_label_efficiency import DATASET, PLANS, check_free_vram, expected_splits, sha256  # noqa: E402

CONFIG = '2d'
FOLD = 0
SEEDS = (42, 2026, 3407)


def nnunet_env(root, results=None):
    env = {'nnUNet_raw': str(root / 'nnUNet_raw'), 'nnUNet_preprocessed': str(root / 'nnUNet_preprocessed'),
           'nnUNet_n_proc_DA': '2', 'nnUNet_compile': 'false', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    if results is not None:
        env['nnUNet_results'] = str(results)
    os.environ.update(env)


def model_folder(results):
    return results / DATASET / f'nnUNetTrainer__{PLANS}__{CONFIG}'


def preprocess(args, root):
    nnunet_env(root)
    plans = json.loads((root / 'nnUNet_preprocessed' / DATASET / (PLANS + '.json')).read_text())
    target = root / 'nnUNet_preprocessed' / DATASET / plans['configurations'][CONFIG]['data_identifier']
    if target.exists():
        print(f'2d data already present: {target}')
        return
    subprocess.run([str(Path(sys.executable).with_name('nnUNetv2_preprocess')), '-d', DATASET.split('_')[0][7:],
                    '-plans_name', PLANS, '-c', CONFIG, '-np', '4'], check=True)


def train(args, root):
    conversion = json.loads((root / 'conversion_provenance.json').read_text())
    split_path = root / 'nnUNet_preprocessed' / DATASET / 'splits_final.json'
    folds = json.loads(split_path.read_text())
    if folds != expected_splits(conversion, args.subsets):
        raise ValueError('Installed folds differ from the pre-specified subsets')
    if len(folds[FOLD]['train']) != 108 or len(folds[FOLD]['val']) != 15:
        raise ValueError('Fold 0 must be the 108/15 reference split')
    results = args.results_root / f'ref2d_seed{args.seed}'
    if results.exists() and not args.resume:
        raise FileExistsError('Existing run preserved; use --resume for an interrupted run')
    if args.resume and not results.is_dir():
        raise FileNotFoundError('Cannot resume a missing run')
    check_free_vram(12)
    if shutil.disk_usage(args.results_root).free < 12 * 1024**3:
        raise RuntimeError('At least 12 GiB of free space required')
    results.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    run_lock = (results / 'run.lock').open('a')
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((results / 'run_provenance.json').read_text()) if args.resume else None
    nnunet_env(root, results)
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
        'status': 'RUNNING', 'study': 'decoder_study_v2 (nnU-Net 2d reference)', 'mode': 'train', 'seed': args.seed,
        'fraction_percent': 100, 'fold': FOLD, 'train_volumes': len(folds[FOLD]['train']),
        'nnunetv2': importlib.metadata.version('nnunetv2'), 'plans': PLANS, 'configuration': CONFIG,
        'epochs_schedule': 1000, 'train_iterations_per_epoch': 250, 'validation_iterations_per_epoch': 50,
        'checkpoint_selection': 'official EMA validation-patch foreground Dice; not native-volume Dice',
        'planning_scope': 'reference plans from 123 development scans',
        'augmentation_workers': 2, 'torch_compile': False, 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'seed_scope': 'Python/NumPy/PyTorch main-process seeds; nondeterministic augmentation, no bitwise guarantee',
        'gpu': torch.cuda.get_device_name(0), 'split_sha256': sha256(split_path),
        'source_split_sha256': conversion['split_sha256'],
        'plans_sha256': sha256(root / 'nnUNet_preprocessed' / DATASET / (PLANS + '.json')),
        'trainer_source_sha256': sha256(inspect.getfile(nnUNetTrainer)), 'wrapper_source_sha256': sha256(__file__),
        'timing_note': 'Shared GPU; wall-clock timing is not used for reported results',
    }
    checkpoint_path = None
    if args.resume:
        for key in ('seed', 'fold', 'nnunetv2', 'plans', 'configuration', 'torch', 'split_sha256',
                    'plans_sha256', 'trainer_source_sha256', 'wrapper_source_sha256'):
            if previous.get(key) != metadata[key]:
                raise ValueError(f'Resume provenance mismatch: {key}')
        folder = model_folder(results) / f'fold_{FOLD}'
        candidates = [folder / n for n in ('checkpoint_final.pth', 'checkpoint_latest.pth', 'checkpoint_best.pth')]
        checkpoint_path = next((c for c in candidates if c.is_file()), None)
        if checkpoint_path is None:
            raise FileNotFoundError('No checkpoint exists; refuse to silently restart from scratch')
        metadata['attempt'] = int(previous.get('attempt', 1)) + 1
        metadata['resumed_from'] = {'checkpoint': str(checkpoint_path), 'sha256': sha256(checkpoint_path)}
    else:
        metadata['attempt'] = 1
    trainer = get_trainer_from_args(DATASET, CONFIG, FOLD, plans_identifier=PLANS,
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
    results = args.results_root / f'ref2d_seed{args.seed}'
    metadata = json.loads((results / 'run_provenance.json').read_text())
    if metadata.get('status') != 'TRAIN_COMPLETE_TEST_PENDING' or metadata.get('configuration') != CONFIG:
        raise ValueError('Only a completed 2d training run can be tested')
    output = results / 'heldout_test'
    if output.exists():
        raise FileExistsError('Existing test artifacts preserved; no silent repeat or overwrite')
    conversion = json.loads((root / 'conversion_provenance.json').read_text())
    expected = {case['case'] for case in conversion['cases'] if case['split'] == 'test'}
    images = root / 'nnUNet_raw' / DATASET / 'imagesTs'
    if len(expected) != 31 or {p.name[:-12] for p in images.glob('*_0000.nii.gz')} != expected:
        raise ValueError('The test inputs must be exactly the fixed 31-case set')
    model = model_folder(results)
    checkpoint = model / f'fold_{FOLD}' / 'checkpoint_best.pth'
    if not (model / f'fold_{FOLD}' / 'checkpoint_final.pth').is_file():
        raise FileNotFoundError('Final training checkpoint missing')
    locked = sha256(checkpoint)
    check_free_vram(8)
    nnunet_env(root, results)
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.set_num_threads(2)
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=torch.device('cuda'),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model), use_folds=(FOLD,), checkpoint_name='checkpoint_best.pth')
    output.mkdir()
    audit = {'status': 'TEST_RUNNING', 'seed': args.seed, 'configuration': CONFIG, 'fold': FOLD,
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
    parser.add_argument('--mode', choices=('preprocess', 'train', 'test'), required=True)
    parser.add_argument('--root', type=Path, required=True, help='nnU-Net data folder from prepare_nnunet_baseline.py')
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--subsets', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    args.results_root = args.results_root.resolve()
    if importlib.metadata.version('nnunetv2') != '2.8.1':
        raise RuntimeError('This wrapper was checked against nnunetv2 2.8.1')
    if args.mode == 'preprocess':
        preprocess(args, root)
        return
    if args.seed is None:
        raise ValueError('--seed is required for train/test')
    (train if args.mode == 'train' else test)(args, root)


if __name__ == '__main__':
    main()
