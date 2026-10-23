"""Pinned official nnU-Net recipe with measured pilot/full-run modes (no test selection).

Pilot executes two full 250-update epochs under the 1000-epoch LR schedule.
It is stored separately and never treated as a completed baseline.
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
import time
import types

ROOT = Path(os.environ.get('LA_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
DATASET = 'Dataset701_UtahLA'
PLANS = 'nnUNetResEncUNetMPlans'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def select_resume_checkpoint(folder):
    for name in ('checkpoint_final.pth', 'checkpoint_latest.pth', 'checkpoint_best.pth'):
        candidate = Path(folder) / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('No checkpoint exists; refuse to silently restart from scratch')


def validate_resume(previous, current):
    if previous.get('mode') != 'train' or previous.get('status') not in ('RUNNING', 'FAILED_OR_INTERRUPTED'):
        raise ValueError('Resume is only for an interrupted full-training attempt')
    keys = ('mode', 'seed', 'nnunetv2', 'plans', 'configuration', 'epochs_schedule',
            'train_iterations_per_epoch', 'validation_iterations_per_epoch',
            'augmentation_workers', 'torch_compile', 'torch', 'cuda_runtime', 'cudnn',
            'cudnn_benchmark', 'cudnn_deterministic', 'gpu', 'split_sha256',
            'source_split_sha256', 'plans_sha256', 'trainer_source_sha256', 'wrapper_source_sha256')
    for key in keys:
        if key not in previous or previous[key] != current.get(key):
            raise ValueError(f'Resume provenance mismatch: {key}')


def validate_split(split, conversion):
    if len(split) != 1:
        raise ValueError('Use one predefined fold, not automatic cross-validation')
    train, val = split[0]['train'], split[0]['val']
    expected = {s: {r['case'] for r in conversion['cases'] if r['split'] == s}
                for s in ('train', 'val', 'test')}
    if len(train) != 108 or len(val) != 15 or len(expected['test']) != 31:
        raise ValueError('Unexpected split counts')
    if set(train) != expected['train'] or set(val) != expected['val']:
        raise ValueError('Split differs from converted source protocol')
    if set(train) & set(val) or (set(train) | set(val)) & expected['test']:
        raise ValueError('Case overlap')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('pilot', 'train'), required=True)
    parser.add_argument('--seed', type=int, choices=(42, 2026, 3407), default=42)
    parser.add_argument('--resume', action='store_true', help='Resume an interrupted full run with unchanged recipe and source')
    parser.add_argument('--pilot-tag', default='', help='New pilot output suffix; preserves failed pilot artifacts')
    args = parser.parse_args()
    if args.pilot_tag and (args.mode != 'pilot' or not args.pilot_tag.replace('_', '').isalnum()):
        raise ValueError('pilot-tag must be alphanumeric/underscore and used only for pilot')
    if args.resume and args.mode != 'train':
        raise ValueError('Pilot runs are not resumable full-training baselines')
    if importlib.metadata.version('nnunetv2') != '2.8.1':
        raise RuntimeError('This wrapper was checked against nnunetv2 2.8.1')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('GPU must be idle before starting a measured run')
    if shutil.disk_usage(ROOT).free < 12 * 1024**3:
        raise RuntimeError('At least 12 GiB of free space required')
    data = ROOT / 'NNUNET_UtahLA_20260919'
    preprocessed = data / 'nnUNet_preprocessed'
    split_path = preprocessed / DATASET / 'splits_final.json'
    conversion = json.loads((data / 'conversion_provenance.json').read_text())
    validate_split(json.loads(split_path.read_text()), conversion)
    results = data / f'{args.mode}_seed{args.seed}'
    if args.pilot_tag:
        results = data / f'pilot_seed{args.seed}_{args.pilot_tag}'
    if results.exists() and not args.resume:
        raise FileExistsError('Existing run preserved; explicit audited resume required')
    if args.resume and not results.is_dir():
        raise FileNotFoundError('Cannot resume a missing run')
    results.mkdir(exist_ok=args.resume)
    # A live process holds the lock; reboot releases it without relying on a stale PID file.
    import fcntl
    run_lock = (results / 'run.lock').open('a')
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((results / 'run_provenance.json').read_text()) if args.resume else None
    os.environ.update({
        'nnUNet_raw': str(data / 'nnUNet_raw'),
        'nnUNet_preprocessed': str(preprocessed),
        'nnUNet_results': str(results),
        'nnUNet_n_proc_DA': '2', 'nnUNet_compile': 'false',
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
        'status': 'RUNNING', 'mode': args.mode, 'seed': args.seed,
        'nnunetv2': '2.8.1', 'plans': PLANS, 'configuration': '3d_fullres',
        'epochs_schedule': 1000, 'executed_epoch_budget': 2 if args.mode == 'pilot' else 1000,
        'train_iterations_per_epoch': 250, 'validation_iterations_per_epoch': 50,
        'checkpoint_selection': 'official EMA validation-patch foreground Dice; not native-volume Dice',
        'evaluation': 'native-volume validation of selected checkpoint; separate test evaluation still required',
        'planning_scope': '123 development cases, 108 train / 15 validation; no test data',
        'augmentation_workers': 2, 'torch_compile': False,
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'cudnn_benchmark': True, 'cudnn_deterministic': False,
        'seed_scope': 'Python/NumPy/PyTorch main-process seeds; nondeterministic augmentation, no bitwise guarantee',
        'gpu': torch.cuda.get_device_name(0), 'split_sha256': sha256(split_path),
        'source_split_sha256': conversion['split_sha256'],
        'plans_sha256': sha256(preprocessed / DATASET / (PLANS + '.json')),
        'trainer_source_sha256': sha256(inspect.getfile(nnUNetTrainer)),
        'wrapper_source_sha256': sha256(__file__),
        'epoch_units': '250 sampled 3D patch batches, NOT a complete sweep over volumes',
        'throughput_units': 'training 3D patches/second, NOT 2D slices/second',
    }
    checkpoint_path = None
    if args.resume:
        validate_resume(previous, metadata)
        checkpoint_path = select_resume_checkpoint(
            results / DATASET / f'nnUNetTrainer__{PLANS}__3d_fullres/fold_0')
        audit = {'previous_provenance': previous,
                 'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha256(checkpoint_path),
                 'note': 'Restores official model/optimizer/scaler/logger checkpoint; restarted stochastic augmentation is not bitwise continuation. Lost partial work and downtime are not reconstructed.'}
        (results / f'resume_audit_{time.time_ns()}.json').write_text(json.dumps(audit, indent=2) + '\n')
    trainer = get_trainer_from_args(DATASET, '3d_fullres', 0,
                                    plans_identifier=PLANS, continue_training=args.resume,
                                    device=torch.device('cuda'))
    assert trainer.num_epochs == 1000 and trainer.num_iterations_per_epoch == 250
    assert trainer.num_val_iterations_per_epoch == 50
    if checkpoint_path is not None:
        trainer.load_checkpoint(str(checkpoint_path))
    metadata['attempt'] = int(previous.get('attempt', 1)) + 1 if previous else 1
    metadata['resumed_from_epoch_zero_based'] = trainer.current_epoch if args.resume else None
    metadata['timing_scope_across_attempts'] = 'Per-attempt epoch records are retained; repeated epochs and lost partial work must be handled explicitly, not labeled uninterrupted runtime'
    metadata_path = results / 'run_provenance.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    started = time.perf_counter()
    records = []
    timers = {}

    original_epoch_start = trainer.on_epoch_start
    original_train_start = trainer.on_train_epoch_start
    original_train_end = trainer.on_train_epoch_end
    original_epoch_end = trainer.on_epoch_end

    def epoch_start(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timers['epoch'] = time.perf_counter()
        original_epoch_start()

    def train_start(self):
        original_train_start()
        torch.cuda.synchronize()
        timers['train'] = time.perf_counter()

    def train_end(self, outputs):
        torch.cuda.synchronize()
        timers['train_seconds'] = time.perf_counter() - timers['train']
        original_train_end(outputs)

    def epoch_end(self):
        epoch = self.current_epoch
        before_best = self._best_ema
        original_epoch_end()
        torch.cuda.synchronize()
        now = time.perf_counter()
        record = {
            'attempt': metadata['attempt'],
            'epoch_one_based': epoch + 1,
            'complete_epoch_seconds': now - timers['epoch'],
            'training_phase_seconds': timers['train_seconds'],
            'train_patches_per_second': self.batch_size * 250 / timers['train_seconds'],
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'elapsed_seconds_including_setup': now - started,
            'new_best_ema_checkpoint': bool(self._best_ema != before_best),
            'mean_patch_dice': float(self.logger.get_value('mean_fg_dice', step=-1)),
            'ema_patch_dice': float(self.logger.get_value('ema_fg_dice', step=-1)),
        }
        records.append(record)
        with (results / 'epoch_measurements.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print('MEASURED_EPOCH ' + json.dumps(record), flush=True)

    trainer.on_epoch_start = types.MethodType(epoch_start, trainer)
    trainer.on_train_epoch_start = types.MethodType(train_start, trainer)
    trainer.on_train_epoch_end = types.MethodType(train_end, trainer)
    trainer.on_epoch_end = types.MethodType(epoch_end, trainer)
    try:
        if args.mode == 'train':
            trainer.run_training()
            trainer.load_checkpoint(str(Path(trainer.output_folder) / 'checkpoint_best.pth'))
            trainer.perform_actual_validation(False)
            metadata['status'] = 'TRAIN_COMPLETE_TEST_PENDING'
        else:
            # Exact official epoch body, retaining num_epochs=1000 and its LR schedule.
            trainer.on_train_start()
            for _ in range(2):
                trainer.on_epoch_start()
                trainer.on_train_epoch_start()
                outputs = [trainer.train_step(next(trainer.dataloader_train)) for _ in range(250)]
                trainer.on_train_epoch_end(outputs)
                with torch.no_grad():
                    trainer.on_validation_epoch_start()
                    outputs = [trainer.validation_step(next(trainer.dataloader_val)) for _ in range(50)]
                    trainer.on_validation_epoch_end(outputs)
                trainer.on_epoch_end()
            metadata['status'] = 'PILOT_COMPLETE_NOT_A_BASELINE_RESULT'
            metadata['projected_1000_epoch_hours_from_second_epoch'] = records[1]['complete_epoch_seconds'] * 1000 / 3600
            metadata['projection_scope'] = 'budget estimate only; excludes final inference, not measured full-run time'
        metadata['total_parameters'] = sum(p.numel() for p in trainer.network.parameters())
        metadata['trainable_parameters'] = sum(p.numel() for p in trainer.network.parameters() if p.requires_grad)
    except BaseException as exc:
        metadata['status'] = 'FAILED_OR_INTERRUPTED'
        metadata['error'] = repr(exc)
        raise
    finally:
        if args.mode == 'pilot':
            for loader in (trainer.dataloader_train, trainer.dataloader_val):
                if callable(getattr(loader, '_finish', None)):
                    loader._finish()
        metadata['elapsed_seconds'] = time.perf_counter() - started
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        run_lock.close()
        print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
