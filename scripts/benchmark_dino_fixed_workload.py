"""Measure DINO configuration throughput on a shared GPU, without changing training results.

These are fresh fixed-workload timing runs, not re-training experiments or
reconstructions of historical full-run times. Their variability is measurement
repeat variability, not three-seed segmentation uncertainty.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--experiment', choices=tuple(f'e{i}' for i in range(8)), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source = a.source.resolve()
    if a.output.exists():
        raise FileExistsError('Existing measurements preserved')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('Do not benchmark while another GPU process is active')
    sys.path.insert(0, str(source))
    os.chdir(source)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from src.dataset import get_dataloaders
    from src.metrics import DiceBCELoss
    from src.train_e5_e6 import (audit_protocol_data, build_model, optimizer_groups,
                                set_seed, train_epoch, pretrained_weights_provenance,
                                dinov2_hub_revision, EXPERIMENTS)
    if '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This shared-hardware timing protocol requires RTX 4090')
    set_seed(42)
    torch.set_num_threads(4)
    _, provenance = audit_protocol_data(source / 'data/utah2018_run2')
    dataset = get_dataloaders(str(source / 'data/utah2018_run2'), 4, 0)['train'].dataset
    indices = np.random.default_rng(20260919).choice(len(dataset), size=480, replace=False).tolist()
    warm_loader = DataLoader(Subset(dataset, indices[:80]), batch_size=4, shuffle=False, num_workers=0)
    # Reuse precisely the same 400 processed training slices and ordering for every configuration.
    measured_loader = DataLoader(Subset(dataset, indices[80:]), batch_size=4, shuffle=False, num_workers=0)
    # Equalize the local file-cache condition before any timed repetitions.
    # Decoding/loading is still repeated and included during each timed pass.
    for cached_batch in measured_loader:
        pass
    del cached_batch
    model, _, blocks = build_model(a.experiment, 4, 4)
    model.cuda()
    groups, _ = optimizer_groups(model, 1e-5, 1e-3, .01)
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)
    criterion = (torch.nn.BCEWithLogitsLoss() if a.experiment == 'e0'
                 else DiceBCELoss(bce_weight=.5, dice_weight=.5))
    scaler = torch.amp.GradScaler('cuda')
    gradient_audit = {'completed': False}
    _, _, warm_skipped = train_epoch(model, warm_loader, criterion, optimizer, scaler,
                                     torch.device('cuda'), gradient_audit)
    if not gradient_audit['completed'] or warm_skipped >= 10:
        raise RuntimeError('Warm-up did not establish successful optimization')
    records = []
    for repeat in range(3):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        _, _, skipped = train_epoch(model, measured_loader, criterion, optimizer, scaler,
                                     torch.device('cuda'), gradient_audit)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        records.append({'repeat': repeat + 1, 'batches': 100, 'slices': 400,
                        'seconds': elapsed, 'slices_per_second': 400 / elapsed,
                        'amp_skipped_updates': skipped, 'successful_updates': 100 - skipped,
                        'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
                        'peak_reserved_gb': torch.cuda.max_memory_reserved() / 1e9})
    if any(r['amp_skipped_updates'] for r in records):
        status = 'MEASURED_WITH_SKIPPED_UPDATES_REQUIRES_REVIEW'
    else:
        status = 'MEASURED_FIXED_WORKLOAD'
    report = {
        'status': status, 'experiment': a.experiment,
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'input_shape': [4, 3, 448, 448], 'gradient_accumulation': 1,
        'warmup_batches': 20, 'warmup_skipped_updates': warm_skipped,
        'loader_workers': 0, 'cpu_threads': 4,
        'input_file_cache': 'one untimed read of every measured slice before model timing',
        'timing_scope': 'processed-slice loading, host-to-device transfer, forward/loss/backward/optimizer, training metrics and gradient audit; no validation/checkpoint/test',
        'weights_scope': 'pretrained encoder and freshly initialized adaptation/decoder; timing-only updates, not fitted segmentation checkpoints',
        'uncertainty_scope': 'three timing repetitions in one process, NOT independent training seeds',
        'not_historical_training_time': True,
        'data_split_sha256': provenance['split_sha256'],
        'ordered_dataset_indices': indices,
        'partial_trainable_blocks': list(blocks),
        'source_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(),
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dinov2_source_revision': dinov2_hub_revision(),
        'pretrained_weights': pretrained_weights_provenance(EXPERIMENTS[a.experiment]['model']),
        'total_parameters': model.num_total_params(), 'trainable_parameters': model.num_trainable_params(),
        'measurements': records,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'experiment', 'measurements')}, indent=2))


if __name__ == '__main__':
    main()
