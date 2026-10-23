"""Test a completed nnU-Net baseline once, using its preselected best checkpoint."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_completed_run(metadata):
    if metadata.get('mode') != 'train' or metadata.get('status') != 'TRAIN_COMPLETE_TEST_PENDING':
        raise ValueError('Only a completed full training run can enter the held-out test stage')
    if metadata.get('epochs_schedule') != 1000 or metadata.get('executed_epoch_budget') != 1000:
        raise ValueError('This evaluator expects the locked 1000-epoch recipe, not a pilot')
    if metadata.get('plans') != 'nnUNetResEncUNetMPlans':
        raise ValueError('Unexpected baseline preset')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--metric-source', type=Path, required=True)
    args = parser.parse_args()
    if importlib.metadata.version('nnunetv2') != '2.8.1':
        raise RuntimeError('Expected nnunetv2==2.8.1')
    metadata = json.loads((args.run / 'run_provenance.json').read_text())
    validate_completed_run(metadata)
    output = args.run / 'heldout_test'
    if output.exists():
        raise FileExistsError('Existing test artifacts preserved; no silent repeat or overwrite')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('GPU must be idle before measuring test inference')
    conversion = json.loads((args.data / 'conversion_provenance.json').read_text())
    expected = {case['case'] for case in conversion['cases'] if case['split'] == 'test'}
    images = args.data / 'nnUNet_raw/Dataset701_UtahLA/imagesTs'
    actual = {path.name[:-12] for path in images.glob('*_0000.nii.gz')}
    if len(expected) != 31 or actual != expected:
        raise ValueError('The test inputs must be exactly the fixed 31-case set')
    if conversion['split_sha256'] != metadata['source_split_sha256']:
        raise ValueError('Training and test source split hashes differ')
    model = args.run / 'Dataset701_UtahLA/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres'
    checkpoint = model / 'fold_0/checkpoint_best.pth'
    if not (model / 'fold_0/checkpoint_final.pth').is_file():
        raise FileNotFoundError('Final training checkpoint missing')
    locked_checkpoint_sha256 = digest(checkpoint)
    os.environ.update({'nnUNet_compile': 'false', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.set_num_threads(2)
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=torch.device('cuda'),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(str(model), use_folds=(0,),
                                                   checkpoint_name='checkpoint_best.pth')
    output.mkdir()
    audit = {'status': 'TEST_RUNNING', 'seed': metadata['seed'], 'num_cases': 31,
             'checkpoint_sha256': locked_checkpoint_sha256,
             'selection': metadata['checkpoint_selection'], 'test_time_mirroring': True,
             'folds': [0], 'postprocessing': 'none',
             'timing_scope': '31-case input preprocessing, sliding-window inference, native export; excludes model loading and metric computation',
             'wrapper_sha256': digest(__file__), 'gpu': torch.cuda.get_device_name(0)}
    audit_path = output / 'test_provenance.json'
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        predictor.predict_from_files(str(images), str(output / 'predictions'), save_probabilities=False,
                                      overwrite=False, num_processes_preprocessing=2,
                                      num_processes_segmentation_export=2)
        torch.cuda.synchronize()
        audit['inference_pipeline_seconds'] = time.perf_counter() - started
        audit['mean_pipeline_seconds_per_case'] = audit['inference_pipeline_seconds'] / 31
        audit['inference_peak_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
        audit['inference_peak_reserved_gb'] = torch.cuda.max_memory_reserved() / 1e9
        if digest(checkpoint) != locked_checkpoint_sha256:
            raise RuntimeError('Selected checkpoint changed during testing')
        subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate_nnunet_native.py')),
                        '--data', str(args.data), '--predictions', str(output / 'predictions'),
                        '--metric-source', str(args.metric_source), '--split', 'test',
                        '--seed', str(metadata['seed']), '--output', str(output / 'metrics'), '--hd95'], check=True)
        audit['status'] = 'TEST_COMPLETE_NATIVE_CASE_EVALUATION'
    except BaseException as exc:
        audit['status'] = 'TEST_FAILED_OR_INTERRUPTED'
        audit['error'] = repr(exc)
        raise
    finally:
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()
