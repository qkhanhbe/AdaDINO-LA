"""Locked readout-study runner. Pilots never evaluate test; training stops before test.

Run as python -m src.train_readout_study. Existing E0--E7 artifacts are read-only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import LeftAtriumDataset
from .metrics import DiceBCELoss
from .model import SegmentationDecoder
from .native_evaluation import evaluate_model_3d, write_evaluation
from .readout_study import CELLS, SEEDS, PROTOCOL, EarlyStop, ReadoutSegmenter, load_backbone, cpu_rng
from .train_e5_e6 import (
    audit_protocol_data, optimizer_groups, trainable_state_dict, load_trainable_state_dict,
    sha256_file, set_seed, rng_state, restore_rng_state, _audit_successful_gradients,
)

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_checkpoint(path, value):
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


@contextmanager
def study_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".study.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def gpu_guard(root):
    """Fail closed; never stop another user's process or run a report on a laptop GPU."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one RTX 4090 GPU")
    props = torch.cuda.get_device_properties(0)
    if not any(model in props.name for model in ("4090", "5090")) or props.total_memory < 23 * 1024**3:
        raise RuntimeError(f"RTX 4090/5090 (>=24GB) required, found {props.name}")
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits",
    ], text=True).strip()
    other = []
    for line in processes.splitlines():
        pid, _, name = (part.strip() for part in line.partition(","))
        # The MPS control daemon is a system service, not another user's job.
        if pid.isdigit() and int(pid) != os.getpid() and "mps-server" not in name:
            other.append(int(pid))
    if other:
        raise RuntimeError(f"Other GPU jobs active; leave them untouched: {other}")
    if shutil.disk_usage(root).free < 12 * 1024**3:
        raise RuntimeError("At least 12 GiB free disk required")
    return props.name


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hashes():
    # Include dependencies and tests, including dirty working-tree bytes, not just git HEAD.
    paths = list((ROOT / "src").glob("*.py"))
    paths += list((ROOT / "scripts").glob("*readout*.py"))
    paths += list((ROOT / "tests").glob("*readout*.py"))
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}


def provenance(args):
    repo = Path(args.dino_repo).resolve()
    data = Path(args.data_root).resolve()
    sources = {p.relative_to(repo).as_posix(): sha256_file(p) for p in sorted(repo.rglob("*.py"))}
    if not sources or "hubconf.py" not in sources:
        raise RuntimeError("No verifiable DINO source")
    gpu = torch.cuda.get_device_properties(0)
    try:
        xformers_version = version("xformers")
    except PackageNotFoundError:
        xformers_version = None
    return {
        "protocol": PROTOCOL, "source_sha256": source_hashes(),
        "dino_source_sha256": digest_json(sources), "dino_source_files": sources,
        "pretrained_sha256": sha256_file(Path(args.weights)),
        "data_metadata_sha256": {name: sha256_file(data / name) for name in (
            "train_patients.txt", "val_patients.txt", "test_patients.txt",
            "dataset_provenance.json", "preprocessing_manifest.csv", "patient_meta.json")},
        "python": platform.python_version(), "torch": str(torch.__version__),
        "packages": {name: version(name) for name in ("torchvision", "numpy", "scipy", "pillow", "pynrrd", "matplotlib", "tqdm")},
        "xformers": xformers_version,
        "thread_settings": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu.name, "total_vram_bytes": gpu.total_memory,
        "driver": subprocess.check_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "tf32": False, "strict_deterministic_algorithms": False,
        "patient_independence": "unverified", "extension_status": "exploratory; test previously seen",
    }


def loader(data, split, seed, epoch=0):
    dataset = LeftAtriumDataset(str(data / f"{split}_2d"), augment=False)
    expected = {"train": 108 * 88, "val": 15 * 88, "test": 31 * 88}[split]
    images = {p.name for p in (data / f"{split}_2d/images").iterdir() if p.is_file()}
    masks = {p.name for p in (data / f"{split}_2d/masks").iterdir() if p.is_file()}
    if images != masks or len(dataset) != expected:
        raise RuntimeError(f"Missing/extra images or masks for {split}")
    # New generator per epoch makes all cells see the same order, including after resume.
    generator = torch.Generator().manual_seed(seed + 30000 + epoch)
    return DataLoader(dataset, batch_size=4, shuffle=split == "train", num_workers=0,
                      drop_last=split == "train", pin_memory=True, generator=generator)


def evaluate(model, args, ids, meta, split, cell, seed, output=None):
    with torch.no_grad():
        return evaluate_model_3d(
            model, loader(Path(args.data_root), split, seed), torch.device("cuda"),
            ids[split], meta, Path(args.raw_root), cell, seed,
            criterion=DiceBCELoss(), threshold=0.5, compute_hd95=split == "test", output_dir=output,
        )


def finite_state(model):
    for name, tensor in model.state_dict().items():
        if (name.startswith("decoder.") or "lora_" in name) and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite parameter/buffer: {name}")


def train_epoch(model, batches, optimizer, scaler, trace, limit=None):
    model.train()
    criterion = DiceBCELoss()
    total, count, skips, consecutive, successful = 0.0, 0, 0, 0, 0
    order_hash = hashlib.sha256()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with trace.open("x", encoding="utf-8") as stream:
        for step, (images, masks, names) in enumerate(batches):
            if limit is not None and step >= limit:
                break
            images, masks = images.cuda(non_blocking=True), masks.cuda(non_blocking=True)
            order_hash.update(("\n".join(names) + "\n").encode())
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(images)
            with torch.amp.autocast("cuda", enabled=False):
                loss = criterion(logits.float(), masks.float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norms = [p.grad.detach().float().norm() for p in model.parameters() if p.requires_grad and p.grad is not None]
            grad_norm = float(torch.stack(norms).norm()) if norms else float("nan")
            scaler.step(optimizer)
            scaler.update()
            after = float(scaler.get_scale())
            skipped = after < before
            if skipped:
                skips += 1
                consecutive += 1
            else:
                if successful == 0:
                    _audit_successful_gradients(model)
                successful += 1
                consecutive = 0
            stream.write(json.dumps({"step": step, "loss": float(loss), "loss_dtype": str(loss.dtype),
                "grad_norm": grad_norm if math.isfinite(grad_norm) else None,
                "grad_norm_finite": math.isfinite(grad_norm), "scale_before": before,
                "scale_after": after, "skipped_update": skipped,
                "elapsed_train_seconds": time.perf_counter() - started}, allow_nan=False) + "\n")
            stream.flush()
            if consecutive >= PROTOCOL["max_consecutive_skips"]:
                raise FloatingPointError("20 consecutive skipped updates; stop and diagnose")
            total += float(loss) * len(names)
            count += len(names)
            if (step + 1) % 100 == 0:
                print(f"step={step + 1} mean_loss={total/count:.6f} skips={skips}", flush=True)
    if successful == 0 or count == 0:
        raise RuntimeError("No successful optimizer update")
    finite_state(model)
    torch.cuda.synchronize()
    return {"train_seconds": time.perf_counter() - started, "train_loss": total / count,
            "slices": count, "successful_updates": successful, "skipped_updates": skips,
            "batch_order_sha256": order_hash.hexdigest()}


def parity_checks(model, seed):
    """Small FP32 check on actual pretrained backend; before any pilot updates."""
    x = torch.zeros(1, 3, 448, 448, device="cuda")
    model.eval()
    with torch.no_grad():
        features = model.encoder(x)
        last = model.encoder.backbone.forward_features(x)["x_norm_patchtokens"]
        last = last.transpose(1, 2).reshape(1, 1024, 32, 32)
        torch.testing.assert_close(features[-1], last, rtol=1e-4, atol=1e-5)
        # Both readouts use the same split projection; check mapped original path.
        with cpu_rng(seed + 10000):
            original = SegmentationDecoder(1024, 448).cuda().eval()
        expected = original(last)
        observed = model.decoder((last,) * 4)
        torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-5)
        if model.encoder._lora_enabled:
            # Zero-adapter identity is also checked independently of sampled training masks.
            for name, parameter in model.lora_named_parameters():
                if name.endswith("lora_B") and torch.count_nonzero(parameter):
                    raise RuntimeError("LoRA B must initially be zero")
    del original, x, features, last, expected, observed


def checkpoint(model, optimizer, scaler, stopper, epoch, identity, history):
    return {"model": trainable_state_dict(model), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "rng": rng_state(), "stopper": vars(stopper),
            "epoch": epoch, "identity": identity, "history": history}


def check_pilots(root, fingerprint):
    for cell in CELLS:
        path = root / "pilot" / cell / "seed42" / "pilot.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS" or result.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Pilot is missing/stale/failed for {cell}")


def verify_evaluation(out, identity):
    result = json.loads((out / "evaluation_complete.json").read_text(encoding="utf-8"))
    if result["identity"] != identity or result["best_sha256"] != sha256_file(out / "best.pth"):
        raise RuntimeError("Evaluation identity/checkpoint mismatch")
    if len([n for n in result["files"] if n.endswith("_prediction.nrrd")]) != 31:
        raise RuntimeError("Expected all 31 native prediction volumes")
    for relative, digest in result["files"].items():
        path = (out / relative).resolve()
        if not path.is_relative_to(out.resolve()) or sha256_file(path) != digest:
            raise RuntimeError(f"Evaluation checksum/path mismatch: {relative}")
    return result


def verify_completion(out, identity):
    result = json.loads((out / "train_complete.json").read_text(encoding="utf-8"))
    if result["identity"] != identity:
        raise RuntimeError("Completed run belongs to different source/protocol")
    if sha256_file(out / "best.pth") != result["best_sha256"]:
        raise RuntimeError("Best checkpoint checksum mismatch")
    return result


def run(args):
    root, data = Path(args.output_root).resolve(), Path(args.data_root).resolve()
    if root == ROOT or root.is_relative_to(ROOT / "results/utah2018_run2_master") or root.is_relative_to(data):
        raise ValueError("Use a separate study output directory")
    with study_lock(root):
        gpu_guard(root)
        set_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        ids, _ = audit_protocol_data(data)
        meta = json.loads((data / "patient_meta.json").read_text(encoding="utf-8"))
        prov = provenance(args)
        fingerprint = digest_json(prov)
        identity = {"fingerprint": fingerprint, "cell": args.cell, "seed": args.seed}
        phase = "pilot" if args.mode == "pilot" else "runs"
        out = root / phase / args.cell / f"seed{args.seed}"
        if args.mode == "pilot" and args.seed != 42:
            raise ValueError("Pilots use seed 42 only")
        if args.mode != "pilot":
            check_pilots(root, fingerprint)
        if out.exists():
            prior = json.loads((out / "config.json").read_text(encoding="utf-8"))
            if prior["identity"] != identity:
                raise RuntimeError("Refusing to overwrite a different study run")
            if args.mode == "pilot":
                result = json.loads((out / "pilot.json").read_text(encoding="utf-8"))
                if result["status"] == "PASS":
                    print("SKIP verified completed pilot")
                    return
                raise RuntimeError("Failed pilot retained; use a fresh output root after diagnosis")
            if args.mode == "train" and (out / "train_complete.json").exists():
                verify_completion(out, identity)
                print("SKIP verified completed training")
                return
            if args.mode == "train" and not args.resume:
                raise RuntimeError("Existing run: --resume required")
        else:
            if args.mode == "evaluate":
                raise RuntimeError("Training must finish before evaluation")
            out.mkdir(parents=True)
            atomic_json(out / "config.json", {"identity": identity, "protocol": PROTOCOL,
                "cell": CELLS[args.cell], "seed": args.seed, "data_root": str(data),
                "raw_root": str(Path(args.raw_root).resolve()), "dino_repo": str(Path(args.dino_repo).resolve()),
                "weights": str(Path(args.weights).resolve())})
            atomic_json(out / "provenance.json", prov)
        if args.mode == "evaluate":
            # Never expose test until all eight matched runs have completed.
            for cell in CELLS:
                for seed in SEEDS:
                    verify_completion(root / "runs" / cell / f"seed{seed}",
                                      {"fingerprint": fingerprint, "cell": cell, "seed": seed})
            if (out / "evaluation_complete.json").exists():
                verify_evaluation(out, identity)
                print("SKIP verified completed evaluation")
                return
        model = ReadoutSegmenter(load_backbone(Path(args.dino_repo), Path(args.weights)), args.cell, args.seed).cuda()
        expected = 655329 + (589824 if args.cell.startswith("L") else 0)
        if model.num_trainable_params() != expected:
            raise RuntimeError("Unexpected trainable parameter count")
        if args.mode == "evaluate":
            state = torch.load(out / "best.pth", map_location="cpu", weights_only=False)
            load_trainable_state_dict(model, state["model"])
            # Separate evaluation attempt: never overwrite predictions from an interrupted attempt.
            dest = out / "evaluations" / uuid.uuid4().hex
            rows, summary, _ = evaluate(model, args, ids, meta, "test", args.cell, args.seed, dest)
            write_evaluation(rows, summary, dest, "test")
            files = {p.relative_to(out).as_posix(): sha256_file(p) for p in dest.rglob("*") if p.is_file()}
            atomic_json(out / "evaluation_complete.json", {"identity": identity,
                "best_sha256": sha256_file(out / "best.pth"), "directory": dest.relative_to(out).as_posix(), "files": files})
            return
        groups, _ = optimizer_groups(model, 1e-5, 1e-3, 0.01)
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        scaler = torch.amp.GradScaler("cuda")
        stopper, history, start_epoch, best_file = EarlyStop(), [], 1, None
        if args.mode == "train" and (out / "last.pth").exists():
            state = torch.load(out / "last.pth", map_location="cpu", weights_only=False)
            if state["identity"] != identity:
                raise RuntimeError("Resume identity mismatch")
            load_trainable_state_dict(model, state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            restore_rng_state(state["rng"])
            stopper, history = EarlyStop(**state["stopper"]), state["history"]
            start_epoch = state["epoch"] + 1
            best_file = state["best_file"]
            if not (out / best_file).is_file():
                raise RuntimeError("Committed best checkpoint missing")
        session = out / "sessions" / uuid.uuid4().hex
        session.mkdir(parents=True)
        atomic_json(session / "start.json", {"identity": identity, "started_unix": time.time(),
                                           "start_epoch": start_epoch, "pid": os.getpid()})
        if args.mode == "pilot":
            parity_checks(model, args.seed)
        torch.cuda.reset_peak_memory_stats()
        epochs = [1] if args.mode == "pilot" else range(start_epoch, 51)
        for epoch in epochs:
            if stopper.stale >= 10:
                break
            gpu_guard(root)
            stats = train_epoch(model, loader(data, "train", args.seed, epoch), optimizer, scaler,
                                session / f"epoch{epoch:03d}_steps.jsonl", 100 if args.mode == "pilot" else None)
            torch.cuda.synchronize()
            val_start = time.perf_counter()
            _, val_summary, val_loss = evaluate(model, args, ids, meta, "val", args.cell, args.seed)
            torch.cuda.synchronize()
            stats.update(epoch=epoch, validation_seconds=time.perf_counter() - val_start,
                         validation_dice=val_summary["dice_3d"]["mean"], validation_loss=val_loss,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved())
            stats["gpu_snapshot"] = subprocess.check_output([
                "nvidia-smi", "--query-gpu=uuid,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True).strip()
            improved, stop = stopper.update(stats["validation_dice"], epoch)
            history.append(stats)
            save_start = time.perf_counter()
            if args.mode == "pilot" or improved:
                # Immutable candidates + last.pth commit prevent best/last inconsistency
                # if power fails between writes. Uncommitted candidates are not selected.
                best_file = (session / f"best_epoch{epoch:03d}.pth").relative_to(out).as_posix()
                atomic_checkpoint(out / best_file, {"model": trainable_state_dict(model),
                    "identity": identity, "epoch": epoch})
            saved = checkpoint(model, optimizer, scaler, stopper, epoch, identity, history)
            saved["best_file"] = best_file
            atomic_checkpoint(out / "last.pth", saved)
            stats["checkpoint_seconds"] = time.perf_counter() - save_start
            atomic_json(session / f"epoch{epoch:03d}.json", stats)
            atomic_json(out / "history.json", history)
            print(json.dumps(stats), flush=True)
            # If another job entered mid-epoch, preserve checkpoint and flag contaminated timing.
            try:
                gpu_guard(root)
            except RuntimeError as error:
                atomic_json(session / "resource_conflict.json", {"epoch": epoch, "reason": str(error)})
                raise
            if args.mode == "pilot":
                model.eval()
                x = next(iter(loader(data, "val", args.seed)))[0].cuda()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    before = model(x).clone()
                state = torch.load(out / best_file, map_location="cpu", weights_only=False)
                with torch.no_grad():
                    for p in model.decoder.parameters():
                        p.zero_()
                load_trainable_state_dict(model, state["model"])
                model.eval()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    torch.testing.assert_close(model(x), before, rtol=1e-3, atol=1e-3)
                shutil.copyfile(out / best_file, out / "best.pth")
                atomic_json(out / "pilot.json", {"status": "PASS", "fingerprint": fingerprint,
                    "identity": identity, "stats": stats, "test_evaluated": False,
                    "checks": ["readout_parity", "parameter_count", "successful_gradient_audit", "finite_state", "save_reload"]})
                return
            if stop:
                break
        if best_file is None:
            raise RuntimeError("No committed best checkpoint")
        shutil.copyfile(out / best_file, out / "best.pth")
        atomic_json(out / "train_complete.json", {"identity": identity, "epochs": len(history),
            "best_epoch": stopper.best_epoch, "best_validation_dice": stopper.best,
            "best_sha256": sha256_file(out / "best.pth"), "test_evaluated": False,
            "timing_note": "Per-session measured epochs only; crashes/replayed work retained separately, not imputed"})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("pilot", "train", "evaluate"))
    p.add_argument("--cell", required=True, choices=tuple(CELLS))
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)
    for name in ("data-root", "raw-root", "dino-repo", "weights", "output-root"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
