"""Augmentation-study runner: label-efficiency runner with nnU-Net-style augmentation of training slices.

Run as python -m src.train_augmentation_study. Pilots never evaluate test. Each completed run is
evaluated on test once. The protocol (src/augmentation_study.py) is fixed before the first run
and is not changed afterwards. Use one output root per cell so that two cells can run side by
side on one GPU (the study lock is per output root).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import torch
from torch.utils.data import DataLoader

from .augmentation_study import (BATCH_SIZE, CELLS, EXPECTED_TRAINABLE, FRACTIONS, MIN_FREE_VRAM_AT_START,
                                 PILOT_FRACTION, PROTOCOL, RUNS, SEEDS, STEPS_PER_EPOCH, AUG_STREAM,
                                 augment_batch, encoder_lr)
from .dataset import LeftAtriumDataset
from .label_efficiency_study import SLICES_PER_VOLUME, all_subsets, random_backbone
from .native_evaluation import write_evaluation
from .protocol_v2_study import LR_DECODER, ProtocolV2Segmenter
from .readout_study import EarlyStop, load_backbone
from .train_e5_e6 import (audit_protocol_data, optimizer_groups, trainable_state_dict, load_trainable_state_dict,
                          sha256_file, set_seed, restore_rng_state)
from .train_protocol_v2 import (atomic_checkpoint, atomic_json, checkpoint, digest_json, evaluate,
                                foreign_gpu_processes, gpu_guard as v2_gpu_guard, loader, parity_checks,
                                provenance as v2_provenance, study_lock, train_epoch, verify_completion,
                                verify_evaluation)

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    paths = list((ROOT / "src").glob("*.py"))
    paths += list((ROOT / "scripts").glob("*augmentation*"))
    paths += list((ROOT / "tests").glob("*augmentation*.py"))
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}


def provenance(args):
    prov = v2_provenance(args)
    prov.update(protocol=PROTOCOL, source_sha256=source_hashes(),
                subsets_sha256=sha256_file(Path(args.subsets)),
                extension_status="augmentation study; pre-specified 2026-11-10 before any augmented run; post hoc")
    return prov


def gpu_guard(root, start=False):
    """Hardware and disk checks of the v2 runner; a lower free-memory floor lets two cells share the GPU."""
    name = v2_gpu_guard(root, start=False)
    if start and torch.cuda.mem_get_info()[0] < MIN_FREE_VRAM_AT_START:
        raise RuntimeError(f"Insufficient free GPU memory; foreign jobs untouched: {foreign_gpu_processes()}")
    return name


def load_subset(args, train_ids):
    stored = json.loads(Path(args.subsets).read_text(encoding="utf-8"))
    if stored.get("subsets") != all_subsets(train_ids):
        raise RuntimeError("Subset file differs from the pre-specified subset rule")
    return stored["subsets"][str(args.seed)][str(args.fraction)]


def augmented_batches(data, case_ids, seed, epoch, steps=STEPS_PER_EPOCH):
    """Same batch order as the label-efficiency runner; each batch is then augmented on the GPU."""
    dataset = LeftAtriumDataset(str(data / "train_2d"), augment=False, case_ids=case_ids)
    if len(dataset) != len(case_ids) * SLICES_PER_VOLUME:
        raise RuntimeError("Missing/extra slices for the training subset")
    produced, sweep = 0, 0
    while True:
        generator = torch.Generator().manual_seed(seed + 30000 + epoch + 1000 * sweep)
        aug_generator = torch.Generator().manual_seed(seed + AUG_STREAM + epoch + 1000 * sweep)
        for images, masks, names in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                               drop_last=True, pin_memory=True, generator=generator):
            images, masks = augment_batch(images.cuda(non_blocking=True), masks.cuda(non_blocking=True), aug_generator)
            yield images, masks, names
            produced += 1
            if produced == steps:
                return
        sweep += 1


def check_pilot(root, cell, fingerprint):
    path = root / "pilot" / cell / f"seed{SEEDS[0]}" / "pilot.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or result.get("fingerprint") != fingerprint:
        raise RuntimeError(f"Pilot is missing/stale/failed for {cell}")


def build_model(args):
    if args.cell == "V_SC":
        model = ProtocolV2Segmenter(random_backbone(Path(args.dino_repo), args.seed), "V_FU", args.seed)
    else:
        model = ProtocolV2Segmenter(load_backbone(Path(args.dino_repo), Path(args.weights)), args.cell, args.seed)
    model = model.cuda()
    if model.num_trainable_params() != EXPECTED_TRAINABLE[args.cell]:
        raise RuntimeError("Unexpected trainable parameter count")
    return model


def run(args):
    root, data = Path(args.output_root).resolve(), Path(args.data_root).resolve()
    if root == ROOT or root.is_relative_to(ROOT / "results/utah2018_run2_master") or root.is_relative_to(data):
        raise ValueError("Use a separate study output directory")
    if args.mode == "pilot":
        if args.seed != SEEDS[0] or args.fraction != PILOT_FRACTION:
            raise ValueError(f"Pilots use seed {SEEDS[0]} at {PILOT_FRACTION}% only")
    elif (args.fraction, args.cell, args.seed) not in RUNS:
        raise ValueError("Run is not part of the pre-specified matrix")
    torch.set_num_threads(4)  # two cells share the CPU; avoid thread oversubscription
    with study_lock(root):
        gpu_guard(root, start=args.mode != "evaluate")
        set_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        ids, _ = audit_protocol_data(data)
        meta = json.loads((data / "patient_meta.json").read_text(encoding="utf-8"))
        case_ids = load_subset(args, ids["train"])
        prov = provenance(args)
        fingerprint = digest_json(prov)
        identity = {"fingerprint": fingerprint, "cell": args.cell, "seed": args.seed, "fraction": args.fraction}
        if args.mode == "pilot":
            out = root / "pilot" / args.cell / f"seed{args.seed}"
        else:
            out = root / "runs" / f"f{args.fraction:03d}" / args.cell / f"seed{args.seed}"
            check_pilot(root, args.cell, fingerprint)
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
                "cell": CELLS[args.cell], "seed": args.seed, "fraction_percent": args.fraction,
                "train_volumes": len(case_ids), "train_cases": case_ids,
                "backbone_init": "random" if args.cell == "V_SC" else "dinov2_pretrained",
                "encoder_lr": encoder_lr(args.cell), "augmentation": True, "data_root": str(data),
                "raw_root": str(Path(args.raw_root).resolve()), "dino_repo": str(Path(args.dino_repo).resolve()),
                "weights": str(Path(args.weights).resolve())})
            atomic_json(out / "provenance.json", prov)
        if args.mode == "evaluate":
            verify_completion(out, identity)
            if (out / "evaluation_complete.json").exists():
                verify_evaluation(out, identity)
                print("SKIP verified completed evaluation")
                return
        model = build_model(args)
        if args.mode == "evaluate":
            state = torch.load(out / "best.pth", map_location="cpu", weights_only=False)
            load_trainable_state_dict(model, state["model"])
            dest = out / "evaluations" / uuid.uuid4().hex
            rows, summary, _ = evaluate(model, args, ids, meta, "test", args.cell, args.seed, dest)
            write_evaluation(rows, summary, dest, "test")
            files = {p.relative_to(out).as_posix(): sha256_file(p) for p in dest.rglob("*") if p.is_file()}
            atomic_json(out / "evaluation_complete.json", {"identity": identity,
                "best_sha256": sha256_file(out / "best.pth"), "directory": dest.relative_to(out).as_posix(), "files": files})
            return
        groups, _ = optimizer_groups(model, encoder_lr(args.cell), LR_DECODER, 0.01)
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        scaler = torch.amp.GradScaler("cuda")
        stopper, history, start_epoch, best_file = EarlyStop(), [], 1, None
        run._committed_best = None
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
            run._committed_best = best_file
            if not (out / best_file).is_file():
                raise RuntimeError("Committed best checkpoint missing")
        session = out / "sessions" / uuid.uuid4().hex
        session.mkdir(parents=True)
        atomic_json(session / "start.json", {"identity": identity, "started_unix": time.time(),
                                           "start_epoch": start_epoch, "pid": os.getpid(),
                                           "foreign_gpu_processes": foreign_gpu_processes()})
        if args.mode == "pilot":
            parity_checks(model, args.seed)
        torch.cuda.reset_peak_memory_stats()
        epochs = [1] if args.mode == "pilot" else range(start_epoch, PROTOCOL["max_epochs"] + 1)
        for epoch in epochs:
            if stopper.stale >= PROTOCOL["patience"]:
                break
            gpu_guard(root)
            stats = train_epoch(model, augmented_batches(data, case_ids, args.seed, epoch), optimizer, scaler,
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
                best_file = (session / f"best_epoch{epoch:03d}.pth").relative_to(out).as_posix()
                atomic_checkpoint(out / best_file, {"model": trainable_state_dict(model),
                    "identity": identity, "epoch": epoch})
            previous_best = run._committed_best
            saved = checkpoint(model, optimizer, scaler, stopper, epoch, identity, history)
            saved["best_file"] = best_file
            atomic_checkpoint(out / "last.pth", saved)
            if previous_best and previous_best != best_file and (out / previous_best).is_file():
                (out / previous_best).unlink()
            run._committed_best = best_file
            stats["checkpoint_seconds"] = time.perf_counter() - save_start
            atomic_json(session / f"epoch{epoch:03d}.json", stats)
            atomic_json(out / "history.json", history)
            print(json.dumps(stats), flush=True)
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
                atomic_json(out / "pilot.json", {"status": "PASS", "fingerprint": fingerprint,
                    "identity": identity, "stats": stats, "test_evaluated": False,
                    "checks": ["readout_parity", "parameter_count", "successful_gradient_audit", "finite_state",
                               "save_reload", "augmented_batches"]})
                for path in out.rglob("*.pth"):
                    path.unlink()
                return
            if stop:
                break
        if best_file is None:
            raise RuntimeError("No committed best checkpoint")
        shutil.copyfile(out / best_file, out / "best.pth")
        atomic_json(out / "train_complete.json", {"identity": identity, "epochs": len(history),
            "best_epoch": stopper.best_epoch, "best_validation_dice": stopper.best,
            "best_sha256": sha256_file(out / "best.pth"), "test_evaluated": False,
            "timing_note": "Shared GPU (two cells side by side); wall-clock timing is not used for reported results"})
        (out / "last.pth").unlink()
        for path in (out / "sessions").rglob("*.pth"):
            path.unlink()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("pilot", "train", "evaluate"))
    p.add_argument("--cell", required=True, choices=tuple(CELLS))
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)
    p.add_argument("--fraction", required=True, type=int, choices=tuple(FRACTIONS))
    for name in ("data-root", "raw-root", "dino-repo", "weights", "subsets", "output-root"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
