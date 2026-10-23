"""Train protocol-v1.0 experiments on the shared Utah2018 split.

The current entry points are E0 Large-Frozen+BCE, E1 Large-Frozen+BCE+Dice,
E2 Large-Full-FT+BCE+Dice, E3 Large-Partial-FT+BCE+Dice, E4 Giant-LoRA,
E5 Giant-Frozen+BCE+Dice, and E6 Large-LoRA.  All use the same data audit,
optimizer policy, case-level 3D validation, checkpointing, and
native-volume evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders
from metrics import DiceBCELoss, LegacyDiceBCELoss, dice_per_sample
from model import DINOv2Segmenter
from native_evaluation import evaluate_model_3d, read_patient_ids, write_evaluation


EXPERIMENTS = {
    "e0": {
        "method": "E0_Large_Frozen_BCE",
        "model": "vit_large",
        "adaptation": "frozen",
        "loss": "bce",
    },
    "e1": {
        "method": "E1_Large_Frozen_BCEDice",
        "model": "vit_large",
        "adaptation": "frozen",
        "loss": "bce_dice",
    },
    "e2": {
        "method": "E2_Large_Full_FT_BCEDice",
        "model": "vit_large",
        "adaptation": "full",
        "loss": "bce_dice",
    },
    "e3": {
        "method": "E3_Large_Partial_FT_BCEDice",
        "model": "vit_large",
        "adaptation": "partial",
        "loss": "bce_dice",
    },
    "e4": {
        "method": "E4_Giant_LoRA",
        "model": "vit_giant",
        "adaptation": "lora",
        "loss": "bce_dice",
    },
    "e5": {
        "method": "E5_Giant_Frozen_BCEDice",
        "model": "vit_giant",
        "adaptation": "frozen",
        "loss": "bce_dice",
    },
    "e7": {
        "method": "E7_Giant_Partial_FT_BCEDice",
        "model": "vit_giant",
        "adaptation": "partial",
        "loss": "bce_dice",
    },
    "e6": {
        "method": "E6_Large_LoRA",
        "model": "vit_large",
        "adaptation": "lora",
        "loss": "bce_dice",
    },
}

EXPECTED_SPLIT_SHA256 = {
    "train": "2f2b6c441f2646e0a06e8ea470bdd96f7864f4f6706f42b71b1eb3bd8ec49316",
    "val": "56677d26b6f8a16ad3f6239ec7043c5fd0caaca4adae0334447d8f9242ff7456",
    "test": "2e59b4ca4c0ad393bdb456b6dc2f2fcbab7c7727fb7012806d49e84ef394cae8",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_output(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_protocol_data(data_root: Path) -> tuple[dict[str, list[str]], dict]:
    """Validate the immutable Utah2018 v1.0 split and processed-data manifest."""
    patient_ids = {
        name: read_patient_ids(data_root / f"{name}_patients.txt")
        for name in ("train", "val", "test")
    }
    counts = {name: len(values) for name, values in patient_ids.items()}
    if counts != {"train": 108, "val": 15, "test": 31}:
        raise ValueError(f"Expected fixed 108/15/31 split, got {counts}.")
    patient_sets = {name: set(values) for name, values in patient_ids.items()}
    if (
        patient_sets["train"] & patient_sets["val"]
        or patient_sets["train"] & patient_sets["test"]
        or patient_sets["val"] & patient_sets["test"]
    ):
        raise ValueError("Patient overlap detected between train/validation/test.")
    if len(set().union(*patient_sets.values())) != 154:
        raise ValueError("The fixed split must contain 154 unique patient IDs.")
    split_hashes = {
        name: sha256_file(data_root / f"{name}_patients.txt")
        for name in ("train", "val", "test")
    }
    if split_hashes != EXPECTED_SPLIT_SHA256:
        raise ValueError(
            "Split hashes do not match Unified Experiment Protocol v1.0: "
            f"{split_hashes}"
        )

    provenance = json.loads(
        (data_root / "dataset_provenance.json").read_text(encoding="utf-8")
    )
    preprocessing = provenance.get("preprocessing", {})
    expected_preprocessing = {
        "normalization_scope": "per_patient_3d_volume",
        "percentiles": [0.5, 99.5],
        "exclude_exact_zero_background_when_mixed": True,
        "image_size": 448,
        "image_interpolation": "bilinear",
        "mask_interpolation": "nearest",
        "augmentation": "none",
    }
    if provenance.get("dataset") != "Utah2018":
        raise ValueError("Processed data is not identified as Utah2018.")
    if provenance.get("preprocessing_version") != "utah2018_run2_v1.0":
        raise ValueError("Unexpected preprocessing version.")
    if provenance.get("split_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError("Dataset provenance records different split hashes.")
    if preprocessing != expected_preprocessing:
        raise ValueError(
            "Processed-data configuration differs from protocol v1.0: "
            f"{preprocessing}"
        )

    manifest_path = data_root / "preprocessing_manifest.csv"
    per_patient: dict[tuple[str, str], int] = {}
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 154 * 88:
        raise ValueError(f"Expected 13,552 processed slices, got {len(rows)}.")
    for row in rows:
        key = (row["split"], row["patient_id"])
        per_patient[key] = per_patient.get(key, 0) + 1
    if any(count != 88 for count in per_patient.values()) or len(per_patient) != 154:
        raise ValueError("Every Utah2018 patient must have exactly 88 processed slices.")
    for split, ids in patient_ids.items():
        if {patient_id for row_split, patient_id in per_patient if row_split == split} != set(ids):
            raise ValueError(f"Manifest patient IDs do not match the {split} split.")
    return patient_ids, provenance


def dinov2_hub_revision() -> str:
    hub_root = Path(torch.hub.get_dir())
    candidates = sorted(hub_root.glob("facebookresearch_dinov2_*"))
    for candidate in candidates:
        try:
            commit = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            return f"git:{commit}"
        except (OSError, subprocess.CalledProcessError):
            python_files = sorted(candidate.rglob("*.py"))
            if python_files:
                digest = hashlib.sha256()
                for path in python_files:
                    relative = path.relative_to(candidate).as_posix().encode("utf-8")
                    digest.update(relative)
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                return f"python-tree-sha256:{digest.hexdigest()}"
    return "unknown"


def pretrained_weights_provenance(model_name: str) -> dict[str, object]:
    hub_root = Path(torch.hub.get_dir())
    hub_names = {
        "vit_large": "dinov2_vitl14",
        "vit_giant": "dinov2_vitg14",
    }
    hub_name = hub_names[model_name]
    candidates = sorted(hub_root.rglob(f"{hub_name}*pretrain*.pth"))
    if not candidates:
        return {"hub_name": hub_name, "path": "unknown", "sha256": "unknown"}
    path = candidates[0]
    return {
        "hub_name": hub_name,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_model(experiment: str, rank: int, alpha: float, pretrained: bool = True):
    config = EXPERIMENTS[experiment]
    if not pretrained and config["adaptation"] != "full":
        raise ValueError("--random_init is only meaningful with full fine-tuning (e2)")
    model = DINOv2Segmenter(model_name=config["model"], pretrained=pretrained)
    lora_layers: tuple[str, ...] = ()
    partial_trainable_blocks: tuple[int, ...] = ()
    if config["adaptation"] == "full":
        model.configure_full_finetuning()
    elif config["adaptation"] == "partial":
        partial_trainable_blocks = model.configure_partial_finetuning(num_blocks=4)
        if experiment == "e7" and partial_trainable_blocks != (36, 37, 38, 39):
            raise ValueError("E7 requires a 40-block Giant with only blocks 36--39 unfrozen.")
    elif config["adaptation"] == "lora":
        lora_layers = model.configure_lora(rank=rank, alpha=alpha)
    return model, lora_layers, partial_trainable_blocks


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    """Return the compact adapter/decoder state required for exact inference.

    Decoder BatchNorm running statistics are buffers rather than trainable
    parameters.  They must still be captured with the best checkpoint;
    otherwise reloading best weights after later epochs mixes weights and
    BatchNorm state from different epochs.
    """
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    all_parameters_trainable = all(
        parameter.requires_grad for parameter in model.parameters()
    )
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if all_parameters_trainable
        or name in trainable_names
        or name.startswith("decoder.")
    }


def load_trainable_state_dict(model, state: dict[str, torch.Tensor]) -> None:
    named_parameters = dict(model.named_parameters())
    model_state = model.state_dict()
    trainable_names = {
        name
        for name, parameter in named_parameters.items()
        if parameter.requires_grad
    }
    all_parameters_trainable = all(
        parameter.requires_grad for parameter in model.parameters()
    )
    expected = sorted(
        name
        for name in model_state
        if all_parameters_trainable
        or name in trainable_names
        or name.startswith("decoder.")
    )
    missing = sorted(set(state) - set(model_state))
    absent = sorted(set(expected) - set(state))
    if missing or absent:
        raise RuntimeError(f"Trainable checkpoint mismatch: unknown={missing}, absent={absent}")
    with torch.no_grad():
        for name, value in state.items():
            model_state[name].copy_(value)


def optimizer_groups(model, lr_adapter: float, lr_decoder: float, weight_decay: float):
    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    names: dict[tuple[float, float], list[str]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        learning_rate = lr_decoder if name.startswith("decoder.") else lr_adapter
        no_decay = parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower()
        decay = 0.0 if no_decay else weight_decay
        groups.setdefault((learning_rate, decay), []).append(parameter)
        names.setdefault((learning_rate, decay), []).append(name)
    if not groups:
        raise RuntimeError("Experiment has no trainable parameters.")
    return [
        {"params": parameters, "lr": lr, "weight_decay": decay}
        for (lr, decay), parameters in groups.items()
    ], {f"lr={lr:g},wd={decay:g}": value for (lr, decay), value in names.items()}


def _audit_successful_gradients(model) -> None:
    missing = []
    nonfinite = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            # DINOv2's mask token is trainable in E2 but unused by an ordinary
            # unmasked forward pass, so it correctly has no gradient here.
            if parameter.grad is None:
                if not name.endswith("mask_token"):
                    missing.append(name)
            elif not torch.isfinite(parameter.grad).all():
                nonfinite.append(name)
        elif parameter.grad is not None:
            raise RuntimeError(f"Gradient leaked into frozen parameter: {name}")
    if missing or nonfinite:
        raise RuntimeError(
            f"Gradient audit failed: missing={missing[:8]}, nonfinite={nonfinite[:8]}"
        )


def _assert_finite_trainable_parameters(model) -> None:
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not torch.isfinite(parameter).all()
    ]
    if nonfinite:
        raise FloatingPointError(f"Non-finite trainable parameters: {nonfinite[:8]}")


def train_epoch(model, loader, criterion, optimizer, scaler, device, gradient_audit,
                grad_clip_norm: float = 0.0):
    model.train()
    loss_sum = dice_sum = 0.0
    sample_count = 0
    skipped_optimizer_updates = 0
    for batch_index, (images, masks, _) in enumerate(tqdm(loader, desc="train", leave=False)):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite training loss at batch {batch_index}")
        scale_before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            # Unscale first so the norm is measured on true gradients; the
            # scaler still skips the step if any gradient is non-finite.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [q for q in model.parameters() if q.requires_grad], grad_clip_norm
            )
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        step_skipped = scale_after < scale_before
        if step_skipped:
            skipped_optimizer_updates += 1
        elif not gradient_audit["completed"]:
            _audit_successful_gradients(model)
            gradient_audit["completed"] = True
        with torch.no_grad():
            prediction = torch.sigmoid(logits) >= 0.5
            batch_dice = dice_per_sample(prediction, masks >= 0.5).sum().item()
        batch_size = images.shape[0]
        loss_sum += float(loss.item()) * batch_size
        dice_sum += batch_dice
        sample_count += batch_size
    _assert_finite_trainable_parameters(model)
    return loss_sum / sample_count, dice_sum / sample_count, skipped_optimizer_updates


def write_history(path: Path, history: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        # Resume may restore a row saved before its final checkpoint duration existed.
        fields = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    path: Path,
    model,
    epoch: int,
    best_val_dice: float,
    args,
    metadata: dict,
    optimizer=None,
    scaler=None,
    extra: dict | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "experiment": args.experiment,
        "method": EXPERIMENTS[args.experiment]["method"],
        "model": EXPERIMENTS[args.experiment]["model"],
        "adaptation": EXPERIMENTS[args.experiment]["adaptation"],
        "backbone_init": "random" if args.random_init else "dinov2_pretrained",
        "seed": args.seed,
        "epoch": epoch,
        "best_val_dice": best_val_dice,
        "lora_rank": (
            args.lora_rank
            if EXPERIMENTS[args.experiment]["adaptation"] == "lora"
            else None
        ),
        "lora_alpha": (
            args.lora_alpha
            if EXPERIMENTS[args.experiment]["adaptation"] == "lora"
            else None
        ),
        "trainable_state_dict": trainable_state_dict(model),
        "training_args": vars(args),
        "run_metadata": metadata,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra:
        payload.update(extra)
    atomic_save(payload, path)


def atomic_save(payload: dict, path: Path) -> None:
    """Write to a temporary file, flush it to disk, then rename atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def reconcile_pending_best(output_dir: Path, committed_best_epoch: int) -> str:
    """Resolve a best-checkpoint transaction interrupted before completion.

    best.pth.pending is promoted only if last.pth already committed that epoch
    as best; otherwise it belongs to an uncommitted epoch and is discarded.
    """
    pending = output_dir / "best.pth.pending"
    if not pending.exists():
        return "none"
    pending_epoch = int(torch.load(pending, map_location="cpu", weights_only=False)["epoch"])
    if pending_epoch == committed_best_epoch:
        os.replace(pending, output_dir / "best.pth")
        return "promoted"
    pending.unlink()
    return "discarded"


def run(args: argparse.Namespace) -> None:
    if args.seed not in {42, 2026, 3407}:
        raise ValueError("Training seed must be one of 42, 2026, 3407.")
    data_root = Path(args.data_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for required in (
        data_root / "train_patients.txt",
        data_root / "val_patients.txt",
        data_root / "test_patients.txt",
        data_root / "patient_meta.json",
        data_root / "dataset_provenance.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    patient_ids, data_provenance = audit_protocol_data(data_root)
    train_ids = patient_ids["train"]
    val_ids = patient_ids["val"]
    test_ids = patient_ids["test"]
    patient_meta = json.loads((data_root / "patient_meta.json").read_text(encoding="utf-8"))
    if set(patient_meta) != set(train_ids + val_ids + test_ids):
        raise ValueError("patient_meta.json does not match the fixed 154-patient split.")
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("A CUDA GPU is required for a reported protocol run.")

    train_case_ids = None
    train_cases_sha256 = None
    if args.train_cases:
        train_cases_path = Path(args.train_cases).resolve()
        train_case_ids = read_patient_ids(train_cases_path)
        if not set(train_case_ids) <= set(train_ids) or len(set(train_case_ids)) != len(train_case_ids):
            raise ValueError("--train_cases must list unique cases from the fixed training split.")
        train_cases_sha256 = sha256_file(train_cases_path)
    loaders = get_dataloaders(
        data_root=str(data_root), batch_size=args.batch_size, num_workers=args.num_workers,
        train_case_ids=train_case_ids,
    )
    if set(loaders) != {"train", "val", "test"}:
        raise RuntimeError(f"Incomplete data loaders: {sorted(loaders)}")
    model, lora_layers, partial_trainable_blocks = build_model(
        args.experiment, args.lora_rank, args.lora_alpha, pretrained=not args.random_init
    )
    model.to(device)
    groups, group_names = optimizer_groups(
        model, args.lr_adapter, args.lr_decoder, args.weight_decay
    )
    optimizer = torch.optim.AdamW(
        groups, betas=(0.9, 0.999), eps=1e-8
    )
    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if EXPERIMENTS[args.experiment]["loss"] == "bce"
        else (LegacyDiceBCELoss if args.loss_impl == "legacy" else DiceBCELoss)(
            bce_weight=0.5, dice_weight=0.5
        )
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    total_parameters = model.num_total_params()
    trainable_parameters = model.num_trainable_params()
    metadata = {
        "git_branch": git_output("branch", "--show-current"),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": git_output("status", "--porcelain") not in {"", "unknown"},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "dinov2_source_revision": dinov2_hub_revision(),
        "pretrained_weights": pretrained_weights_provenance(
            EXPERIMENTS[args.experiment]["model"]
        ),
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "preprocessing_provenance_sha256": sha256_file(
            data_root / "dataset_provenance.json"
        ),
    }
    configuration = {
        "protocol": "Unified Experiment Protocol v1.0",
        "dataset": "Utah2018, 154 LGE-MRI volumes; patient independence unverified",
        "split": "108/15/31 fixed case-level split, seed 2024",
        "training_seeds": [42, 2026, 3407],
        "experiment": args.experiment,
        "method": EXPERIMENTS[args.experiment]["method"],
        "model": EXPERIMENTS[args.experiment]["model"],
        "adaptation": EXPERIMENTS[args.experiment]["adaptation"],
        "backbone_init": "random" if args.random_init else "dinov2_pretrained",
        "loss": (
            "BCEWithLogitsLoss"
            if EXPERIMENTS[args.experiment]["loss"] == "bce"
            else "0.5 BCE + 0.5 soft Dice"
        ),
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_epsilon": 1e-8,
        "scheduler": "none",
        "optimizer_groups": group_names,
        "batch_size": args.batch_size,
        "physical_batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": args.batch_size,
        "shared_gpu_execution": os.environ.get("ALLOW_SHARED_GPU", "0") == "1",
        "validation_batch_size": args.batch_size,
        "test_batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "augmentation": "none",
        "postprocessing": "none",
        "threshold": 0.5,
        "preprocessing_version": data_provenance["preprocessing_version"],
        "amp_dtype": "float16",
        "loss_reduction_dtype": "float32" if args.loss_impl == "fp32" else "autocast (historical E1/E5/E6 path)",
        "loss_impl": args.loss_impl,
        "grad_clip_norm": args.grad_clip_norm,
        "train_cases_file": str(Path(args.train_cases).resolve()) if args.train_cases else None,
        "train_cases_sha256": train_cases_sha256,
        "num_train_cases": len(train_case_ids) if train_case_ids is not None else len(train_ids),
        "protocol_tag": args.protocol_tag,
        "amp_grad_scaler": "PyTorch dynamic defaults",
        "checkpoint_selection": "unweighted mean case-level 3D Dice on 15 validation volumes",
        "max_epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "lora_rank": (
            args.lora_rank
            if EXPERIMENTS[args.experiment]["adaptation"] == "lora"
            else None
        ),
        "lora_alpha": (
            args.lora_alpha
            if EXPERIMENTS[args.experiment]["adaptation"] == "lora"
            else None
        ),
        "lora_layers": list(lora_layers),
        "partial_trainable_blocks": list(partial_trainable_blocks),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "data_root": str(data_root),
        "raw_root": str(raw_root),
    }
    if args.experiment == "e7":
        configuration["protocol_extension"] = "E7 Giant Partial FT: last four of 40 blocks"
        configuration["evaluation_unit"] = "volume (case), not patient aggregation"
        configuration["patient_independence_verified"] = False
        configuration["known_near_duplicate_test_case"] = "CMPXO4J23G58J53Q98SZ"
    # Refuse to overwrite an existing run; resuming requires identical config/provenance.
    config_path = output_dir / "config.json"
    if config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if not args.resume or previous_config != configuration:
            raise ValueError("Output already exists with a different configuration, or --resume is missing.")
        prior_metadata = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
        for key in ("git_commit", "torch", "cuda", "gpu", "dinov2_source_revision", "split_sha256"):
            if metadata[key] != prior_metadata[key]:
                raise ValueError(f"Resume provenance changed: {key}")
        if metadata["pretrained_weights"]["sha256"] != prior_metadata["pretrained_weights"]["sha256"]:
            raise ValueError("Resume pretrained weights changed")
        if (output_dir / "best.pth").exists() and not (output_dir / "last.pth").exists():
            raise ValueError("best.pth exists but last.pth is missing; inspect before restarting.")
    (output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    history: list[dict[str, object]] = []
    best_val_dice = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    previous_training_seconds = 0.0
    previous_peak_vram_gb = 0.0
    skipped_optimizer_updates = 0
    gradient_audit = {"completed": False}
    last_path = output_dir / "last.pth"
    best_path = output_dir / "best.pth"
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("experiment") != args.experiment or checkpoint.get("seed") != args.seed:
            raise ValueError("Resume checkpoint experiment/seed mismatch")
        load_trainable_state_dict(model, checkpoint["trainable_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint.get("history", [])
        best_val_dice = float(checkpoint["best_val_dice"])
        best_epoch = int(checkpoint.get("best_epoch", 0))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        previous_training_seconds = float(
            checkpoint.get("training_seconds_completed", 0.0)
        )
        previous_peak_vram_gb = float(checkpoint.get("training_peak_vram_gb", 0.0))
        skipped_optimizer_updates = int(
            checkpoint.get("skipped_optimizer_updates", 0)
        )
        gradient_audit["completed"] = bool(
            checkpoint.get("gradient_audit_completed", False)
        )
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        resolution = reconcile_pending_best(output_dir, best_epoch)
        if best_epoch and not best_path.exists():
            raise ValueError("Committed best epoch has no best.pth; inspect before resuming.")
        resume_log = output_dir / "resume_events.jsonl"
        with resume_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "resumed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "last_committed_epoch": int(checkpoint["epoch"]),
                "best_epoch": best_epoch,
                "pending_best": resolution,
            }) + "\n")
        if stale_epochs >= args.patience:
            start_epoch = args.epochs + 1
        print(f"Resuming from epoch {start_epoch}")
        del checkpoint

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    training_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_phase_started = time.perf_counter()
        train_loss, train_dice, skipped_this_epoch = train_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            gradient_audit,
            grad_clip_norm=args.grad_clip_norm,
        )
        skipped_optimizer_updates += skipped_this_epoch
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_phase_seconds = time.perf_counter() - train_phase_started
        val_records, val_summary, val_loss = evaluate_model_3d(
            model,
            loaders["val"],
            device,
            val_ids,
            patient_meta,
            raw_root,
            EXPERIMENTS[args.experiment]["method"],
            args.seed,
            criterion=criterion,
            compute_hd95=False,
        )
        val_dice = float(val_summary["dice_3d"]["mean"])
        improved = val_dice > best_val_dice
        if improved:
            best_val_dice = val_dice
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(output_dir / "best.pth.pending", model, epoch, best_val_dice, args, metadata)
            write_evaluation(val_records, val_summary, output_dir, "val")
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice_slice_mean": train_dice,
                "val_loss": val_loss,
                "val_dice_3d_patient_mean": val_dice,
                "best_val_dice": best_val_dice,
                "improved": improved,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "train_phase_seconds": train_phase_seconds,
                "train_slices_per_second": len(loaders["train"].dataset) / train_phase_seconds,
                "training_seconds_to_selection": previous_training_seconds + time.perf_counter() - training_started,
                "skipped_optimizer_updates": skipped_this_epoch,
                "amp_scale": float(scaler.get_scale()),
            }
        )
        write_history(output_dir / "history.csv", history)
        save_checkpoint(
            last_path,
            model,
            epoch,
            best_val_dice,
            args,
            metadata,
            optimizer,
            scaler,
            extra={
                "history": history,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "training_seconds_completed": previous_training_seconds
                + (time.perf_counter() - training_started),
                "rng_state": rng_state(),
                "skipped_optimizer_updates": skipped_optimizer_updates,
                "gradient_audit_completed": gradient_audit["completed"],
                "training_peak_vram_gb": max(previous_peak_vram_gb, torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0,
            },
        )
        if improved:
            os.replace(output_dir / "best.pth.pending", best_path)
        # Includes validation and both checkpoint writes; unlike epoch_seconds above.
        history[-1]["complete_epoch_seconds"] = time.perf_counter() - epoch_started
        write_history(output_dir / "history.csv", history)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss={train_loss:.4f} "
            f"slice Dice={train_dice:.4f} | val loss={val_loss:.4f} "
            f"case-level 3D Dice={val_dice:.4f} | best={best_val_dice:.4f} "
            f"(epoch {best_epoch})"
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epochs without improvement.")
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    if not gradient_audit["completed"]:
        raise RuntimeError("No successful optimizer step completed the gradient audit.")
    training_seconds = previous_training_seconds + (time.perf_counter() - training_started)
    training_peak_vram_gb = (
        max(previous_peak_vram_gb, torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0
    )
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    load_trainable_state_dict(model, best_checkpoint["trainable_state_dict"])
    test_records, test_summary, _ = evaluate_model_3d(
        model,
        loaders["test"],
        device,
        test_ids,
        patient_meta,
        raw_root,
        EXPERIMENTS[args.experiment]["method"],
        args.seed,
        threshold=0.5,
        compute_hd95=True,
        output_dir=output_dir,
    )
    write_evaluation(test_records, test_summary, output_dir, "test")
    efficiency = {
        "method": EXPERIMENTS[args.experiment]["method"],
        "seed": args.seed,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_ratio_percent": trainable_parameters / total_parameters * 100,
        "physical_batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": args.batch_size,
        "shared_gpu_execution": os.environ.get("ALLOW_SHARED_GPU", "0") == "1",
        "num_workers": args.num_workers,
        "skipped_optimizer_updates": skipped_optimizer_updates,
        "gradient_audit_completed": gradient_audit["completed"],
        "peak_vram_gb": training_peak_vram_gb,
        "training_seconds": training_seconds,
        "training_gpu_hours_to_best_checkpoint": next(
            (row.get("training_seconds_to_selection", None) / 3600
             for row in history if row["epoch"] == best_epoch
             and row.get("training_seconds_to_selection") is not None), None
        ),
        "timing_scope": "train_phase includes data loading and optimizer; time-to-best includes preceding epochs and selected best save; complete_epoch includes validation and best/last saves",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_val_dice_3d_patient_mean": best_val_dice,
        "test_inference_seconds": test_summary["inference_total_seconds"],
        "test_inference_ms_per_slice": test_summary["inference_ms_per_slice"],
    }
    (output_dir / "efficiency.json").write_text(
        json.dumps(efficiency, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"efficiency": efficiency, "test": test_summary}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data_root", default="data/utah2018_run2")
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_adapter", type=float, default=1e-5)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=4.0)
    parser.add_argument("--loss_impl", choices=["fp32", "legacy"], default="fp32",
                        help="fp32: explicit FP32 loss reductions (current); legacy: historical autocast path")
    parser.add_argument("--grad_clip_norm", type=float, default=0.0,
                        help="Clip the global gradient norm of trainable parameters; 0 disables")
    parser.add_argument("--train_cases", default=None,
                        help="Optional file of training case IDs (subset of the fixed split)")
    parser.add_argument("--protocol_tag", default="",
                        help="Free-text tag recorded in config.json for revision runs")
    parser.add_argument("--random_init", action="store_true",
                        help="Same ViT architecture with random initialisation instead of DINOv2 weights (e2 only)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
