"""Build compact research reports from a completed protocol-v1.0 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SEEDS = (42, 2026, 3407)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def build(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, object]] = []
    patient_rows: list[dict[str, str]] = []
    checkpoint_rows: list[dict[str, object]] = []
    seed_provenance: dict[str, dict] = {}
    seed_configs: dict[str, dict] = {}
    patient_sets: list[set[str]] = []
    for seed in SEEDS:
        seed_dir = experiment_dir / f"seed{seed}"
        efficiency = load_json(seed_dir / "efficiency.json")
        summary = load_json(seed_dir / "test_summary.json")
        provenance = load_json(seed_dir / "provenance.json")
        config = load_json(seed_dir / "config.json")
        seed_provenance[str(seed)] = provenance
        seed_configs[str(seed)] = config
        if provenance.get("git_dirty") is not False:
            raise ValueError(f"Seed {seed} was not run from a clean Git worktree.")
        if int(efficiency.get("seed", -1)) != seed or int(summary.get("seed", -1)) != seed:
            raise ValueError(f"Seed metadata mismatch in seed{seed}.")
        if int(summary.get("num_patients", -1)) != 31:
            raise ValueError(f"Seed {seed} test summary does not contain 31 patients.")
        row = {
            "method": efficiency["method"],
            "seed": seed,
            "best_epoch": efficiency["best_epoch"],
            "best_val_dice_3d": efficiency["best_val_dice_3d_patient_mean"],
            "test_dice_3d": summary["dice_3d"]["mean"],
            "test_iou_3d": summary["iou_3d"]["mean"],
            "test_hd95_header_units": summary["hd95_header_units"]["mean"],
            "total_parameters": efficiency["total_parameters"],
            "trainable_parameters": efficiency["trainable_parameters"],
            "physical_batch_size": efficiency["physical_batch_size"],
            "gradient_accumulation_steps": efficiency["gradient_accumulation_steps"],
            "effective_batch_size": efficiency["effective_batch_size"],
            "shared_gpu_execution": bool(efficiency.get("shared_gpu_execution", False)),
            "skipped_optimizer_updates": efficiency["skipped_optimizer_updates"],
            "peak_vram_gb": efficiency["peak_vram_gb"],
            "training_minutes": efficiency["training_seconds"] / 60,
            "inference_ms_per_slice": efficiency["test_inference_ms_per_slice"],
        }
        seed_rows.append(row)
        with (seed_dir / "test_patient_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 31:
            raise ValueError(f"Expected 31 test rows for seed {seed}, got {len(rows)}")
        if any(not np.isfinite(float(row[key])) for row in rows for key in (
            "dice_3d", "iou_3d", "hd95_header_units"
        )):
            raise ValueError(f"Seed {seed} contains a non-finite test metric.")
        patient_sets.append({row["patient_id"] for row in rows})
        patient_rows.extend(rows)
        for name in ("best.pth", "last.pth"):
            path = seed_dir / name
            checkpoint_rows.append(
                {
                    "seed": seed,
                    "checkpoint": str(path.relative_to(experiment_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "seed_metrics.csv", seed_rows)
    write_csv(output_dir / "patient_metrics.csv", patient_rows)
    write_csv(output_dir / "checkpoint_manifest.csv", checkpoint_rows)
    efficiency_rows = [
        {
            key: row[key]
            for key in (
                "method",
                "seed",
                "best_epoch",
                "best_val_dice_3d",
                "total_parameters",
                "trainable_parameters",
                "physical_batch_size",
                "gradient_accumulation_steps",
                "effective_batch_size",
                "shared_gpu_execution",
                "skipped_optimizer_updates",
                "peak_vram_gb",
                "training_minutes",
                "inference_ms_per_slice",
            )
        }
        for row in seed_rows
    ]
    write_csv(output_dir / "efficiency_metrics.csv", efficiency_rows)

    first_config = seed_configs[str(SEEDS[0])]
    first_provenance = seed_provenance[str(SEEDS[0])]
    if any(values != patient_sets[0] for values in patient_sets[1:]):
        raise ValueError("Test patient sets differ between training seeds.")
    if any(
        provenance.get("split_sha256") != first_provenance.get("split_sha256")
        for provenance in seed_provenance.values()
    ):
        raise ValueError("Split hashes differ between training seeds.")
    shared_config = {key: value for key, value in first_config.items() if key != "seed"}
    if any(
        {key: value for key, value in config.items() if key != "seed"} != shared_config
        for config in seed_configs.values()
    ):
        raise ValueError("Experiment configuration differs between training seeds.")
    # JSON is valid YAML 1.2 and avoids an additional runtime dependency.
    (output_dir / "config.yaml").write_text(
        json.dumps(shared_config, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "training_seeds": list(SEEDS),
                "seed_provenance": seed_provenance,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "split_hashes.json").write_text(
        json.dumps(first_provenance["split_sha256"], indent=2) + "\n",
        encoding="utf-8",
    )
    aggregate = {
        "method": seed_rows[0]["method"],
        "training_seeds": list(SEEDS),
        "split_seed": 2024,
        "num_test_cases_per_seed": 31,
        "num_case_seed_records": len(patient_rows),
        # Retain the legacy keys so existing table-building scripts can read
        # newly generated E3 reports without a schema migration.
        "num_test_patients_per_seed": 31,
        "num_patient_seed_records": len(patient_rows),
        "aggregation": "mean and sample SD across the three seed-level case means",
        "dice_3d": stats([float(row["test_dice_3d"]) for row in seed_rows]),
        "iou_3d": stats([float(row["test_iou_3d"]) for row in seed_rows]),
        "hd95_header_units": stats(
            [float(row["test_hd95_header_units"]) for row in seed_rows]
        ),
        "peak_vram_gb": stats([float(row["peak_vram_gb"]) for row in seed_rows]),
        "training_minutes": stats(
            [float(row["training_minutes"]) for row in seed_rows]
        ),
        "total_parameters": int(seed_rows[0]["total_parameters"]),
        "trainable_parameters": int(seed_rows[0]["trainable_parameters"]),
        "physical_batch_size": int(seed_rows[0]["physical_batch_size"]),
        "gradient_accumulation_steps": int(seed_rows[0]["gradient_accumulation_steps"]),
        "effective_batch_size": int(seed_rows[0]["effective_batch_size"]),
        "shared_gpu_execution": any(
            bool(row["shared_gpu_execution"]) for row in seed_rows
        ),
        "skipped_optimizer_updates_by_seed": {
            str(row["seed"]): int(row["skipped_optimizer_updates"])
            for row in seed_rows
        },
        "hd95_warning": (
            "Exploratory header-unit result: local NRRD spacing provenance remains unresolved."
        ),
        "run_status": "VALID",
    }
    if first_config.get("experiment") == "e7":
        aggregate["run_status"] = "COMPLETE_CASE_LEVEL_WITH_LIMITATIONS"
        aggregate["patient_independence_verified"] = False
        aggregate["protocol_extension"] = first_config["protocol_extension"]
        aggregate["limitation"] = "Case-ID separation does not prove patient independence; report near-duplicate sensitivity."
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    vietnamese = f"""# {aggregate['method']} — Utah2018 Run2

- Protocol: Unified Experiment Protocol v1.0
- Dataset: Utah2018, 154 LGE-MRI volumes; patient independence unverified
- Status: {aggregate['run_status']} (does not establish patient independence)
- Split: 108/15/31 volumes theo fixed case-level split, split seed 2024
- Training seeds: 42, 2026, 3407
- Test records: {aggregate['num_case_seed_records']} case-seed
- 3D Dice: {aggregate['dice_3d']['mean']:.4f} ± {aggregate['dice_3d']['sd']:.4f}
- 3D IoU: {aggregate['iou_3d']['mean']:.4f} ± {aggregate['iou_3d']['sd']:.4f}
- HD95 (header units, exploratory): {aggregate['hd95_header_units']['mean']:.4f} ± {aggregate['hd95_header_units']['sd']:.4f}
- Trainable parameters: {aggregate['trainable_parameters']:,}
- Physical batch size: {aggregate['physical_batch_size']}
- Gradient accumulation steps: {aggregate['gradient_accumulation_steps']}
- Effective batch size: {aggregate['effective_batch_size']}
- Shared-GPU execution: {aggregate['shared_gpu_execution']} (if True, do not compare training time with exclusive-GPU runs)
- Optimizer updates skipped by GradScaler, per seed: {aggregate['skipped_optimizer_updates_by_seed']}
- Peak VRAM: {aggregate['peak_vram_gb']['mean']:.3f} ± {aggregate['peak_vram_gb']['sd']:.3f} GB
- Training time: {aggregate['training_minutes']['mean']:.1f} ± {aggregate['training_minutes']['sd']:.1f} min

The checkpoint is selected by mean case-level 3D Dice on the 15 validation volumes. The test set is evaluated only with each seed's best validation checkpoint.
"""
    (output_dir / "RESULTS_SUMMARY_VI.md").write_text(vietnamese, encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
