"""Shared patient-level 3D validation and test evaluation."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import torch
from PIL import Image

try:
    from .volume_metrics import hd95_mm, overlap_metrics
except ImportError:
    from volume_metrics import hd95_mm, overlap_metrics


def read_patient_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Empty or duplicate patient IDs in {path}")
    return values


def _restore_prediction(prediction: np.ndarray, meta: dict) -> np.ndarray:
    height, width, _ = meta["original_shape"]
    side = int(meta["pad_side"])
    top = int(meta["pad_top"])
    left = int(meta["pad_left"])
    square = np.asarray(
        Image.fromarray(prediction.astype(np.uint8) * 255).resize(
            (side, side), Image.Resampling.NEAREST
        )
    ) > 127
    return square[top : top + height, left : left + width]


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _save_overlay(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    patient_id: str,
    slice_index: int,
    dice: float,
    output_path: Path,
) -> None:
    image = image.astype(np.float32)
    image -= image.min()
    if image.max() > 0:
        image /= image.max()
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("MRI")
    axes[1].imshow(image, cmap="gray")
    axes[1].contour(target, colors="red", linewidths=0.7)
    axes[1].set_title("Ground truth")
    axes[2].imshow(image, cmap="gray")
    axes[2].contour(prediction, colors="lime", linewidths=0.7)
    axes[2].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"{patient_id} | slice {slice_index} | 3D Dice={dice:.4f}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def evaluate_model_3d(
    model,
    loader,
    device: torch.device,
    patient_ids: list[str],
    patient_meta: dict,
    raw_root: Path,
    method: str,
    seed: int,
    criterion=None,
    threshold: float = 0.5,
    compute_hd95: bool = False,
    output_dir: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, object], float]:
    """Predict all slices, restore native volumes, and compute patient metrics."""
    model.eval()
    predictions: dict[str, dict[int, np.ndarray]] = {}
    loss_sum = 0.0
    sample_count = 0
    inference_seconds = 0.0

    # Warm up kernels and allocator outside the reported inference timer.
    warmup_images, _, _ = next(iter(loader))
    warmup_images = warmup_images.to(device, non_blocking=True)
    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        model(warmup_images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    del warmup_images

    for images, masks, filenames in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks) if criterion is not None else None
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started
        if loss is not None:
            loss_sum += float(loss.item()) * images.shape[0]
        sample_count += images.shape[0]
        binary = (torch.sigmoid(logits) >= threshold).squeeze(1).cpu().numpy()
        for filename, array in zip(filenames, binary):
            patient_id, slice_text = filename.rsplit("_slice", 1)
            slice_index = int(Path(slice_text).stem)
            predictions.setdefault(patient_id, {})[slice_index] = array.astype(bool)
    if set(predictions) != set(patient_ids):
        missing = sorted(set(patient_ids) - set(predictions))
        unexpected = sorted(set(predictions) - set(patient_ids))
        raise RuntimeError(f"Prediction patient mismatch: missing={missing}, unexpected={unexpected}")

    prediction_dir = visualization_dir = None
    if output_dir is not None:
        prediction_dir = output_dir / "predictions"
        visualization_dir = output_dir / "overlays"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        visualization_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for patient_id in patient_ids:
        meta = patient_meta[patient_id]
        expected_slices = int(meta["original_shape"][2])
        if sorted(predictions[patient_id]) != list(range(expected_slices)):
            raise RuntimeError(f"Incomplete slice predictions for {patient_id}")
        label_path = raw_root / patient_id / "laendo.nrrd"
        image_path = raw_root / patient_id / "lgemri.nrrd"
        target_raw, label_header = nrrd.read(str(label_path), index_order="F")
        target = np.asarray(target_raw) > 0
        restored = np.zeros_like(target, dtype=bool)
        for slice_index in range(expected_slices):
            restored[:, :, slice_index] = _restore_prediction(
                predictions[patient_id][slice_index], meta
            )
        dice, iou = overlap_metrics(restored, target)
        spacing = tuple(float(value) for value in meta["spacing"])
        hd95 = hd95_mm(restored, target, spacing) if compute_hd95 else float("nan")
        records.append(
            {
                "patient_id": patient_id,
                "method": method,
                "seed": seed,
                "dice_3d": dice,
                "iou_3d": iou,
                "hd95_header_units": hd95,
                "num_slices": expected_slices,
                "gt_voxels": int(target.sum()),
                "pred_voxels": int(restored.sum()),
            }
        )
        if output_dir is not None:
            prediction_header = dict(label_header)
            prediction_header["type"] = "uint8"
            nrrd.write(
                str(prediction_dir / f"{patient_id}_prediction.nrrd"),
                restored.astype(np.uint8),
                header=prediction_header,
                index_order="F",
            )
            image_raw, _ = nrrd.read(str(image_path), index_order="F")
            representative = int(np.argmax(target.sum(axis=(0, 1))))
            _save_overlay(
                np.asarray(image_raw)[:, :, representative],
                target[:, :, representative],
                restored[:, :, representative],
                patient_id,
                representative,
                dice,
                visualization_dir / f"{patient_id}_overlay.png",
            )

    summary: dict[str, object] = {
        "method": method,
        "seed": seed,
        "num_patients": len(records),
        "num_slices": sample_count,
        "threshold": threshold,
        "postprocessing": "none",
        "dice_3d": _summary([float(row["dice_3d"]) for row in records]),
        "iou_3d": _summary([float(row["iou_3d"]) for row in records]),
        "inference_total_seconds": inference_seconds,
        "inference_ms_per_slice": inference_seconds * 1000 / sample_count,
    }
    if compute_hd95:
        summary["hd95_header_units"] = _summary(
            [float(row["hd95_header_units"]) for row in records]
        )
        summary["hd95_warning"] = (
            "Local NRRD headers report 1 mm spacing; treat HD95 as exploratory "
            "header-unit output until spatial provenance is verified."
        )
    mean_loss = loss_sum / sample_count if criterion is not None else float("nan")
    return records, summary, mean_loss


def write_evaluation(
    records: list[dict[str, object]], summary: dict[str, object], output_dir: Path, split: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{split}_patient_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / f"{split}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
