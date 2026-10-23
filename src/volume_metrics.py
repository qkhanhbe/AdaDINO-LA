"""Binary overlap and surface-distance metrics for complete 3D volumes."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def overlap_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return exact binary Dice and IoU for a complete 3D volume."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = np.count_nonzero(prediction & target)
    pred_count = np.count_nonzero(prediction)
    target_count = np.count_nonzero(target)
    total = pred_count + target_count
    union = pred_count + target_count - intersection
    dice = 1.0 if total == 0 else (2.0 * intersection) / total
    iou = 1.0 if union == 0 else intersection / union
    return float(dice), float(iou)


def hd95_mm(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """Return symmetric 95th-percentile Hausdorff distance in millimetres.

    If both masks are empty the distance is zero. If exactly one is empty, the
    physical diagonal of the volume is used as a finite worst-case penalty.
    """
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    spacing_array = np.asarray(spacing, dtype=np.float64)

    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        extent = np.maximum(np.asarray(prediction.shape) - 1, 0) * spacing_array
        return float(np.linalg.norm(extent))

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    pred_surface = prediction ^ ndimage.binary_erosion(
        prediction, structure=structure, border_value=0
    )
    target_surface = target ^ ndimage.binary_erosion(
        target, structure=structure, border_value=0
    )
    distance_to_target = ndimage.distance_transform_edt(
        ~target_surface, sampling=spacing_array
    )
    distance_to_prediction = ndimage.distance_transform_edt(
        ~pred_surface, sampling=spacing_array
    )
    prediction_to_target = distance_to_target[pred_surface]
    target_to_prediction = distance_to_prediction[target_surface]
    return float(
        max(
            np.percentile(prediction_to_target, 95),
            np.percentile(target_to_prediction, 95),
        )
    )
