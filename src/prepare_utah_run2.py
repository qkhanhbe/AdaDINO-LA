"""Prepare the shared Utah2018 Run2 data from Nam's raw NRRD dataset.

The patient split is read verbatim from ``splits`` directory. Intensity
percentiles are computed once per 3D patient volume, then every axial slice is
written to the flat PNG layout used by this repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import nrrd
import numpy as np
from PIL import Image
from tqdm import tqdm


SPLIT_FILES = {
    "train": "train_patients.txt",
    "val": "val_patients.txt",
    "test": "test_patients.txt",
}
EXPECTED_SPLIT_SHA256 = {
    "train": "2f2b6c441f2646e0a06e8ea470bdd96f7864f4f6706f42b71b1eb3bd8ec49316",
    "val": "56677d26b6f8a16ad3f6239ec7043c5fd0caaca4adae0334447d8f9242ff7456",
    "test": "2e59b4ca4c0ad393bdb456b6dc2f2fcbab7c7727fb7012806d49e84ef394cae8",
}


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Empty or duplicate patient IDs in {path}")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path, paths: list[Path]) -> str:
    """Hash relative names and bytes of a deterministic processed-artifact set."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def normalize_volume(
    volume: np.ndarray, lower_percentile: float = 0.5, upper_percentile: float = 99.5
) -> tuple[np.ndarray, float, float, bool]:
    """Clip a complete MRI volume and min-max normalize it to uint8."""
    values = np.asarray(volume, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("MRI volume has no finite intensities.")
    nonzero = finite[finite != 0]
    used_nonzero = nonzero.size > 0 and nonzero.size < finite.size
    percentile_values = nonzero if used_nonzero else finite
    low, high = np.percentile(
        percentile_values, [lower_percentile, upper_percentile]
    ).astype(float)
    if high <= low:
        normalized = np.zeros_like(values, dtype=np.uint8)
    else:
        clipped = np.clip(values, low, high)
        clipped[~np.isfinite(clipped)] = low
        normalized = np.rint((clipped - low) / (high - low) * 255.0).astype(np.uint8)
    return normalized, low, high, used_nonzero


def pad_to_square(array: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    height, width = array.shape
    side = max(height, width)
    top = (side - height) // 2
    left = (side - width) // 2
    if height == width:
        return array, side, top, left
    output = np.zeros((side, side), dtype=array.dtype)
    output[top : top + height, left : left + width] = array
    return output, side, top, left


def spacing_from_header(header: dict) -> list[float]:
    directions = np.asarray(header.get("space directions"), dtype=np.float64)
    if directions.shape != (3, 3) or not np.isfinite(directions).all():
        raise ValueError("NRRD header has invalid space directions.")
    return [float(value) for value in np.linalg.norm(directions, axis=1)]


def prepare(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root).resolve()
    splits_root = Path(args.splits_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not raw_root.is_dir() or not splits_root.is_dir():
        raise FileNotFoundError("Raw dataset root or split root does not exist.")

    split_ids = {
        split: read_ids(splits_root / filename)
        for split, filename in SPLIT_FILES.items()
    }
    expected_counts = {"train": 108, "val": 15, "test": 31}
    observed_counts = {name: len(ids) for name, ids in split_ids.items()}
    if observed_counts != expected_counts:
        raise ValueError(f"Expected split counts {expected_counts}; got {observed_counts}")
    sets = {name: set(ids) for name, ids in split_ids.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("Patient split overlap detected.")
    all_ids = sets["train"] | sets["val"] | sets["test"]
    if len(all_ids) != 154:
        raise ValueError(f"Expected 154 unique patients, got {len(all_ids)}.")
    split_hashes = {
        name: sha256_file(splits_root / filename)
        for name, filename in SPLIT_FILES.items()
    }
    if split_hashes != EXPECTED_SPLIT_SHA256:
        raise ValueError(
            "Split files do not match the protocol-locked Nam split: "
            f"expected={EXPECTED_SPLIT_SHA256}, observed={split_hashes}"
        )

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output is not empty: {output_root}. Use --overwrite intentionally."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split, filename in SPLIT_FILES.items():
        shutil.copyfile(splits_root / filename, output_root / filename)
        (output_root / f"{split}_2d" / "images").mkdir(parents=True)
        (output_root / f"{split}_2d" / "masks").mkdir(parents=True)

    manifest_rows: list[dict[str, object]] = []
    patient_meta: dict[str, dict[str, object]] = {}
    for split, patient_ids in split_ids.items():
        for patient_id in tqdm(patient_ids, desc=f"prepare {split}"):
            patient_dir = raw_root / patient_id
            image_path = patient_dir / "lgemri.nrrd"
            label_path = patient_dir / "laendo.nrrd"
            if not image_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"Missing MRI/label for patient {patient_id}")
            image, image_header = nrrd.read(str(image_path), index_order="F")
            label, label_header = nrrd.read(str(label_path), index_order="F")
            image = np.asarray(image)
            label = np.asarray(label) > 0
            if image.shape != label.shape or image.ndim != 3:
                raise ValueError(
                    f"Shape mismatch for {patient_id}: {image.shape} vs {label.shape}"
                )
            if image.shape[2] != args.expected_slices:
                raise ValueError(
                    f"{patient_id} has {image.shape[2]} slices; expected {args.expected_slices}."
                )
            image_u8, low, high, excluded_zero = normalize_volume(
                image, args.lower_percentile, args.upper_percentile
            )
            spacing = spacing_from_header(image_header)
            label_spacing = spacing_from_header(label_header)
            if not np.allclose(spacing, label_spacing):
                raise ValueError(f"Image/label spacing mismatch for {patient_id}")

            height, width, depth = image.shape
            pad_side = max(height, width)
            pad_top = (pad_side - height) // 2
            pad_left = (pad_side - width) // 2
            image_dir = output_root / f"{split}_2d" / "images"
            mask_dir = output_root / f"{split}_2d" / "masks"
            for slice_index in range(depth):
                image_slice, _, _, _ = pad_to_square(image_u8[:, :, slice_index])
                mask_slice, _, _, _ = pad_to_square(
                    label[:, :, slice_index].astype(np.uint8) * 255
                )
                image_resized = Image.fromarray(image_slice).resize(
                    (args.image_size, args.image_size), Image.Resampling.BILINEAR
                )
                mask_resized = Image.fromarray(mask_slice).resize(
                    (args.image_size, args.image_size), Image.Resampling.NEAREST
                )
                filename = f"{patient_id}_slice{slice_index:04d}.png"
                image_resized.save(image_dir / filename)
                mask_resized.save(mask_dir / filename)
                manifest_rows.append(
                    {
                        "patient_id": patient_id,
                        "split": split,
                        "slice_index": slice_index,
                        "filename": filename,
                        "has_foreground": bool(label[:, :, slice_index].any()),
                    }
                )

            patient_meta[patient_id] = {
                "split": split,
                "original_shape": [int(v) for v in image.shape],
                "spacing": spacing,
                "space": str(image_header.get("space", "unknown")),
                "space_directions": np.asarray(
                    image_header.get("space directions"), dtype=np.float64
                ).tolist(),
                "space_origin": np.asarray(
                    image_header.get("space origin", [float("nan")] * 3),
                    dtype=np.float64,
                ).tolist(),
                "pad_side": pad_side,
                "pad_top": pad_top,
                "pad_left": pad_left,
                "lower_percentile": low,
                "upper_percentile": high,
                "excluded_zero_background": excluded_zero,
                "source_image": str(image_path),
                "source_label": str(label_path),
                "source_image_sha256": sha256_file(image_path) if args.hash_sources else None,
                "source_label_sha256": sha256_file(label_path) if args.hash_sources else None,
            }

    with (output_root / "preprocessing_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_root / "patient_meta.json").write_text(
        json.dumps(patient_meta, indent=2) + "\n", encoding="utf-8"
    )
    artifact_paths = [
        *(output_root / filename for filename in SPLIT_FILES.values()),
        output_root / "preprocessing_manifest.csv",
        output_root / "patient_meta.json",
        *output_root.glob("*_2d/images/*.png"),
        *output_root.glob("*_2d/masks/*.png"),
    ]
    provenance = {
        "dataset": "Utah2018",
        "target": "left_atrium_cavity_from_laendo.nrrd",
        "preprocessing_version": "utah2018_run2_v1.0",
        "raw_root": str(raw_root),
        "split_root": str(splits_root),
        "split_seed": 2024,
        "split_counts": expected_counts,
        "split_sha256": split_hashes,
        "preprocessing": {
            "normalization_scope": "per_patient_3d_volume",
            "percentiles": [args.lower_percentile, args.upper_percentile],
            "exclude_exact_zero_background_when_mixed": True,
            "image_size": args.image_size,
            "image_interpolation": "bilinear",
            "mask_interpolation": "nearest",
            "augmentation": "none",
        },
        "num_patients": len(patient_meta),
        "num_slices": len(manifest_rows),
        "processed_artifact_count": len(artifact_paths),
        "processed_artifact_tree_sha256": sha256_tree(output_root, artifact_paths),
        "processed_artifact_hash_definition": (
            "SHA-256 over sorted relative UTF-8 path names (8-byte big-endian "
            "length prefix) followed by each file's bytes"
        ),
    }
    (output_root / "dataset_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--splits_root", required=True)
    parser.add_argument("--output_root", default="data/utah2018_run2")
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--expected_slices", type=int, default=88)
    parser.add_argument("--lower_percentile", type=float, default=0.5)
    parser.add_argument("--upper_percentile", type=float, default=99.5)
    parser.add_argument("--hash_sources", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        parser.error("Percentiles must satisfy 0 <= lower < upper <= 100.")
    return args


if __name__ == "__main__":
    prepare(parse_args())
