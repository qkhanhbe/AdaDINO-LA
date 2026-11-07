"""External test of the common-improved-protocol (v2) checkpoints, inference only.

Same external protocol as the main-protocol evaluation (ext_prepare.py / ext_infer_dino.py, 17/10):
CDEMRIS LGE volumes from the two non-Utah centres (Utah/Seg3D-header cases excluded a priori) and
MSD Task02 bSSFP volumes; Utah2018 intensity normalisation, 0.625 mm in-plane resampling, centre
padding to the 640 grid, resize to 448; threshold 0.5; predictions restored to the native grid and
scored there. No retraining, no tuning, no threshold change.

  prepare   build the 2D inputs, native labels and geometry from the raw external data
  check     re-run one main-protocol checkpoint and compare per-case Dice with the archived
            external results (validates this machine and the preparation)
  infer     run the 12 v2 checkpoints (4 strategies x 3 seeds)
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import nrrd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import LeftAtriumDataset  # noqa: E402
from src.metrics import volume_metrics  # noqa: E402
from src.prepare_utah_run2 import normalize_volume  # noqa: E402
from src.protocol_v2_study import CELLS, EXPECTED_TRAINABLE, SEEDS, ProtocolV2Segmenter  # noqa: E402
from src.readout_study import load_backbone  # noqa: E402
from src.train_e5_e6 import build_model, load_trainable_state_dict  # noqa: E402

PIXEL, GRID, SIZE = 0.625, 640, 448
PRETRAINED_SHA256 = "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"
FIELDS = ["config", "seed", "dataset", "case", "dice_3d", "iou_3d", "hd95_mm", "pred_voxels", "label_voxels"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cdemris_cases(root):
    for subset, letter in (("pre", "a"), ("post", "b")):
        for folder in sorted((root / subset).glob("p*"), key=lambda p: int(p.name[1:])):
            n = folder.name[1:]
            yield f"cdemris_{subset}", f"{subset}_p{n}", folder / f"de_{letter}_{n}.nrrd", folder / f"la_seg_{letter}_{n}.nrrd"


def load_cdemris(image_path, label_path):
    image, header = nrrd.read(str(image_path), index_order="F")
    label, _ = nrrd.read(str(label_path), index_order="F")
    if "space directions" not in header:
        return None, None, None, "utah_seg3d_header"
    if header.get("space") != "left-posterior-superior":
        raise ValueError(f"unexpected space {header.get('space')}")
    directions = np.asarray(header["space directions"], dtype=float)
    off = directions - np.diag(np.diag(directions))
    if np.abs(off).max() > 1e-3 or (np.diag(directions) <= 0).any():
        raise ValueError("non axis-aligned or flipped directions")
    if image.shape != label.shape:
        raise ValueError("image/label shape mismatch")
    return np.asarray(image, np.float32), np.asarray(label) > 0, tuple(np.diag(directions)), "ok"


def load_msd(image_path, label_path):
    # MSD Task02 headers carry a placeholder identity affine; reoriented once, by anatomy, to the
    # Utah2018 (L, P, S) voxel order, exactly as in the main-protocol external evaluation.
    zooms = nib.load(str(image_path)).header.get_zooms()[:3]

    def reorient(a):
        return np.ascontiguousarray(np.transpose(a, (2, 0, 1))[:, ::-1, :])
    image = reorient(np.asarray(nib.load(str(image_path)).dataobj, np.float32))
    label = reorient(np.asarray(nib.load(str(label_path)).dataobj) > 0)
    return image, label, (float(zooms[2]), float(zooms[0]), float(zooms[1])), "ok"


def to_input(slice_u8, sx, sy, resample):
    h, w = slice_u8.shape
    nh, nw = round(h * sx / PIXEL), round(w * sy / PIXEL)
    if max(nh, nw) > GRID:
        raise ValueError(f"field of view exceeds {GRID * PIXEL} mm")
    img = Image.fromarray(slice_u8).resize((nw, nh), resample)
    canvas = np.zeros((GRID, GRID), np.uint8)
    top, left = (GRID - nh) // 2, (GRID - nw) // 2
    canvas[top:top + nh, left:left + nw] = np.asarray(img)
    return Image.fromarray(canvas).resize((SIZE, SIZE), resample), (nh, nw, top, left)


def restore(pred448, m):
    nh, nw = m["resampled_hw"]
    top, left = m["pad_top_left"]
    h, w = m["shape"][:2]
    grid = Image.fromarray(pred448.astype(np.uint8) * 255).resize((GRID, GRID), Image.Resampling.NEAREST)
    crop = np.asarray(grid)[top:top + nh, left:left + nw]
    return np.asarray(Image.fromarray(crop).resize((w, h), Image.Resampling.NEAREST)) > 127


def prepare(work):
    cdemris = work / "raw/cdemris"
    if not (cdemris / "pre").is_dir():
        for name in ("pre.zip", "post.zip"):
            with zipfile.ZipFile(ROOT / "data/external/cdemris_raw" / name) as z:
                for member in z.namelist():
                    if member.startswith("/") or ".." in Path(member).parts:
                        raise ValueError(f"unsafe member {member}")
                z.extractall(cdemris)
    cases = list(cdemris_cases(cdemris))
    for p in sorted((ROOT / "data/imagesTr").glob("la_*.nii.gz")):
        cases.append(("msd", "msd_" + p.name[3:6], p, ROOT / "data/labelsTr" / p.name))
    (work / "labels").mkdir(parents=True, exist_ok=True)
    meta, excluded = {}, []
    for dataset, case, image_path, label_path in cases:
        image, label, spacing, status = (load_msd if dataset == "msd" else load_cdemris)(image_path, label_path)
        if status != "ok":
            excluded.append({"dataset": dataset, "case": case, "reason": status})
            continue
        u8, low, high, nz = normalize_volume(image, 0.5, 99.5)
        out = work / "prepared" / dataset
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "masks").mkdir(parents=True, exist_ok=True)
        for k in range(u8.shape[2]):
            im, geom = to_input(u8[:, :, k], spacing[0], spacing[1], Image.Resampling.BILINEAR)
            mk, _ = to_input(label[:, :, k].astype(np.uint8) * 255, spacing[0], spacing[1], Image.Resampling.NEAREST)
            im.save(out / "images" / f"{case}_slice{k:04d}.png")
            mk.save(out / "masks" / f"{case}_slice{k:04d}.png")
        nh, nw, top, left = geom
        np.save(work / "labels" / f"{case}.npy", label)
        meta[case] = {"dataset": dataset, "shape": list(label.shape), "spacing": list(spacing),
                      "resampled_hw": [nh, nw], "pad_top_left": [top, left], "label_voxels": int(label.sum()),
                      "intensity_window": [low, high], "nonzero_percentiles": bool(nz)}
    (work / "prepared/meta.json").write_text(json.dumps(meta, indent=1))
    (work / "prepared/excluded.json").write_text(json.dumps(excluded, indent=1))
    print("prepared", len(meta), "excluded", len(excluded), {d: sum(m["dataset"] == d for m in meta.values())
                                                            for d in sorted({m["dataset"] for m in meta.values()})})


def score(model, work, key, seed, batch_size):
    meta = json.loads((work / "prepared/meta.json").read_text())
    rows = []
    for ds in sorted({m["dataset"] for m in meta.values()}):
        loader = DataLoader(LeftAtriumDataset(str(work / "prepared" / ds)), batch_size=batch_size,
                            num_workers=0, shuffle=False)
        preds = {}
        with torch.no_grad():
            for images, _, names in loader:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(images.to("cuda", non_blocking=True))
                binary = (torch.sigmoid(logits) >= 0.5).squeeze(1).cpu().numpy()
                for name, b in zip(names, binary):
                    case, s = name.rsplit("_slice", 1)
                    preds.setdefault(case, {})[int(Path(s).stem)] = b
        for case in sorted(preds):
            m = meta[case]
            label = np.load(work / "labels" / f"{case}.npy")
            vol = np.stack([restore(preds[case][k], m) for k in range(m["shape"][2])], axis=2)
            r = volume_metrics(vol, label, m["spacing"])
            rows.append({"config": key, "seed": seed, "dataset": ds, "case": case, "dice_3d": r["dice_3d"],
                         "iou_3d": r["iou_3d"], "hd95_mm": r["hd95"], "pred_voxels": int(vol.sum()),
                         "label_voxels": int(label.sum())})
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def check(work, ckpt_root, archived, key, seed, batch_size):
    """Re-score one main-protocol checkpoint; per-case Dice must match the archived external CSV."""
    cfg = json.loads((ckpt_root / key / "config.yaml").read_text())
    state = torch.load(ckpt_root / key / f"seed{seed}/best.pth", map_location="cpu", weights_only=False)
    model, _, _ = build_model(cfg["experiment"], int(cfg.get("lora_rank") or 4), float(cfg.get("lora_alpha") or 4.0))
    load_trainable_state_dict(model, state["trainable_state_dict"])
    rows = score(model.cuda().eval(), work, key, seed, batch_size)
    ref = {r["case"]: float(r["dice_3d"]) for r in csv.DictReader(open(archived / key / f"seed{seed}.csv"))}
    diffs = [abs(r["dice_3d"] - ref[r["case"]]) for r in rows]
    assert len(rows) == len(ref), (len(rows), len(ref))
    print(f"check {key} seed{seed}: {len(rows)} cases, max |dDice| {max(diffs):.2e}, mean {np.mean(diffs):.2e}")
    if max(diffs) > 2e-3:
        raise SystemExit("External preparation or inference does not reproduce the archived results")


def infer(work, runs, dino_repo, weights, batch_size, cells, seeds):
    if sha256_file(weights) != PRETRAINED_SHA256:
        raise SystemExit("Pretrained weights differ from the training runs")
    out_root = work / "results"
    out_root.mkdir(parents=True, exist_ok=True)
    prov_path = out_root / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {
        "status": "inference only; post hoc common improved protocol; external protocol unchanged",
        "pretrained_sha256": PRETRAINED_SHA256, "threshold": 0.5, "amp": "float16", "checkpoints": {}}
    for cell in cells:
        for seed in seeds:
            out = out_root / cell / f"seed{seed}.csv"
            if out.exists():
                print("skip", out)
                continue
            folder = runs / cell / f"seed{seed}"  # train_protocol_v2 layout; flat <cell>_seed<seed> also accepted
            if not folder.is_dir():
                folder = runs / f"{cell}_seed{seed}"
            ckpt = folder / "best.pth"
            digest = sha256_file(ckpt)
            done = json.loads((folder / "train_complete.json").read_text())
            if done.get("best_sha256") not in (None, digest):
                raise SystemExit(f"Checkpoint hash mismatch for {cell} seed{seed}")
            model = ProtocolV2Segmenter(load_backbone(dino_repo, weights), cell, seed)
            if model.num_trainable_params() != EXPECTED_TRAINABLE[cell]:
                raise SystemExit("Unexpected trainable parameter count")
            load_trainable_state_dict(model, torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
            rows = score(model.cuda().eval(), work, cell, seed, batch_size)
            write_rows(out, rows)
            prov["checkpoints"][f"{cell}_seed{seed}"] = {
                "best_sha256": digest, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "python": platform.python_version()}
            prov_path.write_text(json.dumps(prov, indent=1))
            for ds in sorted({r["dataset"] for r in rows}):
                d = [r["dice_3d"] for r in rows if r["dataset"] == ds]
                print(cell, seed, ds, len(d), round(float(np.mean(d)), 4), flush=True)
            del model
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("prepare", "check", "infer"))
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--runs", type=Path, help="v2 runs: <cell>/seed<seed>/best.pth (or <cell>_seed<seed>/best.pth)")
    ap.add_argument("--main-ckpt", type=Path, help="main-protocol checkpoints (<config>/seed<s>/best.pth)")
    ap.add_argument("--archived", type=Path, default=ROOT / "results/revision_v2_20261021/external/dino_fp32")
    ap.add_argument("--dino-repo", type=Path)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mode == "prepare":
        prepare(args.work)
    elif args.mode == "check":
        check(args.work, args.main_ckpt, args.archived, "E6", 42, args.batch_size)
    else:
        infer(args.work, args.runs, args.dino_repo, args.weights, args.batch_size,
              args.cells.split(","), [int(s) for s in args.seeds.split(",")])


if __name__ == "__main__":
    main()
