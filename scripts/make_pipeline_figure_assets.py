"""Raster panels for the redesigned Figure 1 (pipeline schematic), written next to the manuscript as fig1_*.png.

Uses the median-ranked test volume of the qualitative figure (qualitative_main.json from
scripts/build_qualitative_figure_v2.py), its axial slice with the largest reference LA area, and the seed-42
main-protocol Large LoRA prediction saved by that script. Panels: preprocessed 448x448 input, the same input with
the 14x14 patch grid, an RGB projection (first three principal components) of the final-block DINOv2 ViT-L/14 patch
tokens of the pretrained encoder, the predicted mask on the native slice, and an exploded stack of four predicted
slice masks (slice-wise predictions stacked into a volume). The figure is illustrative; no reported number
depends on it.

Run: python -B scripts/make_pipeline_figure_assets.py --qual-dir <qualitative output> --tmp <qualitative tmp>
     --paper-dir <manuscript working copy>
"""
import argparse
import json
from pathlib import Path

import nrrd
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/utah2018_raw_eval"
HUB = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
PRED_COLOR = (61 / 255, 183 / 255, 255 / 255)


def preprocess(slice_hw):
    """Training-time slice preprocessing: percentile clip, min-max to 8 bit, zero-pad to square, resize to 448."""
    v = slice_hw.astype(np.float32)
    nz = v[np.isfinite(v) & (v != 0)]
    lo, hi = np.percentile(nz, [0.5, 99.5]) if nz.size else (v.min(), v.max())
    v = np.clip(v, lo, hi)
    v = np.round((v - lo) / max(hi - lo, 1e-6) * 255.0) / 255.0
    h, w = v.shape
    s = max(h, w)
    pad = np.zeros((s, s), np.float32)
    pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = v
    t = torch.from_numpy(pad)[None, None]
    return F.interpolate(t, size=(448, 448), mode="bilinear", align_corners=False)[0, 0].numpy()


def save_gray(arr, path, grid=False):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(1.6, 1.6), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr, cmap="gray", interpolation="bilinear", vmin=0, vmax=1)
    if grid:
        for k in range(0, 449, 14):
            ax.axhline(k - 0.5, color="white", lw=0.25, alpha=0.55)
            ax.axvline(k - 0.5, color="white", lw=0.25, alpha=0.55)
    ax.set_axis_off()
    fig.savefig(path, dpi=400)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qual-dir", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--paper-dir", required=True)
    args = ap.parse_args()
    out = Path(args.paper_dir)
    meta = json.loads((Path(args.qual_dir) / "qualitative_main.json").read_text(encoding="utf-8"))
    case = meta["cases"]["Median case"]["case"]
    img, hdr = nrrd.read(str(RAW / case / "lgemri.nrrd"), index_order="F")
    lab = nrrd.read(str(RAW / case / "laendo.nrrd"), index_order="F")[0] > 0
    pred = nrrd.read(str(Path(args.tmp) / "Large_LoRA" / "predictions" / f"{case}_prediction.nrrd"), index_order="F")[0] > 0
    z = int(np.argmax(lab.sum(axis=(0, 1))))
    x448 = preprocess(img[:, :, z].T)
    save_gray(x448, out / "fig1_mri.png")
    save_gray(x448, out / "fig1_patches.png", grid=True)

    # DINOv2 ViT-L/14 final-block patch tokens of the pretrained encoder -> first three principal components
    model = torch.hub.load(str(HUB), "dinov2_vitl14", source="local", pretrained=True).eval()
    x = torch.from_numpy(np.repeat(x448[None], 3, axis=0))[None]
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    with torch.no_grad():
        tok = model.forward_features((x - mean) / std)["x_norm_patchtokens"][0].numpy()  # 1024 x 1024
    tok = tok - tok.mean(axis=0)
    u, s, vt = np.linalg.svd(tok, full_matrices=False)
    pcs = tok @ vt[:3].T
    lo, hi = np.percentile(pcs, 1, axis=0), np.percentile(pcs, 99, axis=0)
    rgb = np.clip((pcs - lo) / (hi - lo), 0, 1).reshape(32, 32, 3)
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(1.6, 1.6), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(out / "fig1_tokens.png", dpi=400)
    plt.close(fig)

    # predicted mask on the native slice, cropped around the heart as in Figure 2
    nat = img[:, :, z].T.astype(np.float32)
    nat = (nat - nat.min()) / max(nat.max() - nat.min(), 1e-6)
    ref = lab[:, :, z].T
    ys, xs = np.where(ref)
    cy, cx, half = int(ys.mean()), int(xs.mean()), 110
    win = (slice(max(cy - half, 0), cy + half), slice(max(cx - half, 0), cx + half))
    p2 = pred[:, :, z].T[win]
    fig = plt.figure(figsize=(1.6, 1.6), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(nat[win], cmap="gray", interpolation="bilinear")
    ax.imshow(np.ma.masked_where(~p2, p2), cmap=mpl.colors.ListedColormap([PRED_COLOR]), alpha=0.45, interpolation="nearest")
    ax.contour(p2, levels=[0.5], colors=[PRED_COLOR], linewidths=0.9)
    ax.set_axis_off()
    fig.savefig(out / "fig1_pred.png", dpi=400)
    plt.close(fig)

    picks = slice_stack(img, pred, out / "fig1_stack.png")
    print("case", case, "slice", z + 1, "stack slices", [k + 1 for k in picks], "assets written")


def slice_stack(img, pred, path, n=4, k=0.42, shear=0.55, gap=0.85, margin=25, tile=480):
    """Exploded stack of n axial slices of the predicted mask (evenly spaced between 15% and 85% of the predicted
    slice range, inferior at the bottom), each drawn as an oblique plane: the model predicts slice by slice and the
    slice masks are stacked into a volume. No smoothing or 3D processing is applied."""
    from PIL import Image, ImageDraw
    from scipy import ndimage
    zs = np.where(pred.any(axis=(0, 1)))[0]
    picks = [int(round(zs[0] + f * (zs[-1] - zs[0]))) for f in np.linspace(0.15, 0.85, n)]
    xs, ys = np.where(pred.any(axis=2))
    x0, x1, y0, y1 = xs.min() - margin, xs.max() + margin, ys.min() - margin, ys.max() + margin
    side = max(x1 - x0, y1 - y0)
    x0, y0 = max((x0 + x1) // 2 - side // 2, 0), max((y0 + y1) // 2 - side // 2, 0)
    crop = (slice(x0, x0 + side), slice(y0, y0 + side))
    lo, hi = np.percentile(img[crop][..., picks].astype(np.float32), [1, 99.5])
    color = np.array(PRED_COLOR)
    planes = []
    for z in picks:
        g = np.clip((img[crop][..., z].T.astype(np.float32) - lo) / (hi - lo), 0, 1)
        m = pred[crop][..., z].T
        edge = m & ~ndimage.binary_erosion(m, iterations=2)
        rgb = np.repeat(g[..., None], 3, axis=2)
        rgb = np.where(m[..., None], rgb * 0.55 + color * 0.45, rgb)
        rgb = np.where(edge[..., None], color, rgb)
        t = Image.fromarray((rgb * 255).astype(np.uint8)).resize((tile, tile), Image.LANCZOS).convert("RGBA")
        ImageDraw.Draw(t).rectangle([0, 0, tile - 1, tile - 1], outline=(110, 110, 110, 255), width=3)
        w, h = int(tile * (1 + shear * k)), int(tile * k)
        # oblique plane: output (x, y) -> input (u, v) with v = y / k and u = x - shear * k * tile + shear * y
        planes.append(t.transform((w, h), Image.AFFINE, (1, shear, -shear * k * tile, 0, 1 / k, 0),
                                  resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0)))
    w, h = planes[0].size
    dy = int(gap * h)
    canvas = Image.new("RGBA", (w, h + dy * (n - 1)), (0, 0, 0, 0))
    for i, plane in enumerate(planes):
        canvas.alpha_composite(plane, (0, dy * (n - 1 - i)))
    canvas.save(path)
    return picks


if __name__ == "__main__":
    main()
