import torch

from src.augmentation_study import AUGMENTATION, RUNS, augment_batch, augment_sample
from src.dataset import IMAGENET_MEAN, IMAGENET_STD


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    gray = torch.rand(4, 1, 448, 448, generator=g)
    masks = (torch.rand(4, 1, 448, 448, generator=g) > 0.7).float()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (gray.expand(-1, 3, -1, -1) - mean) / std, masks, gray


def test_matrix_has_twelve_runs():
    assert len(RUNS) == 12 and len(set(RUNS)) == 12
    assert {r[1] for r in RUNS} == {"V_LO", "V_SC"} and {r[0] for r in RUNS} == {10, 100}


def test_deterministic_for_same_generator():
    images, masks, _ = _batch()
    a = augment_batch(images, masks, torch.Generator().manual_seed(7))
    b = augment_batch(images, masks, torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_shapes_binary_mask_and_replicated_channels():
    images, masks, _ = _batch(1)
    out_i, out_m = augment_batch(images, masks, torch.Generator().manual_seed(3))
    assert out_i.shape == images.shape and out_m.shape == masks.shape
    assert set(out_m.unique().tolist()) <= {0.0, 1.0}
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    gray = out_i * std + mean
    assert torch.allclose(gray[:, 0], gray[:, 1], atol=1e-5) and torch.allclose(gray[:, 0], gray[:, 2], atol=1e-5)
    assert gray.min() >= -1e-5 and gray.max() <= 1 + 1e-5


def test_identity_when_all_probabilities_zero():
    params = {k: (dict(v) if isinstance(v, dict) else v) for k, v in AUGMENTATION.items()}
    params["rotation_scaling"].update(p_rotation=0.0, p_scaling=0.0)
    for key in ("gaussian_noise", "gaussian_blur", "brightness", "contrast", "low_resolution", "inverted_gamma", "gamma"):
        params[key]["p"] = 0.0
    params["mirroring"]["p_per_axis"] = 0.0
    _, masks, gray = _batch(2)
    g, m = augment_sample(gray[:1], masks[:1], torch.Generator().manual_seed(0), params)
    assert torch.equal(g, gray[:1].clamp(0, 1)) and torch.equal(m, masks[:1])


def test_spatial_transform_moves_mask_with_image():
    gray = torch.zeros(1, 1, 64, 64)
    gray[..., 10:20, 40:50] = 1.0
    mask = (gray > 0.5).float()
    params = {k: (dict(v) if isinstance(v, dict) else v) for k, v in AUGMENTATION.items()}
    for key in ("gaussian_noise", "gaussian_blur", "brightness", "contrast", "low_resolution", "inverted_gamma", "gamma"):
        params[key]["p"] = 0.0
    params["rotation_scaling"].update(p_rotation=1.0, p_scaling=0.0)
    params["mirroring"]["p_per_axis"] = 0.0
    g, m = augment_sample(gray, mask, torch.Generator().manual_seed(5), params)
    overlap = ((g > 0.5) & (m > 0.5)).sum() / (m > 0.5).sum().clamp_min(1)
    assert overlap > 0.8
