"""
dataset.py
----------
PyTorch Dataset for the prepared 2D slices.
Input processing required by DINOv2:
  - Grayscale → RGB (three identical channels)
  - Resize to 448x448
  - Normalise with ImageNet mean/std
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────
IMG_SIZE = 448  # DINOv2 patch_size=14 → 448/14 = 32 patches moi chieu

# Chuan hoa theo ImageNet (DINOv2 duoc huan luyen tren ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LeftAtriumDataset(Dataset):
    """
    Dataset cho phan vung tam nhi trai.

    - image: tensor (3, 448, 448) - Grayscale nhan ban 3 kenh RGB
    - mask:  tensor (1, 448, 448) nhi phan {0, 1}
    """

    def __init__(self, data_dir, augment=False, case_ids=None):
        """
        Args:
            data_dir: duong dan toi thu muc (vd: 'data/train_2d')
                      ben trong phai co 'images/' va 'masks/'
            augment: co ap dung data augmentation hay khong
        """
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "images")
        self.msk_dir = os.path.join(data_dir, "masks")
        self.augment = augment

        # Chi lay nhung file ton tai o ca 2 thu muc images va masks
        img_files = set(os.listdir(self.img_dir))
        msk_files = set(os.listdir(self.msk_dir)) if os.path.exists(self.msk_dir) else set()
        common_files = img_files.intersection(msk_files)
        self.filenames = sorted(list(common_files if len(common_files) > 0 else img_files))
        if case_ids is not None:
            # Restrict to whole cases (filenames are '<case>_sliceNNNN.png').
            wanted = set(case_ids)
            self.filenames = [f for f in self.filenames if f.rsplit('_slice', 1)[0] in wanted]
            found = {f.rsplit('_slice', 1)[0] for f in self.filenames}
            if found != wanted:
                raise ValueError(f'Missing training cases for subset: {sorted(wanted - found)[:5]}')

        # Transform cho anh RGB
        self.img_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Transform cho mask (dung NEAREST de khong tao gia tri trung gian)
        self.msk_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),  # → [0, 1], shape (1, H, W)
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Grayscale → RGB (nhan ban 3 kenh)
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        img = self.img_transform(img)

        # Doc mask
        msk = Image.open(os.path.join(self.msk_dir, fname)).convert("L")
        msk = self.msk_transform(msk)  # (1, H, W), gia tri 0.0 hoac 1.0

        return img, msk, fname


def get_dataloaders(data_root="data", batch_size=32, num_workers=4, train_case_ids=None):
    """
    Tao DataLoader cho train, val, test.

    Returns:
        dict voi key 'train', 'val', 'test' → DataLoader
    """
    loaders = {}

    for split, shuffle, augment in [
        ("train_2d", True, False),
        ("val_2d", False, False),
        ("test_2d", False, False),
    ]:
        data_dir = os.path.join(data_root, split)
        if not os.path.exists(os.path.join(data_dir, "images")):
            print(f"  [CANH BAO] Thu muc {data_dir}/images khong ton tai. Bo qua.")
            continue

        dataset = LeftAtriumDataset(
            data_dir, augment=augment,
            case_ids=train_case_ids if split == "train_2d" else None,
        )
        loader_options = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": True,
            "drop_last": split == "train_2d",
        }
        if num_workers > 0:
            # The dataset has no stochastic augmentation in the unified
            # protocol. Keeping workers alive and prefetching more batches
            # therefore improves I/O overlap without changing sample order or
            # any mathematical training setting.
            loader_options.update(
                persistent_workers=True,
                prefetch_factor=4,
            )
        loader = DataLoader(**loader_options)
        key = split.replace("_2d", "")
        loaders[key] = loader
        print(f"  [{key}] {len(dataset)} samples, {len(loader)} batches (batch_size={batch_size})")

    return loaders


if __name__ == "__main__":
    print("Kiem tra DataLoader...")
    loaders = get_dataloaders(batch_size=4, num_workers=0)
    for key, loader in loaders.items():
        imgs, msks, fnames = next(iter(loader))
        print(f"  [{key}] images: {imgs.shape}, masks: {msks.shape}")
