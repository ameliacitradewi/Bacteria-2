from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .io_utils import as_three_channel, read_image


def _resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


class SegmentationDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        horizontal_flip_probability: float = 0.0,
        resize_long_side: int | None = None,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        frame = pd.read_csv(self.manifest_path, dtype={"image_id": "string"})
        self.frame = frame[frame["split"].astype(str) == split].reset_index(
            drop=True
        )
        self.horizontal_flip_probability = horizontal_flip_probability
        self.resize_long_side = resize_long_side

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[index]
        image_path = _resolve_path(
            self.manifest_path,
            str(row["image_path"]),
        )
        mask_path = _resolve_path(
            self.manifest_path,
            str(row["mask_path"]),
        )
        image = as_three_channel(read_image(image_path, cv2.IMREAD_UNCHANGED))
        mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.resize_long_side:
            scale = self.resize_long_side / float(max(image.shape[:2]))
            new_size = (
                max(1, int(round(image.shape[1] * scale))),
                max(1, int(round(image.shape[0] * scale))),
            )
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

        if random.random() < self.horizontal_flip_probability:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(
            image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        mask_tensor = torch.from_numpy(
            (mask > 127).astype(np.float32)[None, :, :]
        )
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": str(row["image_id"]),
        }


class ResNetCountDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        frame = pd.read_csv(self.manifest_path)
        self.frame = frame[frame["split"].astype(str) == split].reset_index(
            drop=True
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.frame.iloc[index]
        crop_path = _resolve_path(
            self.manifest_path,
            str(row["crop_path"]),
        )
        image = as_three_channel(read_image(crop_path, cv2.IMREAD_UNCHANGED))
        if image.shape[:2] != (128, 128):
            image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(
            image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        return {
            "image": tensor,
            "label": int(row["label"]),
            "crop_id": str(row["crop_id"]),
        }

