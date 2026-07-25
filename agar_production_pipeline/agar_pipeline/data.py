from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .common import ensure_columns, read_image, write_image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalize_rgb(image_rgb: np.ndarray) -> torch.Tensor:
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def ellipse_mask_from_detection(
    shape: tuple[int, int],
    row: pd.Series | dict[str, Any],
) -> np.ndarray:
    height, width = shape
    center = (
        int(round(float(row["ellipse_center_x"]))),
        int(round(float(row["ellipse_center_y"]))),
    )
    expansion = float(row.get("physical_expansion", 1.0))
    axes = (
        max(1, int(round(float(row["ellipse_axis_1"]) * expansion / 2.0))),
        max(1, int(round(float(row["ellipse_axis_2"]) * expansion / 2.0))),
    )
    angle = float(row["ellipse_angle"])
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, thickness=-1)
    return mask


def build_plate_training_frame(config: dict[str, Any]) -> pd.DataFrame:
    metadata_dir = Path(config["metadata_dir"])
    plate_dir = Path(config["classical_plate_dir"])
    manifest = pd.read_csv(metadata_dir / "image_manifest.csv", dtype={"image_id": "string"})
    detections = pd.read_csv(
        plate_dir / "plate_detection_strategy_b.csv", dtype={"image_id": "string"}
    )
    ensure_columns(
        manifest,
        {"image_id", "image_path", "split"},
        "image_manifest.csv",
    )
    ensure_columns(
        detections,
        {
            "image_id",
            "processing_status",
            "ellipse_center_x",
            "ellipse_center_y",
            "ellipse_axis_1",
            "ellipse_axis_2",
            "ellipse_angle",
        },
        "plate_detection_strategy_b.csv",
    )
    frame = manifest.merge(detections, on="image_id", how="inner", suffixes=("", "_det"))
    frame = frame[frame["processing_status"].astype(str) == "success"].copy()
    frame["split"] = frame["split"].replace({"validation": "val"})
    return frame.reset_index(drop=True)


def plate_train_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.ShiftScaleRotate(
                shift_limit=0.12,
                scale_limit=0.15,
                rotate_limit=18,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.8,
            ),
            A.Perspective(scale=(0.01, 0.05), keep_size=True, p=0.25),
            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.20,
                p=0.6,
            ),
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=18,
                val_shift_limit=12,
                p=0.35,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 7)),
                ],
                p=0.20,
            ),
            A.ImageCompression(quality_range=(70, 100), p=0.25),
        ]
    )


def plate_eval_transform(image_size: int) -> A.Compose:
    return A.Compose([A.Resize(image_size, image_size)])


class PlateSegmentationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        dataset_root: str | Path,
        image_size: int,
        training: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform = (
            plate_train_transform(image_size)
            if training
            else plate_eval_transform(image_size)
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = self.dataset_root / str(row["image_path"])
        image_bgr = read_image(image_path)
        mask = ellipse_mask_from_detection(image_bgr.shape[:2], row)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image_rgb, mask=mask)
        image_tensor = _normalize_rgb(transformed["image"])
        mask_tensor = torch.from_numpy(
            (transformed["mask"] > 127).astype(np.float32)[None, ...]
        )
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": str(row["image_id"]),
            "image_path": str(image_path),
        }


def axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_coordinates(
    height: int,
    width: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    return [
        (x0, y0)
        for y0 in axis_starts(height, tile_size, overlap)
        for x0 in axis_starts(width, tile_size, overlap)
    ]


def _visible_boxes_for_tile(
    boxes: np.ndarray,
    x0: int,
    y0: int,
    tile_size: int,
    min_visible_fraction: float,
) -> list[list[float]]:
    output: list[list[float]] = []
    x1_tile = x0 + tile_size
    y1_tile = y0 + tile_size
    for box in boxes:
        bx1, by1, bx2, by2 = [float(value) for value in box]
        center_x = 0.5 * (bx1 + bx2)
        center_y = 0.5 * (by1 + by2)
        if not (x0 <= center_x < x1_tile and y0 <= center_y < y1_tile):
            continue
        ix1 = max(bx1, x0)
        iy1 = max(by1, y0)
        ix2 = min(bx2, x1_tile)
        iy2 = min(by2, y1_tile)
        visible_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        original_area = max(1.0, bx2 - bx1) * max(1.0, by2 - by1)
        if visible_area / original_area < min_visible_fraction:
            continue
        output.append(
            [
                max(0.0, ix1 - x0),
                max(0.0, iy1 - y0),
                min(float(tile_size), ix2 - x0),
                min(float(tile_size), iy2 - y0),
            ]
        )
    return output


def _draw_tile_preview(image_bgr: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    preview = image_bgr.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        preview,
        f"colonies={len(boxes)}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def materialize_colony_tiles(config: dict[str, Any], overwrite: bool = False) -> Path:
    output_dir = Path(config["output_dir"])
    tile_dir = output_dir / "tiles"
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    tile_manifest_path = output_dir / "metadata" / "tile_manifest.csv"
    if tile_manifest_path.exists() and not overwrite:
        return tile_manifest_path

    metadata = pd.read_csv(
        Path(config["metadata_dir"]) / "image_manifest.csv",
        dtype={"image_id": "string"},
    )
    intensity = pd.read_csv(
        Path(config["intensity_dir"]) / "intensity_metrics.csv",
        dtype={"image_id": "string"},
    )
    annotations = pd.read_csv(
        Path(config["classical_plate_dir"]) / "object_annotations_normalized.csv",
        dtype={"image_id": "string"},
    )
    ensure_columns(
        metadata,
        {"image_id", "split", "plate_condition", "colonies_number", "n_colony_boxes"},
        "image_manifest.csv",
    )
    ensure_columns(
        intensity,
        {"image_id", "processing_status", "local_flatfield_output_path"},
        "intensity_metrics.csv",
    )
    ensure_columns(
        annotations,
        {
            "image_id",
            "object_type",
            "x_normalized",
            "y_normalized",
            "width_normalized",
            "height_normalized",
        },
        "object_annotations_normalized.csv",
    )
    image_frame = metadata.merge(
        intensity[["image_id", "processing_status", "local_flatfield_output_path"]],
        on="image_id",
        how="inner",
    )
    image_frame = image_frame[
        (image_frame["processing_status"].astype(str) == "success")
        & (image_frame["plate_condition"].isin(["countable", "empty"]))
    ].copy()
    image_frame["split"] = image_frame["split"].replace({"validation": "val"})

    if "center_inside_counting_mask" in annotations.columns:
        annotations = annotations[
            annotations["center_inside_counting_mask"].fillna(False).astype(bool)
        ].copy()
    if "processing_status" in annotations.columns:
        annotations = annotations[
            annotations["processing_status"].astype(str) == "success"
        ].copy()
    annotations = annotations[annotations["object_type"].astype(str) == "colony"].copy()
    grouped = {key: value for key, value in annotations.groupby("image_id")}

    colony_cfg = config["colony"]
    tile_size = int(colony_cfg["tile_size"])
    overlap = int(colony_cfg["tile_overlap"])
    min_visible_fraction = float(colony_cfg.get("min_visible_fraction", 0.25))
    negatives_per_image = int(colony_cfg.get("negative_tiles_per_image", 2))
    jpeg_quality = int(colony_cfg.get("tile_jpeg_quality", 95))
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)

    records: list[dict[str, Any]] = []
    image_metrics: list[dict[str, Any]] = []
    preview_count = 0
    preview_dir = output_dir / "visual_samples" / "04_training_tiles"
    intensity_root = Path(config["intensity_dir"])

    for _, row in tqdm(image_frame.iterrows(), total=len(image_frame), desc="Materialisasi tile"):
        started = time.perf_counter()
        image_id = str(row["image_id"])
        split = str(row["split"])
        source_path = intensity_root / str(row["local_flatfield_output_path"])
        gray = read_image(source_path, cv2.IMREAD_GRAYSCALE)
        image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        height, width = gray.shape
        group = grouped.get(image_id)
        if group is None:
            boxes = np.zeros((0, 4), dtype=np.float32)
        else:
            boxes = np.stack(
                [
                    group["x_normalized"].to_numpy(np.float32),
                    group["y_normalized"].to_numpy(np.float32),
                    (
                        group["x_normalized"] + group["width_normalized"]
                    ).to_numpy(np.float32),
                    (
                        group["y_normalized"] + group["height_normalized"]
                    ).to_numpy(np.float32),
                ],
                axis=1,
            )
        candidate_rows: list[dict[str, Any]] = []
        for x0, y0 in tile_coordinates(height, width, tile_size, overlap):
            visible = _visible_boxes_for_tile(
                boxes, x0, y0, tile_size, min_visible_fraction
            )
            candidate_rows.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "boxes": visible,
                    "is_negative": int(len(visible) == 0),
                }
            )
        if split == "train":
            positives = [item for item in candidate_rows if not item["is_negative"]]
            negatives = [item for item in candidate_rows if item["is_negative"]]
            if negatives_per_image >= 0 and len(negatives) > negatives_per_image:
                chosen = rng.choice(
                    len(negatives), size=negatives_per_image, replace=False
                ).tolist()
                negatives = [negatives[index] for index in chosen]
            selected = positives + negatives
        else:
            selected = candidate_rows

        for tile_index, item in enumerate(selected):
            x0 = int(item["x0"])
            y0 = int(item["y0"])
            crop = image_bgr[y0 : y0 + tile_size, x0 : x0 + tile_size]
            if crop.shape[:2] != (tile_size, tile_size):
                pad_bottom = tile_size - crop.shape[0]
                pad_right = tile_size - crop.shape[1]
                crop = cv2.copyMakeBorder(
                    crop,
                    0,
                    pad_bottom,
                    0,
                    pad_right,
                    borderType=cv2.BORDER_REFLECT_101,
                )
            tile_id = f"{image_id}__x{x0:04d}_y{y0:04d}"
            path = tile_dir / split / f"{tile_id}.jpg"
            write_image(path, crop, quality=jpeg_quality)
            boxes_list = item["boxes"]
            records.append(
                {
                    "tile_id": tile_id,
                    "image_id": image_id,
                    "split": split,
                    "tile_path": path.relative_to(output_dir).as_posix(),
                    "source_path": str(source_path),
                    "x0": x0,
                    "y0": y0,
                    "tile_size": tile_size,
                    "source_width": width,
                    "source_height": height,
                    "n_boxes": len(boxes_list),
                    "is_negative": int(item["is_negative"]),
                    "boxes_json": json.dumps(boxes_list),
                    "true_count_metadata": int(row["colonies_number"]),
                    "true_count_boxes": int(row["n_colony_boxes"]),
                    "plate_condition": str(row["plate_condition"]),
                }
            )
            if preview_count < 5 and boxes_list:
                preview = _draw_tile_preview(crop, boxes_list)
                write_image(preview_dir / f"{preview_count + 1:02d}_{tile_id}.jpg", preview)
                preview_count += 1
        image_metrics.append(
            {
                "image_id": image_id,
                "split": split,
                "candidate_tiles": len(candidate_rows),
                "saved_tiles": len(selected),
                "saved_positive_tiles": sum(not item["is_negative"] for item in selected),
                "saved_negative_tiles": sum(item["is_negative"] for item in selected),
                "source_boxes": int(len(boxes)),
                "elapsed_seconds": time.perf_counter() - started,
                "status": "success",
            }
        )

    tile_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(tile_manifest_path, index=False)
    pd.DataFrame(image_metrics).to_csv(
        output_dir / "metrics" / "tile_materialization_metrics.csv", index=False
    )
    return tile_manifest_path


def colony_train_transform(tile_size: int) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.04,
                scale_limit=0.08,
                rotate_limit=12,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.35,
            ),
            A.RandomBrightnessContrast(0.12, 0.12, p=0.35),
            A.GaussNoise(std_range=(0.01, 0.04), p=0.15),
            A.OneOf([A.GaussianBlur((3, 5)), A.MotionBlur(5)], p=0.12),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.20,
            clip=True,
        ),
    )


def colony_eval_transform(tile_size: int) -> A.Compose:
    return A.Compose(
        [],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            clip=True,
        ),
    )


def gaussian2d(shape: tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    height, width = shape
    y, x = np.ogrid[-(height // 2) : height // 2 + 1, -(width // 2) : width // 2 + 1]
    gaussian = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    return gaussian


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(0.0, b1 * b1 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2

    a2 = 4.0
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(0.0, b2 * b2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(0.0, b3 * b3 - 4 * a3 * c3))
    r3 = (b3 + sq3) / (2 * a3 + 1e-6)
    return min(r1, r2, r3)


def draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=max(diameter / 6, 1e-3))
    x, y = center
    height, width = heatmap.shape
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    if min(left, right, top, bottom) < 0:
        return
    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom,
        radius - left : radius + right,
    ]
    if masked_heatmap.size and masked_gaussian.size:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def build_centernet_targets(
    boxes: np.ndarray,
    image_size: int,
    stride: int,
) -> dict[str, torch.Tensor]:
    output_size = image_size // stride
    heatmap = np.zeros((1, output_size, output_size), dtype=np.float32)
    size = np.zeros((2, output_size, output_size), dtype=np.float32)
    offset = np.zeros((2, output_size, output_size), dtype=np.float32)
    reg_mask = np.zeros((1, output_size, output_size), dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = [float(value) for value in box]
        width = max(1.0, x2 - x1) / stride
        height = max(1.0, y2 - y1) / stride
        center_x = (x1 + x2) * 0.5 / stride
        center_y = (y1 + y2) * 0.5 / stride
        center_int_x = int(np.clip(math.floor(center_x), 0, output_size - 1))
        center_int_y = int(np.clip(math.floor(center_y), 0, output_size - 1))
        radius = max(0, int(gaussian_radius(height, width)))
        draw_gaussian(heatmap[0], (center_int_x, center_int_y), radius)
        size[:, center_int_y, center_int_x] = (width, height)
        offset[:, center_int_y, center_int_x] = (
            center_x - center_int_x,
            center_y - center_int_y,
        )
        reg_mask[0, center_int_y, center_int_x] = 1.0
    return {
        "heatmap": torch.from_numpy(heatmap),
        "size": torch.from_numpy(size),
        "offset": torch.from_numpy(offset),
        "reg_mask": torch.from_numpy(reg_mask),
    }


class ColonyTileDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        tile_manifest: pd.DataFrame,
        output_root: str | Path,
        tile_size: int,
        stride: int,
        training: bool,
    ) -> None:
        self.frame = tile_manifest.reset_index(drop=True)
        self.output_root = Path(output_root)
        self.tile_size = tile_size
        self.stride = stride
        self.transform = (
            colony_train_transform(tile_size)
            if training
            else colony_eval_transform(tile_size)
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_bgr = read_image(self.output_root / str(row["tile_path"]))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes = json.loads(str(row["boxes_json"]))
        labels = [1] * len(boxes)
        transformed = self.transform(image=image_rgb, bboxes=boxes, labels=labels)
        transformed_boxes = np.asarray(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
        image_tensor = _normalize_rgb(transformed["image"])
        targets = build_centernet_targets(
            transformed_boxes, self.tile_size, self.stride
        )
        return {
            "image": image_tensor,
            "boxes": torch.from_numpy(transformed_boxes),
            **targets,
            "meta": {
                "tile_id": str(row["tile_id"]),
                "image_id": str(row["image_id"]),
                "tile_path": str(row["tile_path"]),
                "x0": int(row["x0"]),
                "y0": int(row["y0"]),
                "source_width": int(row["source_width"]),
                "source_height": int(row["source_height"]),
                "true_count_metadata": int(row["true_count_metadata"]),
                "true_count_boxes": int(row["true_count_boxes"]),
                "plate_condition": str(row["plate_condition"]),
            },
        }


def collate_colony_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "heatmap": torch.stack([item["heatmap"] for item in batch]),
        "size": torch.stack([item["size"] for item in batch]),
        "offset": torch.stack([item["offset"] for item in batch]),
        "reg_mask": torch.stack([item["reg_mask"] for item in batch]),
        "boxes": [item["boxes"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }
