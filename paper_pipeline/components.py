from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .io_utils import as_three_channel, save_image


@dataclass(frozen=True)
class ColonyComponent:
    component_id: int
    area: int
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    rotation_degrees: float
    crop: np.ndarray
    source_label: int = -1


def _principal_axis_rotation(component_mask: np.ndarray) -> float:
    ys, xs = np.where(component_mask > 0)
    if xs.size < 2:
        return 0.0
    coordinates = np.column_stack((xs, ys)).astype(np.float64)
    coordinates -= coordinates.mean(axis=0, keepdims=True)
    covariance = np.cov(coordinates, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle_from_x = float(np.degrees(np.arctan2(major[1], major[0])))
    return 90.0 - angle_from_x


def _rotate_component_crop(
    image_bgr: np.ndarray,
    component_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    rotation = _principal_axis_rotation(component_mask)
    height, width = component_mask.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)

    rotated_image = cv2.warpAffine(
        image_bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rotated_mask = cv2.warpAffine(
        component_mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    points = cv2.findNonZero(rotated_mask)
    if points is None:
        return rotated_image, rotation
    x, y, w, h = cv2.boundingRect(points)
    cropped = rotated_image[y : y + h, x : x + w].copy()
    cropped_mask = rotated_mask[y : y + h, x : x + w]
    cropped[cropped_mask == 0] = 0
    return cropped, rotation


def spatially_normalize_crop(
    crop_bgr: np.ndarray,
    target_size: int = 128,
) -> np.ndarray:
    """Resize sisi panjang bila perlu, lalu zero-pad ke target persegi."""
    crop = as_three_channel(crop_bgr)
    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    if max(height, width) > target_size:
        scale = target_size / float(max(height, width))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        crop = cv2.resize(
            crop,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )
        height, width = crop.shape[:2]

    output = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y = (target_size - height) // 2
    x = (target_size - width) // 2
    output[y : y + height, x : x + width] = crop
    return output


def extract_colony_components(
    image: np.ndarray,
    colony_mask: np.ndarray,
    target_size: int = 128,
    min_area: int = 1,
) -> tuple[list[ColonyComponent], np.ndarray]:
    image_bgr = as_three_channel(image)
    mask = np.where(colony_mask > 0, 1, 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    components: list[ColonyComponent] = []
    for label in range(1, n_labels):
        x, y, width, height, area = stats[label].tolist()
        if int(area) < min_area:
            continue
        local_image = image_bgr[y : y + height, x : x + width].copy()
        local_mask = np.where(
            labels[y : y + height, x : x + width] == label,
            255,
            0,
        ).astype(np.uint8)
        local_image[local_mask == 0] = 0
        rotated, angle = _rotate_component_crop(local_image, local_mask)
        normalized = spatially_normalize_crop(rotated, target_size)
        components.append(
            ColonyComponent(
                component_id=len(components) + 1,
                area=int(area),
                bbox_x=int(x),
                bbox_y=int(y),
                bbox_width=int(width),
                bbox_height=int(height),
                rotation_degrees=float(angle),
                crop=normalized,
                source_label=int(label),
            )
        )
    return components, labels


def save_components(
    components: list[ColonyComponent],
    output_dir: Path,
    image_id: str,
) -> list[Path]:
    paths: list[Path] = []
    for component in components:
        path = output_dir / f"{image_id}_cc{component.component_id:04d}.png"
        save_image(path, component.crop)
        paths.append(path)
    return paths

