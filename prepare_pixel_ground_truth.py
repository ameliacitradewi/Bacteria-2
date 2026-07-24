"""Prepare box-guided pixel masks for U²-Net annotation.

This script does NOT claim automatic masks are final ground truth. It uses the
normalized bounding boxes as spatial priors and creates pseudo-mask candidates
from Otsu, adaptive threshold, and Sauvola. The candidates are combined by
consensus, saved with review overlays, and ranked in review_queue.csv.

Default inputs follow the current Bacteria-2/main layout:
- preprocessed_intensity_sigma040/local_flatfield/
- processed_plate_strategy_b_circle/counting_mask/
- processed_plate_strategy_b_circle/object_annotations_normalized.csv
- agar_metadata/image_manifest.csv

Example:
    python prepare_pixel_ground_truth.py --limit 2
    python prepare_pixel_ground_truth.py --methods otsu sauvola

After running, manually review/correct pseudo_masks/binary before using them as
U²-Net ground-truth masks.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd
from skimage import filters, measure, morphology, util
from tqdm import tqdm

ThresholdMethod = Literal["otsu", "adaptive", "sauvola"]
ForegroundMode = Literal["bright", "dark"]
SCRIPT_VERSION = "2026-07-24-pandas-safe-v2"

SUPPORTED_METHODS: tuple[ThresholdMethod, ...] = (
    "otsu",
    "adaptive",
    "sauvola",
)


@dataclass(frozen=True)
class Config:
    project_root: Path
    image_dir: Path
    counting_mask_dir: Path
    annotations_csv: Path
    metadata_csv: Path
    output_dir: Path

    methods: tuple[ThresholdMethod, ...] = SUPPORTED_METHODS
    foreground_mode: ForegroundMode = "bright"

    bbox_padding_ratio: float = 0.15
    bbox_padding_min_px: int = 6
    bbox_constraint_dilation_px: int = 2
    min_consensus_votes: int = 2

    adaptive_block_size: int = 51
    adaptive_offset: float = 0.0
    sauvola_window_size: int = 51
    sauvola_k: float = 0.20

    min_object_area: int = 8
    max_hole_area: int = 64
    morphology_radius: int = 1
    min_component_score: float = 0.22

    auto_min_area_ratio: float = 0.01
    auto_max_area_ratio: float = 0.90
    auto_min_pairwise_iou: float = 0.30
    auto_min_confidence: float = 0.50

    save_annotation_crops: bool = True


def default_config(project_root: Path | None = None) -> Config:
    root = (
        project_root.resolve()
        if project_root is not None
        else Path(__file__).resolve().parent
    )
    return Config(
        project_root=root,
        image_dir=root / "preprocessed_intensity_sigma040" / "local_flatfield",
        counting_mask_dir=root
        / "processed_plate_strategy_b_circle"
        / "counting_mask",
        annotations_csv=root
        / "processed_plate_strategy_b_circle"
        / "object_annotations_normalized.csv",
        metadata_csv=root / "agar_metadata" / "image_manifest.csv",
        output_dir=root / "pixel_ground_truth_workspace",
    )


def validate_config(cfg: Config) -> None:
    required = {
        "local-flatfield image directory": cfg.image_dir,
        "counting-mask directory": cfg.counting_mask_dir,
        "normalized annotation CSV": cfg.annotations_csv,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Input tidak lengkap:\n- " + "\n- ".join(missing))
    if not cfg.methods:
        raise ValueError("Minimal satu metode threshold harus dipilih.")
    if not 1 <= cfg.min_consensus_votes <= len(cfg.methods):
        raise ValueError(
            "min_consensus_votes harus berada antara 1 dan jumlah metode."
        )
    if cfg.bbox_padding_ratio < 0 or cfg.bbox_padding_min_px < 0:
        raise ValueError("Padding bounding box tidak boleh negatif.")
    if cfg.min_object_area < 0 or cfg.max_hole_area < 0:
        raise ValueError("Luas objek/lubang tidak boleh negatif.")
    for name, value in (
        ("adaptive_block_size", cfg.adaptive_block_size),
        ("sauvola_window_size", cfg.sauvola_window_size),
    ):
        if value < 3 or value % 2 == 0:
            raise ValueError(f"{name} harus ganjil dan minimal 3.")


def normalize_image_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
        return str(value).strip()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def safe_filename(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "annotation"


def read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise RuntimeError(f"Gagal membaca gambar: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix != ".png":
        raise ValueError(f"Output gambar harus PNG: {path}")
    ok, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 5]
    )
    if not ok:
        raise RuntimeError(f"Gagal meng-encode: {path}")
    encoded.tofile(str(path))


def to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Dimensi gambar tidak didukung: {image.shape}")
    gray_float = util.img_as_float32(gray)
    return np.clip(
        np.nan_to_num(gray_float, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    ).astype(np.float32)


def to_display_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        if image.dtype != np.uint8:
            arr = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            arr = image
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        if image.dtype == np.uint8:
            return image.copy()
        return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    raise ValueError(f"Dimensi display tidak didukung: {image.shape}")


def load_annotations(path: Path) -> pd.DataFrame:
    """Load and validate normalized bounding-box annotations.

    The annotation ID column is rebuilt as Python strings instead of being
    modified in place. This avoids Pandas 2/3 LossySetitemError when the CSV
    inferred annotation_id as int64 but fallback IDs are strings.
    """
    annotations = pd.read_csv(path)
    required = {
        "image_id",
        "x_normalized",
        "y_normalized",
        "width_normalized",
        "height_normalized",
    }
    missing = required.difference(annotations.columns)
    if missing:
        raise KeyError(f"Kolom anotasi tidak lengkap: {sorted(missing)}")

    annotations = annotations.copy()

    # Normalize image IDs and discard rows that cannot be matched to an image.
    annotations["image_id"] = annotations["image_id"].map(normalize_image_id)
    annotations = annotations.loc[annotations["image_id"].notna()].copy()

    if "processing_status" in annotations.columns:
        status = annotations["processing_status"].astype("string").str.strip().str.lower()
        annotations = annotations.loc[status.eq("success")].copy()

    if "center_inside_counting_mask" in annotations.columns:
        inside = annotations["center_inside_counting_mask"]
        if pd.api.types.is_bool_dtype(inside.dtype):
            valid_inside = inside.fillna(False)
        elif pd.api.types.is_numeric_dtype(inside.dtype):
            valid_inside = pd.to_numeric(inside, errors="coerce").fillna(0).ne(0)
        else:
            valid_inside = (
                inside.astype("string")
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes", "y"})
            )
        annotations = annotations.loc[valid_inside].copy()

    numeric_columns = [
        "x_normalized",
        "y_normalized",
        "width_normalized",
        "height_normalized",
    ]
    for column in numeric_columns:
        annotations[column] = pd.to_numeric(annotations[column], errors="coerce")

    annotations = annotations.dropna(subset=numeric_columns).copy()
    annotations = annotations.loc[
        (annotations["width_normalized"] > 0)
        & (annotations["height_normalized"] > 0)
    ].copy()

    # Build annotation IDs in a separate Python list. Do not assign strings
    # through .loc into a source column that Pandas may have inferred as int64.
    if "annotation_id" in annotations.columns:
        raw_ids = annotations["annotation_id"].tolist()
    else:
        raw_ids = [None] * len(annotations)

    normalized_ids: list[str] = []
    used_ids: set[str] = set()
    for position, raw_value in enumerate(raw_ids):
        candidate = normalize_image_id(raw_value)
        if candidate is None:
            candidate = f"ann_{position:06d}"

        unique_id = candidate
        duplicate_index = 1
        while unique_id in used_ids:
            unique_id = f"{candidate}_{duplicate_index:02d}"
            duplicate_index += 1

        used_ids.add(unique_id)
        normalized_ids.append(unique_id)

    # Drop the inferred source column before recreating it with object/string
    # dtype. This also works when zero rows remain after filtering.
    if "annotation_id" in annotations.columns:
        annotations = annotations.drop(columns=["annotation_id"])
    annotations = annotations.assign(
        annotation_id=pd.Series(normalized_ids, index=annotations.index, dtype="string")
    )

    # Stable sorting groups rows by image and retains source order within each
    # image. No temporary _row_order column is required.
    annotations = annotations.sort_values(
        by="image_id", kind="stable", na_position="last"
    )
    return annotations.reset_index(drop=True)


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["image_id", "background", "split"]).set_index(
            "image_id", drop=False
        )
    metadata = pd.read_csv(path)
    if "image_id" not in metadata.columns:
        raise KeyError(f"Kolom image_id tidak tersedia pada {path}")
    metadata = metadata.copy()
    metadata["image_id"] = metadata["image_id"].map(normalize_image_id)
    metadata = metadata[metadata["image_id"].notna()].copy()
    for column in ("background", "split"):
        if column not in metadata.columns:
            metadata[column] = "unknown"
    metadata = metadata.drop_duplicates("image_id", keep="first")
    return metadata.set_index("image_id", drop=False)


def build_image_lookup(image_dir: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(image_dir.rglob("*.png")):
        image_id = normalize_image_id(path.stem)
        if image_id is None:
            continue
        if image_id in lookup:
            duplicates.setdefault(image_id, [lookup[image_id]]).append(path)
        else:
            lookup[image_id] = path
    if duplicates:
        examples = "; ".join(
            f"{image_id}: {[str(p) for p in paths]}"
            for image_id, paths in list(duplicates.items())[:5]
        )
        raise ValueError(f"image_id duplikat di image_dir: {examples}")
    return lookup


def find_counting_mask(
    image_path: Path,
    image_dir: Path,
    counting_mask_dir: Path,
) -> Path:
    relative = image_path.relative_to(image_dir)
    direct = counting_mask_dir / relative
    if direct.exists():
        return direct
    matches = sorted(counting_mask_dir.rglob(f"{image_path.stem}.png"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Counting mask tidak ditemukan untuk {image_path.name}"
        )
    raise ValueError(
        f"Counting mask ambigu untuk {image_path.stem}: {matches}"
    )


def clip_box(
    row: pd.Series,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = shape
    x0 = int(math.floor(float(row["x_normalized"])))
    y0 = int(math.floor(float(row["y_normalized"])))
    x1 = int(math.ceil(float(row["x_normalized"] + row["width_normalized"])))
    y1 = int(math.ceil(float(row["y_normalized"] + row["height_normalized"])))
    x0 = int(np.clip(x0, 0, width - 1))
    y0 = int(np.clip(y0, 0, height - 1))
    x1 = int(np.clip(x1, x0 + 1, width))
    y1 = int(np.clip(y1, y0 + 1, height))
    return x0, y0, x1, y1


def expand_box(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    ratio: float,
    minimum_px: int,
) -> tuple[int, int, int, int]:
    height, width = shape
    x0, y0, x1, y1 = box
    padding = max(minimum_px, int(round(max(x1 - x0, y1 - y0) * ratio)))
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


def safe_odd_window(requested: int, shape: tuple[int, int]) -> int | None:
    maximum = min(shape)
    if maximum < 3:
        return None
    if maximum % 2 == 0:
        maximum -= 1
    return max(3, min(requested, maximum))


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(bool)
    labels = measure.label(mask.astype(bool), connectivity=2)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels]


def fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    if max_hole_area <= 0:
        return mask.astype(bool)
    inverted = ~mask.astype(bool)
    labels = measure.label(inverted, connectivity=2)
    sizes = np.bincount(labels.ravel())
    border_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    fill = sizes <= max_hole_area
    fill[0] = False
    fill[border_labels] = False
    return mask.astype(bool) | fill[labels]


def clean_mask(mask: np.ndarray, cfg: Config) -> np.ndarray:
    cleaned = remove_small_components(mask, cfg.min_object_area)
    cleaned = fill_small_holes(cleaned, cfg.max_hole_area)
    if cfg.morphology_radius > 0:
        footprint = morphology.disk(cfg.morphology_radius)
        cleaned = morphology.closing(cleaned, footprint=footprint)
        cleaned = morphology.opening(cleaned, footprint=footprint)
    return remove_small_components(cleaned, cfg.min_object_area).astype(bool)


def threshold_crop(
    crop: np.ndarray,
    valid_mask: np.ndarray,
    method: ThresholdMethod,
    cfg: Config,
) -> tuple[np.ndarray, float]:
    values = crop[valid_mask]
    if values.size < 2 or float(np.ptp(values)) <= 1e-7:
        return np.zeros_like(valid_mask, dtype=bool), float("nan")

    if method == "otsu":
        threshold = float(filters.threshold_otsu(values))
        threshold_map: float | np.ndarray = threshold
    elif method == "adaptive":
        window = safe_odd_window(cfg.adaptive_block_size, crop.shape)
        if window is None:
            return np.zeros_like(valid_mask, dtype=bool), float("nan")
        threshold_map = filters.threshold_local(
            crop,
            block_size=window,
            method="gaussian",
            offset=cfg.adaptive_offset,
            mode="reflect",
        )
        threshold = float(np.mean(threshold_map[valid_mask]))
    elif method == "sauvola":
        window = safe_odd_window(cfg.sauvola_window_size, crop.shape)
        if window is None:
            return np.zeros_like(valid_mask, dtype=bool), float("nan")
        threshold_map = filters.threshold_sauvola(
            crop,
            window_size=window,
            k=cfg.sauvola_k,
        )
        threshold = float(np.mean(threshold_map[valid_mask]))
    else:
        raise ValueError(f"Metode tidak didukung: {method}")

    if cfg.foreground_mode == "bright":
        mask = crop > threshold_map
    else:
        mask = crop < threshold_map
    return clean_mask(mask & valid_mask, cfg), threshold


def box_mask_in_crop(
    inner_box: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
    crop_shape: tuple[int, int],
    dilation_px: int,
) -> np.ndarray:
    x0, y0, x1, y1 = inner_box
    cx0, cy0, _, _ = crop_box
    local_x0 = max(0, x0 - cx0)
    local_y0 = max(0, y0 - cy0)
    local_x1 = min(crop_shape[1], x1 - cx0)
    local_y1 = min(crop_shape[0], y1 - cy0)
    result = np.zeros(crop_shape, dtype=bool)
    result[local_y0:local_y1, local_x0:local_x1] = True
    if dilation_px > 0:
        result = morphology.dilation(
            result, footprint=morphology.disk(dilation_px)
        )
    return result


def component_bbox_iou(
    region_bbox: tuple[int, int, int, int],
    inner_box_local: tuple[int, int, int, int],
) -> float:
    ry0, rx0, ry1, rx1 = region_bbox
    bx0, by0, bx1, by1 = inner_box_local
    ix0, iy0 = max(rx0, bx0), max(ry0, by0)
    ix1, iy1 = min(rx1, bx1), min(ry1, by1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (rx1 - rx0) * (ry1 - ry0) + (bx1 - bx0) * (by1 - by0) - intersection
    return float(intersection / union) if union > 0 else 0.0


def select_best_component(
    mask: np.ndarray,
    inner_box: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
    allowed_box_mask: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, float, dict[str, float]]:
    labels = measure.label(mask.astype(bool), connectivity=2)
    regions = measure.regionprops(labels)
    if not regions:
        return np.zeros_like(mask, dtype=bool), 0.0, {}

    x0, y0, x1, y1 = inner_box
    cx0, cy0, _, _ = crop_box
    local_box = (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0)
    box_width = max(x1 - x0, 1)
    box_height = max(y1 - y0, 1)
    box_area = float(box_width * box_height)
    box_center_x = (local_box[0] + local_box[2]) / 2.0
    box_center_y = (local_box[1] + local_box[3]) / 2.0

    best_score = -float("inf")
    best_mask = np.zeros_like(mask, dtype=bool)
    best_metrics: dict[str, float] = {}

    for region in regions:
        component = labels == region.label
        inside = component & allowed_box_mask
        intersection_area = int(inside.sum())
        if intersection_area == 0:
            continue

        component_area = float(region.area)
        overlap_inside = intersection_area / max(component_area, 1.0)
        box_coverage = intersection_area / box_area
        area_ratio = component_area / box_area

        centroid_y, centroid_x = region.centroid
        dx = (centroid_x - box_center_x) / max(box_width / 2.0, 1.0)
        dy = (centroid_y - box_center_y) / max(box_height / 2.0, 1.0)
        distance = math.hypot(dx, dy)
        center_score = max(0.0, 1.0 - distance / 1.75)

        ideal_ratio = 0.45
        size_score = math.exp(
            -abs(math.log(max(area_ratio, 1e-6) / ideal_ratio))
        )
        bbox_iou = component_bbox_iou(region.bbox, local_box)

        ry0, rx0, ry1, rx1 = region.bbox
        touches_crop = (
            ry0 == 0 or rx0 == 0 or ry1 == mask.shape[0] or rx1 == mask.shape[1]
        )
        border_penalty = 0.15 if touches_crop else 0.0
        huge_penalty = min(max(area_ratio - 1.5, 0.0) * 0.15, 0.35)

        score = (
            0.34 * overlap_inside
            + 0.24 * center_score
            + 0.20 * size_score
            + 0.12 * min(box_coverage / 0.50, 1.0)
            + 0.10 * bbox_iou
            - border_penalty
            - huge_penalty
        )
        if score > best_score:
            best_score = score
            best_mask = inside
            best_metrics = {
                "component_area": component_area,
                "component_area_ratio": area_ratio,
                "component_overlap_inside": overlap_inside,
                "component_box_coverage": box_coverage,
                "component_center_distance": distance,
                "component_bbox_iou": bbox_iou,
                "component_touches_crop": float(touches_crop),
            }

    if best_score < cfg.min_component_score:
        return np.zeros_like(mask, dtype=bool), max(best_score, 0.0), best_metrics
    return clean_mask(best_mask, cfg), float(best_score), best_metrics


def pairwise_iou(masks: list[np.ndarray]) -> float:
    if len(masks) < 2:
        return 0.0
    values: list[float] = []
    for index, first in enumerate(masks):
        for second in masks[index + 1 :]:
            union = np.logical_or(first, second).sum()
            if union == 0:
                continue
            intersection = np.logical_and(first, second).sum()
            values.append(float(intersection / union))
    return float(np.mean(values)) if values else 0.0


def mask_rectangularity(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0.0
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return float(mask.sum() / max(bbox_area, 1))


def make_box_guided_candidate(
    gray: np.ndarray,
    counting_mask: np.ndarray,
    annotation: pd.Series,
    cfg: Config,
) -> tuple[np.ndarray, tuple[int, int, int, int], dict[str, Any], dict[str, np.ndarray]]:
    inner_box = clip_box(annotation, gray.shape)
    crop_box = expand_box(
        inner_box,
        gray.shape,
        cfg.bbox_padding_ratio,
        cfg.bbox_padding_min_px,
    )
    cx0, cy0, cx1, cy1 = crop_box
    crop = gray[cy0:cy1, cx0:cx1]
    valid = counting_mask[cy0:cy1, cx0:cx1].astype(bool)
    allowed = box_mask_in_crop(
        inner_box,
        crop_box,
        crop.shape,
        cfg.bbox_constraint_dilation_px,
    ) & valid

    method_masks: dict[str, np.ndarray] = {}
    method_scores: dict[str, float] = {}
    threshold_values: dict[str, float] = {}
    component_metrics: dict[str, dict[str, float]] = {}

    for method in cfg.methods:
        raw, threshold = threshold_crop(crop, valid, method, cfg)
        selected, score, metrics = select_best_component(
            raw,
            inner_box,
            crop_box,
            allowed,
            cfg,
        )
        method_masks[method] = selected
        method_scores[method] = score
        threshold_values[method] = threshold
        component_metrics[method] = metrics

    nonempty = [mask for mask in method_masks.values() if np.any(mask)]
    votes = np.sum(np.stack(list(method_masks.values()), axis=0), axis=0)
    consensus = votes >= cfg.min_consensus_votes
    used_fallback = False
    selected_method = "consensus"

    if not np.any(consensus):
        nonempty_methods = [
            method for method, mask in method_masks.items() if np.any(mask)
        ]
        if nonempty_methods:
            selected_method = max(nonempty_methods, key=lambda name: method_scores[name])
            consensus = method_masks[selected_method].copy()
            used_fallback = True
        else:
            selected_method = "none"
            consensus = np.zeros_like(crop, dtype=bool)

    consensus = clean_mask(consensus & allowed, cfg)
    pair_iou = pairwise_iou(nonempty)

    x0, y0, x1, y1 = inner_box
    box_area = max((x1 - x0) * (y1 - y0), 1)
    area = int(consensus.sum())
    area_ratio = float(area / box_area)
    rectangularity = mask_rectangularity(consensus)
    n_nonempty = len(nonempty)
    mean_score = (
        float(np.mean([method_scores[name] for name, mask in method_masks.items() if np.any(mask)]))
        if n_nonempty
        else 0.0
    )
    confidence = float(
        np.clip(
            0.40 * mean_score
            + 0.35 * pair_iou
            + 0.25 * (n_nonempty / len(cfg.methods)),
            0.0,
            1.0,
        )
    )

    full_box_like = area_ratio > 0.78 and rectangularity > 0.88
    auto_ok = (
        not used_fallback
        and n_nonempty >= cfg.min_consensus_votes
        and area > 0
        and cfg.auto_min_area_ratio <= area_ratio <= cfg.auto_max_area_ratio
        and pair_iou >= cfg.auto_min_pairwise_iou
        and confidence >= cfg.auto_min_confidence
        and not full_box_like
    )
    review_status = "auto_candidate" if auto_ok else "needs_review"

    metrics: dict[str, Any] = {
        "review_status": review_status,
        "selected_method": selected_method,
        "used_single_method_fallback": used_fallback,
        "n_methods_nonempty": n_nonempty,
        "mean_pairwise_iou": pair_iou,
        "mean_component_score": mean_score,
        "confidence": confidence,
        "mask_area": area,
        "mask_area_ratio_to_box": area_ratio,
        "mask_rectangularity": rectangularity,
        "full_box_like": full_box_like,
    }
    for method in cfg.methods:
        metrics[f"{method}_component_score"] = method_scores[method]
        metrics[f"{method}_threshold"] = threshold_values[method]
        metrics[f"{method}_nonempty"] = bool(np.any(method_masks[method]))
        for key, value in component_metrics[method].items():
            metrics[f"{method}_{key}"] = value

    return consensus, crop_box, metrics, method_masks


def crop_overlay(
    crop_image: np.ndarray,
    mask: np.ndarray,
    inner_box: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
    status: str,
) -> np.ndarray:
    overlay = to_display_bgr(crop_image)
    contours, _ = cv2.findContours(
        (mask.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    color = (0, 200, 0) if status == "auto_candidate" else (0, 165, 255)
    cv2.drawContours(overlay, contours, -1, color, 1, lineType=cv2.LINE_AA)
    x0, y0, x1, y1 = inner_box
    cx0, cy0, _, _ = crop_box
    cv2.rectangle(
        overlay,
        (x0 - cx0, y0 - cy0),
        (x1 - cx0 - 1, y1 - cy0 - 1),
        (255, 0, 255),
        1,
    )
    cv2.putText(
        overlay,
        status,
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
    return overlay


def full_overlay(
    image: np.ndarray,
    instance_mask: np.ndarray,
    annotations: pd.DataFrame,
    records: list[dict[str, Any]],
) -> np.ndarray:
    overlay = to_display_bgr(image)
    record_lookup = {str(row["annotation_id"]): row for row in records}
    for _, annotation in annotations.iterrows():
        annotation_id = str(annotation["annotation_id"])
        record = record_lookup.get(annotation_id, {})
        status = str(record.get("review_status", "needs_review"))
        color = (0, 200, 0) if status == "auto_candidate" else (0, 165, 255)
        x0, y0, x1, y1 = clip_box(annotation, instance_mask.shape)
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, 1)
        label = safe_filename(annotation_id)[:18]
        cv2.putText(
            overlay,
            label,
            (x0, max(12, y0 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    for instance_id in np.unique(instance_mask):
        if instance_id == 0:
            continue
        contour_mask = (instance_mask == instance_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            overlay, contours, -1, (255, 255, 0), 1, lineType=cv2.LINE_AA
        )
    return overlay


def process_image(
    image_id: str,
    image_path: Path,
    image_annotations: pd.DataFrame,
    metadata: pd.DataFrame,
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_image = read_image(image_path)
    gray = to_gray_float(raw_image)
    mask_path = find_counting_mask(
        image_path, cfg.image_dir, cfg.counting_mask_dir
    )
    counting_mask = to_gray_float(read_image(mask_path)) > 0.5
    if counting_mask.shape != gray.shape:
        raise ValueError(
            f"Ukuran counting mask berbeda untuk {image_id}: "
            f"{counting_mask.shape} != {gray.shape}"
        )

    relative = image_path.relative_to(cfg.image_dir)
    background = relative.parent.as_posix()
    split = "unknown"
    if image_id in metadata.index:
        meta = metadata.loc[image_id]
        background = str(meta.get("background", background))
        split = str(meta.get("split", "unknown"))

    instance_mask = np.zeros(gray.shape, dtype=np.uint16)
    owner_score = np.full(gray.shape, -np.inf, dtype=np.float32)
    annotation_records: list[dict[str, Any]] = []

    for instance_id, (_, annotation) in enumerate(
        image_annotations.iterrows(), start=1
    ):
        candidate, crop_box, metrics, _ = make_box_guided_candidate(
            gray, counting_mask, annotation, cfg
        )
        inner_box = clip_box(annotation, gray.shape)
        cx0, cy0, cx1, cy1 = crop_box
        crop_candidate = candidate[: cy1 - cy0, : cx1 - cx0]

        annotation_id = str(annotation["annotation_id"])
        confidence = float(metrics["confidence"])
        x0, y0, x1, y1 = inner_box
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0

        yy, xx = np.mgrid[cy0:cy1, cx0:cx1]
        dx = (xx - center_x) / max((x1 - x0) / 2.0, 1.0)
        dy = (yy - center_y) / max((y1 - y0) / 2.0, 1.0)
        distance_penalty = 0.12 * np.sqrt(dx * dx + dy * dy)
        local_score = confidence - distance_penalty
        global_slice = np.s_[cy0:cy1, cx0:cx1]
        replace_pixels = crop_candidate & (local_score > owner_score[global_slice])
        owner_score_view = owner_score[global_slice]
        instance_view = instance_mask[global_slice]
        owner_score_view[replace_pixels] = local_score[replace_pixels]
        instance_view[replace_pixels] = instance_id

        if cfg.save_annotation_crops:
            crop_image = raw_image[cy0:cy1, cx0:cx1]
            crop_name = (
                f"{safe_filename(image_id)}__{instance_id:04d}__"
                f"{safe_filename(annotation_id)}.png"
            )
            crop_base = Path(split) / background / safe_filename(image_id)
            save_image(
                cfg.output_dir / "annotation_crops" / "images" / crop_base / crop_name,
                crop_image,
            )
            save_image(
                cfg.output_dir / "annotation_crops" / "masks" / crop_base / crop_name,
                candidate.astype(np.uint8) * 255,
            )
            save_image(
                cfg.output_dir / "annotation_crops" / "overlays" / crop_base / crop_name,
                crop_overlay(
                    crop_image,
                    candidate,
                    inner_box,
                    crop_box,
                    str(metrics["review_status"]),
                ),
            )

        record: dict[str, Any] = {
            "image_id": image_id,
            "annotation_id": annotation_id,
            "instance_id": instance_id,
            "split": split,
            "background": background,
            "class_name": annotation.get("class_name", ""),
            "object_type": annotation.get("object_type", ""),
            "source_image": image_path.relative_to(cfg.project_root).as_posix(),
            "source_counting_mask": mask_path.relative_to(cfg.project_root).as_posix(),
            "x": x0,
            "y": y0,
            "width": x1 - x0,
            "height": y1 - y0,
            "crop_x": cx0,
            "crop_y": cy0,
            "crop_width": cx1 - cx0,
            "crop_height": cy1 - cy0,
            **metrics,
        }
        annotation_records.append(record)

    binary_mask = instance_mask > 0
    output_relative = Path(split) / background / f"{image_id}.png"
    binary_path = cfg.output_dir / "pseudo_masks" / "binary" / output_relative
    instance_path = cfg.output_dir / "pseudo_masks" / "instance" / output_relative
    overlay_path = cfg.output_dir / "review_overlays" / output_relative
    save_image(binary_path, binary_mask.astype(np.uint8) * 255)
    save_image(instance_path, instance_mask)
    save_image(
        overlay_path,
        full_overlay(raw_image, instance_mask, image_annotations, annotation_records),
    )

    image_record: dict[str, Any] = {
        "image_id": image_id,
        "split": split,
        "background": background,
        "source_image": image_path.relative_to(cfg.project_root).as_posix(),
        "source_counting_mask": mask_path.relative_to(cfg.project_root).as_posix(),
        "pseudo_binary_mask": binary_path.relative_to(cfg.project_root).as_posix(),
        "pseudo_instance_mask": instance_path.relative_to(cfg.project_root).as_posix(),
        "review_overlay": overlay_path.relative_to(cfg.project_root).as_posix(),
        "height": gray.shape[0],
        "width": gray.shape[1],
        "n_annotations": len(image_annotations),
        "n_nonempty_masks": int(sum(row["mask_area"] > 0 for row in annotation_records)),
        "n_auto_candidates": int(
            sum(row["review_status"] == "auto_candidate" for row in annotation_records)
        ),
        "n_needs_review": int(
            sum(row["review_status"] == "needs_review" for row in annotation_records)
        ),
        "foreground_pixels": int(binary_mask.sum()),
    }
    return annotation_records, image_record


def write_workspace_readme(cfg: Config) -> None:
    text = """PIXEL GROUND-TRUTH WORKSPACE

1. pseudo_masks/binary contains automatic candidate masks (0/255).
2. review_overlays shows boxes and candidate contours.
3. review_queue.csv lists annotations that require priority review.
4. annotation_crops/ contains per-box image, mask, and overlay crops.
5. Do not train U²-Net directly on unreviewed pseudo masks.
6. After manual correction, save final binary masks under:
   approved_masks/binary/<split>/<background>/<image_id>.png
7. Final masks must remain aligned to the original 2048x2048 ROI and contain
   only values 0 and 255.
"""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "README_REVIEW.txt").write_text(text, encoding="utf-8")
    (cfg.output_dir / "approved_masks" / "binary").mkdir(
        parents=True, exist_ok=True
    )


def save_tables(
    annotation_records: list[dict[str, Any]],
    image_records: list[dict[str, Any]],
    failures: list[dict[str, str]],
    cfg: Config,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    annotation_frame = pd.DataFrame(annotation_records)
    image_frame = pd.DataFrame(image_records)
    annotation_frame.to_csv(
        cfg.output_dir / "annotation_mask_manifest.csv", index=False
    )
    image_frame.to_csv(cfg.output_dir / "image_mask_manifest.csv", index=False)
    if not annotation_frame.empty:
        review = annotation_frame[
            annotation_frame["review_status"].eq("needs_review")
        ].sort_values(
            ["confidence", "image_id", "instance_id"],
            ascending=[True, True, True],
        )
        review.to_csv(cfg.output_dir / "review_queue.csv", index=False)
        summary = (
            annotation_frame.groupby(
                ["review_status", "background", "split"], dropna=False
            )
            .agg(
                n_annotations=("annotation_id", "count"),
                mean_confidence=("confidence", "mean"),
                mean_area_ratio=("mask_area_ratio_to_box", "mean"),
                mean_pairwise_iou=("mean_pairwise_iou", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(cfg.output_dir / "mask_generation_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(cfg.output_dir / "review_queue.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(
            cfg.output_dir / "failed_images.csv", index=False
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Membuat pseudo-mask piksel berbasis bounding box untuk direview "
            "sebelum menjadi ground truth U²-Net."
        )
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--counting-mask-dir", type=Path, default=None)
    parser.add_argument("--annotations-csv", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=list(SUPPORTED_METHODS),
    )
    parser.add_argument(
        "--foreground-mode", choices=("bright", "dark"), default="bright"
    )
    parser.add_argument("--min-consensus-votes", type=int, default=2)
    parser.add_argument("--bbox-padding-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-save-annotation-crops",
        action="store_true",
        help="Jangan simpan crop image/mask/overlay per bounding box.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = default_config(args.project_root)
    return replace(
        cfg,
        image_dir=args.image_dir.resolve() if args.image_dir else cfg.image_dir,
        counting_mask_dir=(
            args.counting_mask_dir.resolve()
            if args.counting_mask_dir
            else cfg.counting_mask_dir
        ),
        annotations_csv=(
            args.annotations_csv.resolve()
            if args.annotations_csv
            else cfg.annotations_csv
        ),
        metadata_csv=(
            args.metadata_csv.resolve() if args.metadata_csv else cfg.metadata_csv
        ),
        output_dir=args.output_dir.resolve() if args.output_dir else cfg.output_dir,
        methods=tuple(args.methods),
        foreground_mode=args.foreground_mode,
        min_consensus_votes=args.min_consensus_votes,
        bbox_padding_ratio=args.bbox_padding_ratio,
        save_annotation_crops=not args.no_save_annotation_crops,
    )


def main() -> None:
    print(f"prepare_pixel_ground_truth.py version: {SCRIPT_VERSION}")
    args = parse_args()
    cfg = config_from_args(args)
    validate_config(cfg)

    annotations = load_annotations(cfg.annotations_csv)
    metadata = load_metadata(cfg.metadata_csv)
    image_lookup = build_image_lookup(cfg.image_dir)

    image_ids = [
        image_id
        for image_id in annotations["image_id"].drop_duplicates().tolist()
        if image_id in image_lookup
    ]
    missing_images = sorted(
        set(annotations["image_id"].unique()) - set(image_lookup)
    )
    if missing_images:
        print(
            f"Peringatan: {len(missing_images)} image_id beranotasi tidak "
            "ditemukan di image_dir."
        )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit harus lebih besar dari 0.")
        image_ids = image_ids[: args.limit]
    if not image_ids:
        raise RuntimeError("Tidak ada gambar beranotasi yang dapat diproses.")

    print("Persiapan pixel ground-truth berbasis bounding box")
    print(f"Image input       : {cfg.image_dir}")
    print(f"Counting masks    : {cfg.counting_mask_dir}")
    print(f"Annotations       : {cfg.annotations_csv}")
    print(f"Output            : {cfg.output_dir}")
    print(f"Annotated images  : {len(image_ids)}")
    print(f"Bounding boxes     : {len(annotations):,}")
    print(f"Methods            : {', '.join(cfg.methods)}")
    print(f"Consensus votes    : {cfg.min_consensus_votes}")
    print("Final ground truth : BELUM; hasil wajib direview\n")

    all_annotation_records: list[dict[str, Any]] = []
    all_image_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for image_id in tqdm(image_ids, desc="Membuat pseudo-mask", unit="gambar"):
        try:
            image_annotations = annotations[
                annotations["image_id"].eq(image_id)
            ].copy()
            annotation_records, image_record = process_image(
                image_id=image_id,
                image_path=image_lookup[image_id],
                image_annotations=image_annotations,
                metadata=metadata,
                cfg=cfg,
            )
            all_annotation_records.extend(annotation_records)
            all_image_records.append(image_record)
        except Exception as exc:
            failures.append(
                {
                    "image_id": image_id,
                    "file": str(image_lookup[image_id]),
                    "error": repr(exc),
                }
            )
            tqdm.write(f"Gagal {image_id}: {exc}")

    if not all_image_records:
        raise RuntimeError("Semua gambar gagal diproses.")

    save_tables(
        annotation_records=all_annotation_records,
        image_records=all_image_records,
        failures=failures,
        cfg=cfg,
    )
    write_workspace_readme(cfg)

    n_review = sum(
        row["review_status"] == "needs_review"
        for row in all_annotation_records
    )
    n_auto = sum(
        row["review_status"] == "auto_candidate"
        for row in all_annotation_records
    )
    print("\nSelesai.")
    print(f"Gambar berhasil     : {len(all_image_records)}")
    print(f"Anotasi diproses    : {len(all_annotation_records)}")
    print(f"Auto candidate      : {n_auto}")
    print(f"Needs review        : {n_review}")
    print(f"Gambar gagal        : {len(failures)}")
    print(f"Review queue        : {cfg.output_dir / 'review_queue.csv'}")
    print(f"Review overlays     : {cfg.output_dir / 'review_overlays'}")
    print(
        "Penting: koreksi pseudo_masks/binary dan simpan mask final ke "
        "approved_masks/binary sebelum training U²-Net."
    )


if __name__ == "__main__":
    main()
