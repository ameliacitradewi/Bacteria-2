"""
Baseline segmentasi bakteri berbasis patch.

Pipeline ini sengaja TIDAK menghitung flat-field correction lagi. Input utamanya
adalah hasil final sigma 0.04 dari:

    preprocessed_intensity_sigma040/local_flatfield/

Counting mask dan anotasi harus berasal dari geometri plate yang sama, yaitu:

    processed_plate_strategy_b_circle/

Tahapan:
1. Membaca citra local-flatfield sigma 0.04 dan counting mask pasangannya.
2. Mengekstrak patch overlap dari citra yang sudah dipreproses.
3. Menjalankan Otsu, adaptive/local, dan Sauvola pada setiap patch.
4. Menggabungkan mask patch dengan weighted voting.
5. Melakukan cleanup akhir pada mask gambar penuh.
6. Menyimpan patch, manifest, anotasi patch, mask, overlay, dan metrik.

Contoh:
    python bacteria_segmentation_pipeline.py --limit 2
    python bacteria_segmentation_pipeline.py --methods otsu sauvola
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color, filters, io, measure, morphology, util
from tqdm import tqdm


ForegroundMode = Literal["bright", "dark"]
ThresholdMethod = Literal["otsu", "adaptive", "sauvola"]
SUPPORTED_METHODS: tuple[ThresholdMethod, ...] = (
    "otsu",
    "adaptive",
    "sauvola",
)
PATCH_ANNOTATION_COLUMNS: tuple[str, ...] = (
    "patch_id",
    "image_id",
    "annotation_id",
    "class_name",
    "object_type",
    "x_patch",
    "y_patch",
    "width_patch",
    "height_patch",
    "visible_fraction",
    "is_truncated_by_patch",
    "x_image",
    "y_image",
    "width_image",
    "height_image",
)


@dataclass(frozen=True)
class Config:
    project_root: Path
    input_dir: Path
    counting_mask_dir: Path
    metadata_csv: Path
    annotations_csv: Path
    output_dir: Path

    # Patch 256 dengan overlap 64 menghasilkan stride 192.
    patch_size: int = 256
    patch_overlap: int = 64

    # local_flatfield menggunakan log(background)-log(image), sehingga koloni
    # yang lebih gelap dari medium umumnya tampil lebih terang.
    # Nama folder bright/dark adalah tipe background, BUKAN polaritas objek.
    foreground_mode: ForegroundMode = "bright"
    threshold_methods: tuple[ThresholdMethod, ...] = SUPPORTED_METHODS

    adaptive_block_size: int = 51
    adaptive_offset: float = 0.0
    sauvola_window_size: int = 51
    sauvola_k: float = 0.20

    min_object_area: int = 20
    max_hole_area: int = 20
    morphology_radius: int = 1
    apply_opening: bool = True
    apply_closing: bool = True
    stitch_threshold: float = 0.50

    # Patch rendah variasi tetap diproses secara default agar koloni kecil
    # tidak terbuang tanpa pemeriksaan.
    skip_low_variance_patches: bool = False
    min_patch_std: float = 0.01

    save_patches: bool = True
    save_patch_method_masks: bool = True
    save_comparison_figure: bool = True


def default_config(project_root: Path | None = None) -> Config:
    root = (
        project_root.resolve()
        if project_root is not None
        else Path(__file__).resolve().parent
    )
    return Config(
        project_root=root,
        input_dir=root
        / "preprocessed_intensity_sigma040"
        / "local_flatfield",
        counting_mask_dir=root
        / "processed_plate_strategy_b_circle"
        / "counting_mask",
        metadata_csv=root / "agar_metadata" / "image_manifest.csv",
        annotations_csv=root
        / "processed_plate_strategy_b_circle"
        / "object_annotations_normalized.csv",
        output_dir=root / "results_sigma004_patch",
    )


def validate_config(cfg: Config) -> None:
    required_paths = {
        "input local-flatfield sigma040": cfg.input_dir,
        "counting mask": cfg.counting_mask_dir,
        "metadata": cfg.metadata_csv,
        "anotasi ternormalisasi": cfg.annotations_csv,
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Input pipeline tidak lengkap:\n- " + "\n- ".join(missing)
        )

    if cfg.patch_size < 16:
        raise ValueError("patch_size minimal 16 piksel.")
    if not 0 <= cfg.patch_overlap < cfg.patch_size:
        raise ValueError("patch_overlap harus >= 0 dan lebih kecil dari patch_size.")
    if not 0.0 < cfg.stitch_threshold < 1.0:
        raise ValueError("stitch_threshold harus berada antara 0 dan 1.")
    if cfg.min_object_area < 0 or cfg.max_hole_area < 0:
        raise ValueError("Luas objek/lubang tidak boleh negatif.")
    if cfg.min_patch_std < 0:
        raise ValueError("min_patch_std tidak boleh negatif.")

    for name, value in (
        ("adaptive_block_size", cfg.adaptive_block_size),
        ("sauvola_window_size", cfg.sauvola_window_size),
    ):
        if value < 3 or value % 2 == 0:
            raise ValueError(f"{name} harus ganjil dan minimal 3.")
        if value > cfg.patch_size:
            raise ValueError(f"{name} tidak boleh lebih besar dari patch_size.")


def read_gray(path: Path) -> np.ndarray:
    image = io.imread(path)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = color.rgba2rgb(image)
    image_float = util.img_as_float32(image)
    if image_float.ndim == 3:
        image_float = color.rgb2gray(image_float).astype(np.float32)
    if image_float.ndim != 2:
        raise ValueError(f"Dimensi gambar tidak didukung: {path} -> {image.shape}")
    return np.clip(
        np.nan_to_num(image_float, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    ).astype(np.float32)


def read_binary_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = read_gray(path) > 0.5
    if mask.shape != expected_shape:
        raise ValueError(
            f"Ukuran counting mask tidak sama: {path} "
            f"{mask.shape} != {expected_shape}"
        )
    return mask


def to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == bool:
        return array.astype(np.uint8) * 255
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(path, to_uint8(image), check_contrast=False)


def image_id_from_path(path: Path) -> str:
    return path.stem


def load_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, dtype={"image_id": "string"})
    required = {"image_id", "background", "split"}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Kolom metadata tidak lengkap: {sorted(missing)}")
    metadata = metadata.drop_duplicates("image_id", keep="first").copy()
    return metadata.set_index("image_id", drop=False)


def load_annotations(path: Path) -> pd.DataFrame:
    annotations = pd.read_csv(path, dtype={"image_id": "string"})
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

    if "processing_status" in annotations:
        annotations = annotations[
            annotations["processing_status"].astype(str) == "success"
        ].copy()
    if "center_inside_counting_mask" in annotations:
        annotations = annotations[
            annotations["center_inside_counting_mask"].fillna(False).astype(bool)
        ].copy()
    return annotations


def list_image_pairs(
    cfg: Config,
    metadata: pd.DataFrame,
) -> list[dict[str, object]]:
    images = sorted(cfg.input_dir.rglob("*.png"))
    pairs: list[dict[str, object]] = []
    for image_path in images:
        relative = image_path.relative_to(cfg.input_dir)
        mask_path = cfg.counting_mask_dir / relative
        image_id = image_id_from_path(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Counting mask pasangan tidak ditemukan untuk {relative}: {mask_path}"
            )

        background = relative.parent.as_posix()
        split = "unknown"
        if image_id in metadata.index:
            meta = metadata.loc[image_id]
            split = str(meta["split"])
            expected_background = str(meta["background"])
            if expected_background != background:
                raise ValueError(
                    f"Background metadata tidak cocok untuk {image_id}: "
                    f"{expected_background} != {background}"
                )

        pairs.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "mask_path": mask_path,
                "relative_path": relative,
                "background": background,
                "split": split,
            }
        )
    return pairs


def axis_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def patch_coordinates(
    shape: tuple[int, int],
    patch_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    height, width = shape
    stride = patch_size - overlap
    ys = axis_starts(height, patch_size, stride)
    xs = axis_starts(width, patch_size, stride)
    coordinates: list[tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            valid_height = min(patch_size, height - y)
            valid_width = min(patch_size, width - x)
            coordinates.append((y, x, valid_height, valid_width))
    return coordinates


def extract_padded_patch(
    image: np.ndarray,
    y: int,
    x: int,
    patch_size: int,
) -> tuple[np.ndarray, int, int]:
    height, width = image.shape
    y2 = min(y + patch_size, height)
    x2 = min(x + patch_size, width)
    patch = image[y:y2, x:x2]
    valid_height, valid_width = patch.shape
    pad_y = patch_size - valid_height
    pad_x = patch_size - valid_width
    if pad_y or pad_x:
        mode = "edge" if patch.size else "constant"
        patch = np.pad(patch, ((0, pad_y), (0, pad_x)), mode=mode)
    return patch, valid_height, valid_width


def weighted_window(size: int) -> np.ndarray:
    if size <= 2:
        return np.ones((size, size), dtype=np.float32)
    one_dimensional = np.hanning(size).astype(np.float32)
    window = np.outer(one_dimensional, one_dimensional)
    # Bobot minimum mencegah pembagian nol di batas luar gambar.
    return np.maximum(window, 0.05).astype(np.float32)


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
        np.concatenate(
            (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])
        )
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
        if cfg.apply_opening:
            cleaned = morphology.opening(cleaned, footprint=footprint)
        if cfg.apply_closing:
            cleaned = morphology.closing(cleaned, footprint=footprint)
    return remove_small_components(cleaned, cfg.min_object_area).astype(bool)


def threshold_patch(
    patch: np.ndarray,
    valid_mask: np.ndarray,
    method: ThresholdMethod,
    cfg: Config,
) -> tuple[np.ndarray, float]:
    valid_values = patch[valid_mask]
    if valid_values.size < 2 or float(np.ptp(valid_values)) <= 1e-7:
        return np.zeros_like(valid_mask, dtype=bool), float(
            valid_values[0] if valid_values.size else 0.0
        )

    if method == "otsu":
        threshold = float(filters.threshold_otsu(valid_values))
        threshold_map: float | np.ndarray = threshold
    elif method == "adaptive":
        threshold_map = filters.threshold_local(
            patch,
            block_size=cfg.adaptive_block_size,
            method="gaussian",
            offset=cfg.adaptive_offset,
            mode="reflect",
        )
        threshold = float(np.mean(threshold_map[valid_mask]))
    elif method == "sauvola":
        threshold_map = filters.threshold_sauvola(
            patch,
            window_size=cfg.sauvola_window_size,
            k=cfg.sauvola_k,
        )
        threshold = float(np.mean(threshold_map[valid_mask]))
    else:
        raise ValueError(f"Metode threshold tidak dikenal: {method}")

    if cfg.foreground_mode == "bright":
        raw = patch > threshold_map
    else:
        raw = patch < threshold_map
    return (raw & valid_mask).astype(bool), threshold


def patch_id_for(
    image_id: str,
    y: int,
    x: int,
) -> str:
    return f"{image_id}_y{y:04d}_x{x:04d}"


def intersect_annotations(
    image_annotations: pd.DataFrame,
    patch_id: str,
    y: int,
    x: int,
    valid_height: int,
    valid_width: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    patch_x2 = x + valid_width
    patch_y2 = y + valid_height

    for _, annotation in image_annotations.iterrows():
        box_x1 = float(annotation["x_normalized"])
        box_y1 = float(annotation["y_normalized"])
        box_x2 = box_x1 + float(annotation["width_normalized"])
        box_y2 = box_y1 + float(annotation["height_normalized"])

        intersection_x1 = max(float(x), box_x1)
        intersection_y1 = max(float(y), box_y1)
        intersection_x2 = min(float(patch_x2), box_x2)
        intersection_y2 = min(float(patch_y2), box_y2)
        if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
            continue

        original_area = max((box_x2 - box_x1) * (box_y2 - box_y1), 1e-8)
        intersection_area = (
            (intersection_x2 - intersection_x1)
            * (intersection_y2 - intersection_y1)
        )
        row: dict[str, object] = {
            "patch_id": patch_id,
            "image_id": str(annotation["image_id"]),
            "annotation_id": annotation.get("annotation_id", ""),
            "class_name": annotation.get("class_name", ""),
            "object_type": annotation.get("object_type", ""),
            "x_patch": intersection_x1 - x,
            "y_patch": intersection_y1 - y,
            "width_patch": intersection_x2 - intersection_x1,
            "height_patch": intersection_y2 - intersection_y1,
            "visible_fraction": intersection_area / original_area,
            "is_truncated_by_patch": intersection_area < original_area - 1e-6,
            "x_image": box_x1,
            "y_image": box_y1,
            "width_image": box_x2 - box_x1,
            "height_image": box_y2 - box_y1,
        }
        rows.append(row)
    return rows


def calculate_mask_metrics(
    mask: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float | int]:
    evaluated = mask.astype(bool) & valid_mask.astype(bool)
    labels = measure.label(evaluated, connectivity=2)
    regions = measure.regionprops(labels)
    areas = np.asarray([region.area for region in regions], dtype=np.float64)
    denominator = max(int(valid_mask.sum()), 1)
    return {
        "foreground_fraction_inside_plate": float(evaluated.sum() / denominator),
        "n_objects": int(len(regions)),
        "mean_object_area": float(areas.mean()) if areas.size else 0.0,
        "median_object_area": float(np.median(areas)) if areas.size else 0.0,
        "largest_object_area": float(areas.max()) if areas.size else 0.0,
    }


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], colors="red", linewidths=0.6)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_comparison(
    image: np.ndarray,
    counting_mask: np.ndarray,
    masks: dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    n_columns = 2 + len(masks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, n_columns, figsize=(5 * n_columns, 5))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Local flat-field sigma 0.04")
    axes[1].imshow(counting_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Counting mask")
    for axis, (method, mask) in zip(axes[2:], masks.items()):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        if np.any(mask):
            axis.contour(
                mask.astype(float),
                levels=[0.5],
                colors="red",
                linewidths=0.5,
            )
        axis.set_title(f"{method}: stitched")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def process_one_image(
    pair: dict[str, object],
    cfg: Config,
    annotations: pd.DataFrame,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    image_id = str(pair["image_id"])
    background = str(pair["background"])
    split = str(pair["split"])
    image_path = Path(pair["image_path"])
    mask_path = Path(pair["mask_path"])

    image = read_gray(image_path)
    counting_mask = read_binary_mask(mask_path, image.shape)
    coordinates = patch_coordinates(
        image.shape,
        cfg.patch_size,
        cfg.patch_overlap,
    )
    window = weighted_window(cfg.patch_size)

    accumulators = {
        method: {
            "raw_sum": np.zeros(image.shape, dtype=np.float32),
            "clean_sum": np.zeros(image.shape, dtype=np.float32),
            "weight": np.zeros(image.shape, dtype=np.float32),
        }
        for method in cfg.threshold_methods
    }

    image_annotations = annotations[annotations["image_id"] == image_id]
    patch_records: list[dict[str, object]] = []
    patch_annotation_records: list[dict[str, object]] = []
    patch_metric_records: list[dict[str, object]] = []

    for patch_index, (y, x, valid_height, valid_width) in enumerate(coordinates):
        patch, _, _ = extract_padded_patch(image, y, x, cfg.patch_size)
        mask_patch, _, _ = extract_padded_patch(
            counting_mask.astype(np.uint8),
            y,
            x,
            cfg.patch_size,
        )
        mask_patch = mask_patch > 0
        patch_id = patch_id_for(image_id, y, x)

        valid_rectangle = np.zeros_like(mask_patch, dtype=bool)
        valid_rectangle[:valid_height, :valid_width] = True
        valid_mask = mask_patch & valid_rectangle
        valid_values = patch[valid_mask]
        patch_mean = float(valid_values.mean()) if valid_values.size else 0.0
        patch_std = float(valid_values.std()) if valid_values.size else 0.0
        low_variance = patch_std < cfg.min_patch_std
        processed = bool(valid_values.size and not (
            cfg.skip_low_variance_patches and low_variance
        ))

        annotation_rows = intersect_annotations(
            image_annotations=image_annotations,
            patch_id=patch_id,
            y=y,
            x=x,
            valid_height=valid_height,
            valid_width=valid_width,
        )
        patch_annotation_records.extend(annotation_rows)

        patch_records.append(
            {
                "patch_id": patch_id,
                "image_id": image_id,
                "patch_index": patch_index,
                "split": split,
                "background": background,
                "source_image": image_path.relative_to(cfg.project_root).as_posix(),
                "source_counting_mask": mask_path.relative_to(
                    cfg.project_root
                ).as_posix(),
                "y": y,
                "x": x,
                "patch_size": cfg.patch_size,
                "overlap": cfg.patch_overlap,
                "stride": cfg.patch_size - cfg.patch_overlap,
                "valid_height": valid_height,
                "valid_width": valid_width,
                "valid_plate_fraction": float(valid_mask.mean()),
                "mean_inside_plate": patch_mean,
                "std_inside_plate": patch_std,
                "low_variance": low_variance,
                "processed": processed,
                "n_box_annotations": len(annotation_rows),
                "has_box_annotation": bool(annotation_rows),
            }
        )

        if cfg.save_patches:
            patch_base = Path(split) / background / f"{patch_id}.png"
            save_image(cfg.output_dir / "patches" / "images" / patch_base, patch)
            save_image(
                cfg.output_dir / "patches" / "counting_masks" / patch_base,
                valid_mask,
            )

        for method in cfg.threshold_methods:
            if processed:
                raw_patch, threshold = threshold_patch(
                    patch=patch,
                    valid_mask=valid_mask,
                    method=method,
                    cfg=cfg,
                )
                clean_patch = clean_mask(raw_patch, cfg) & valid_mask
            else:
                raw_patch = np.zeros_like(valid_mask, dtype=bool)
                clean_patch = np.zeros_like(valid_mask, dtype=bool)
                threshold = float("nan")

            if cfg.save_patch_method_masks:
                patch_base = Path(split) / background / f"{patch_id}.png"
                save_image(
                    cfg.output_dir
                    / "patches"
                    / "masks_raw"
                    / method
                    / patch_base,
                    raw_patch,
                )
                save_image(
                    cfg.output_dir
                    / "patches"
                    / "masks_clean"
                    / method
                    / patch_base,
                    clean_patch,
                )

            patch_metric_records.append(
                {
                    "patch_id": patch_id,
                    "image_id": image_id,
                    "split": split,
                    "background": background,
                    "method": method,
                    "threshold_value_or_mean": threshold,
                    "processed": processed,
                    **calculate_mask_metrics(clean_patch, valid_mask),
                }
            )

            valid_window = window[:valid_height, :valid_width]
            image_slice = np.s_[y : y + valid_height, x : x + valid_width]
            accumulators[method]["raw_sum"][image_slice] += (
                raw_patch[:valid_height, :valid_width] * valid_window
            )
            accumulators[method]["clean_sum"][image_slice] += (
                clean_patch[:valid_height, :valid_width] * valid_window
            )
            accumulators[method]["weight"][image_slice] += (
                valid_mask[:valid_height, :valid_width] * valid_window
            )

    image_metric_records: list[dict[str, object]] = []
    final_masks: dict[str, np.ndarray] = {}
    relative_output = Path(background) / f"{image_id}.png"

    for method, accumulator in accumulators.items():
        weight = accumulator["weight"]
        raw_probability = np.divide(
            accumulator["raw_sum"],
            weight,
            out=np.zeros_like(weight),
            where=weight > 0,
        )
        clean_probability = np.divide(
            accumulator["clean_sum"],
            weight,
            out=np.zeros_like(weight),
            where=weight > 0,
        )
        stitched_raw = (raw_probability >= cfg.stitch_threshold) & counting_mask
        stitched_vote = (
            clean_probability >= cfg.stitch_threshold
        ) & counting_mask
        final_mask = clean_mask(stitched_vote, cfg) & counting_mask
        final_masks[method] = final_mask

        save_image(
            cfg.output_dir
            / "stitch_probability_raw"
            / method
            / relative_output,
            raw_probability,
        )
        save_image(
            cfg.output_dir
            / "stitch_probability_clean"
            / method
            / relative_output,
            clean_probability,
        )
        save_image(
            cfg.output_dir / "masks_raw" / method / relative_output,
            stitched_raw,
        )
        save_image(
            cfg.output_dir / "masks_clean" / method / relative_output,
            final_mask,
        )
        create_overlay(
            image=image,
            mask=final_mask,
            output_path=cfg.output_dir
            / "overlays"
            / method
            / relative_output,
            title=(
                f"{image_id} | {method} | foreground={cfg.foreground_mode}"
            ),
        )
        image_metric_records.append(
            {
                "image_id": image_id,
                "split": split,
                "background": background,
                "method": method,
                "height": image.shape[0],
                "width": image.shape[1],
                "n_patches": len(coordinates),
                "foreground_mode": cfg.foreground_mode,
                **calculate_mask_metrics(final_mask, counting_mask),
            }
        )

    if cfg.save_comparison_figure:
        create_comparison(
            image=image,
            counting_mask=counting_mask,
            masks=final_masks,
            output_path=cfg.output_dir
            / "comparisons"
            / background
            / f"{image_id}.png",
            title=f"Image {image_id} | split={split} | background={background}",
        )

    return (
        patch_records,
        patch_annotation_records,
        patch_metric_records,
        image_metric_records,
    )


def save_tables(
    patch_records: list[dict[str, object]],
    patch_annotations: list[dict[str, object]],
    patch_metrics: list[dict[str, object]],
    image_metrics: list[dict[str, object]],
    failures: list[dict[str, str]],
    cfg: Config,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(patch_records).to_csv(
        cfg.output_dir / "patch_manifest.csv",
        index=False,
    )
    pd.DataFrame(
        patch_annotations,
        columns=PATCH_ANNOTATION_COLUMNS,
    ).to_csv(
        cfg.output_dir / "patch_annotations.csv",
        index=False,
    )

    patch_metrics_frame = pd.DataFrame(patch_metrics)
    image_metrics_frame = pd.DataFrame(image_metrics)
    if not patch_metrics_frame.empty:
        patch_metrics_frame.to_csv(
            cfg.output_dir / "segmentation_metrics_per_patch.csv",
            index=False,
        )
    if not image_metrics_frame.empty:
        image_metrics_frame.to_csv(
            cfg.output_dir / "segmentation_metrics_per_image.csv",
            index=False,
        )
        summary = (
            image_metrics_frame.groupby(
                ["method", "background", "split"],
                dropna=False,
            )
            .agg(
                n_images=("image_id", "count"),
                mean_foreground_fraction=(
                    "foreground_fraction_inside_plate",
                    "mean",
                ),
                mean_n_objects=("n_objects", "mean"),
                median_n_objects=("n_objects", "median"),
                mean_object_area=("mean_object_area", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(
            cfg.output_dir / "segmentation_summary_by_method.csv",
            index=False,
        )

    if failures:
        pd.DataFrame(failures).to_csv(
            cfg.output_dir / "failed_images.csv",
            index=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segmentasi patch dari local-flatfield sigma 0.04 tanpa "
            "mengulang flat-field correction."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--counting-mask-dir", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--annotations-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patch-overlap", type=int, default=64)
    parser.add_argument(
        "--foreground-mode",
        choices=("bright", "dark"),
        default="bright",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=list(SUPPORTED_METHODS),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-save-patches",
        action="store_true",
        help="Jangan simpan image/counting-mask patch.",
    )
    parser.add_argument(
        "--no-save-patch-method-masks",
        action="store_true",
        help="Jangan simpan raw/clean mask setiap metode untuk setiap patch.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = default_config(args.project_root)
    return replace(
        cfg,
        input_dir=(args.input_dir.resolve() if args.input_dir else cfg.input_dir),
        counting_mask_dir=(
            args.counting_mask_dir.resolve()
            if args.counting_mask_dir
            else cfg.counting_mask_dir
        ),
        metadata_csv=(
            args.metadata_csv.resolve()
            if args.metadata_csv
            else cfg.metadata_csv
        ),
        annotations_csv=(
            args.annotations_csv.resolve()
            if args.annotations_csv
            else cfg.annotations_csv
        ),
        output_dir=(
            args.output_dir.resolve()
            if args.output_dir
            else cfg.output_dir
        ),
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
        foreground_mode=args.foreground_mode,
        threshold_methods=tuple(args.methods),
        save_patches=not args.no_save_patches,
        save_patch_method_masks=not args.no_save_patch_method_masks,
    )


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    validate_config(cfg)

    metadata = load_metadata(cfg.metadata_csv)
    annotations = load_annotations(cfg.annotations_csv)
    pairs = list_image_pairs(cfg, metadata)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit harus lebih besar dari 0.")
        pairs = pairs[: args.limit]
    if not pairs:
        raise FileNotFoundError(f"Tidak ada PNG di {cfg.input_dir}")

    print("Pipeline segmentasi patch")
    print(f"Input sigma040      : {cfg.input_dir}")
    print(f"Counting mask       : {cfg.counting_mask_dir}")
    print(f"Output              : {cfg.output_dir}")
    print(f"Jumlah gambar       : {len(pairs)}")
    print(
        f"Patch               : {cfg.patch_size} "
        f"(overlap {cfg.patch_overlap}, "
        f"stride {cfg.patch_size - cfg.patch_overlap})"
    )
    print(f"Metode              : {', '.join(cfg.threshold_methods)}")
    print(f"Polaritas foreground: {cfg.foreground_mode}")
    print("Flat-field ulang    : TIDAK\n")

    patch_records: list[dict[str, object]] = []
    patch_annotations: list[dict[str, object]] = []
    patch_metrics: list[dict[str, object]] = []
    image_metrics: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for pair in tqdm(pairs, desc="Memproses gambar", unit="gambar"):
        try:
            (
                records,
                annotation_rows,
                patch_metric_rows,
                image_metric_rows,
            ) = process_one_image(
                pair=pair,
                cfg=cfg,
                annotations=annotations,
            )
            patch_records.extend(records)
            patch_annotations.extend(annotation_rows)
            patch_metrics.extend(patch_metric_rows)
            image_metrics.extend(image_metric_rows)
        except Exception as exc:
            failures.append(
                {
                    "image_id": str(pair["image_id"]),
                    "file": str(pair["image_path"]),
                    "error": repr(exc),
                }
            )
            tqdm.write(f"Gagal {pair['image_id']}: {exc}")

    if not patch_records:
        raise RuntimeError(
            "Semua gambar gagal diproses. Periksa pesan error di atas."
        )

    save_tables(
        patch_records=patch_records,
        patch_annotations=patch_annotations,
        patch_metrics=patch_metrics,
        image_metrics=image_metrics,
        failures=failures,
        cfg=cfg,
    )

    print("\nPipeline selesai.")
    print(f"Patch manifest      : {cfg.output_dir / 'patch_manifest.csv'}")
    print(f"Anotasi patch       : {cfg.output_dir / 'patch_annotations.csv'}")
    print(f"Gambar gagal       : {len(failures)}")
    if patch_annotations:
        print(
            "Catatan             : anotasi yang tersedia berupa bounding box, "
            "bukan mask piksel."
        )


if __name__ == "__main__":
    main()
