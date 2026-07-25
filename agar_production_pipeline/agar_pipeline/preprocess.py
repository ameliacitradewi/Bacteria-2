from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from .common import read_image, save_json, write_image


def _import_project_module(project_root: Path, module_name: str):
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return importlib.import_module(module_name)


def _copy_first_five(source: Path, destination: Path, patterns: tuple[str, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for pattern in patterns:
        files.extend(source.rglob(pattern))
    for index, path in enumerate(sorted(set(files))[:5], start=1):
        shutil.copy2(path, destination / f"{index:02d}_{path.name}")




def _has_five_images(path: Path) -> bool:
    return len(list(path.glob("*"))) >= 5


def _generate_preprocess_visuals(
    dataset_root: Path,
    metadata_dir: Path,
    plate_dir: Path,
    intensity_dir: Path,
    visual_root: Path,
) -> None:
    outer_dir = visual_root / "01_outer_plate"
    roi_dir = visual_root / "02_normalized_roi"
    intensity_visual_dir = visual_root / "03_intensity_flatfield"
    outer_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)
    intensity_visual_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(metadata_dir / "image_manifest.csv", dtype={"image_id": "string"})
    detections = pd.read_csv(plate_dir / "plate_detection_strategy_b.csv", dtype={"image_id": "string"})
    merged = manifest.merge(detections, on="image_id", how="inner")
    merged = merged[merged["processing_status"].astype(str) == "success"].head(5)

    if not _has_five_images(outer_dir):
        for index, (_, row) in enumerate(merged.iterrows(), start=1):
            image = read_image(dataset_root / str(row["image_path"]))
            center = (int(round(row["ellipse_center_x"])), int(round(row["ellipse_center_y"])))
            expansion = float(row.get("physical_expansion", 1.0))
            axes = (
                int(round(float(row["ellipse_axis_1"]) * expansion / 2)),
                int(round(float(row["ellipse_axis_2"]) * expansion / 2)),
            )
            cv2.ellipse(image, center, axes, float(row["ellipse_angle"]), 0, 360, (0, 0, 255), 5)
            cv2.putText(
                image,
                f"{row['image_id']} | outer plate",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            write_image(outer_dir / f"{index:02d}_{row['image_id']}.jpg", image)

    if not _has_five_images(roi_dir):
        for index, (_, row) in enumerate(merged.iterrows(), start=1):
            source = plate_dir / str(row["normalized_raw_path"])
            if source.exists():
                shutil.copy2(source, roi_dir / f"{index:02d}_{row['image_id']}{source.suffix}")

    if not _has_five_images(intensity_visual_dir):
        intensity = pd.read_csv(intensity_dir / "intensity_metrics.csv", dtype={"image_id": "string"})
        intensity = intensity[intensity["processing_status"].astype(str) == "success"].head(5)
        detection_index = detections.set_index("image_id", drop=False)
        for index, (_, row) in enumerate(intensity.iterrows(), start=1):
            image_id = str(row["image_id"])
            flat = read_image(intensity_dir / str(row["local_flatfield_output_path"]), cv2.IMREAD_GRAYSCALE)
            flat_bgr = cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)
            if image_id in detection_index.index:
                raw = read_image(plate_dir / str(detection_index.loc[image_id, "normalized_raw_path"]))
                raw = cv2.resize(raw, (flat.shape[1], flat.shape[0]))
                montage = cv2.hconcat([raw, flat_bgr])
            else:
                montage = flat_bgr
            cv2.putText(
                montage,
                f"{image_id} | normalized ROI -> flat-field sigma 0.04",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            write_image(intensity_visual_dir / f"{index:02d}_{image_id}.jpg", montage)


def run_preprocessing(
    config: dict[str, Any],
    force_manifest: bool = False,
    force_plate: bool = False,
    force_intensity: bool = False,
) -> dict[str, str]:
    project_root = Path(config["project_root"])
    dataset_root = Path(config["dataset_root"])
    metadata_dir = Path(config["metadata_dir"])
    plate_dir = Path(config["classical_plate_dir"])
    intensity_dir = Path(config["intensity_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    build_module = _import_project_module(project_root, "build_agar_manifest")
    plate_module = _import_project_module(project_root, "detect_outer_plate_strategy_b")
    intensity_module = _import_project_module(project_root, "normalize_agar_intensity_v1")

    manifest_path = metadata_dir / "image_manifest.csv"
    object_path = metadata_dir / "object_annotations.csv"
    if force_manifest or not manifest_path.exists() or not object_path.exists():
        build_module.build_manifest(dataset_root, metadata_dir)

    detection_csv = plate_dir / "plate_detection_strategy_b.csv"
    normalized_annotations = plate_dir / "object_annotations_normalized.csv"
    plate_cfg = config.get("classical_plate", {})
    if force_plate or not detection_csv.exists() or not normalized_annotations.exists():
        plate_module.process_dataset(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            objects_path=object_path,
            output_root=plate_dir,
            detection_max_side=int(plate_cfg.get("detection_max_side", 1400)),
            target_size=int(plate_cfg.get("target_size", 2048)),
            physical_expansion=float(plate_cfg.get("physical_expansion", 1.01)),
            counting_scale=float(plate_cfg.get("counting_scale", 1.0)),
            crop_padding_ratio=float(plate_cfg.get("crop_padding_ratio", 1.04)),
            radial_min_ratio=float(plate_cfg.get("radial_min_ratio", 0.82)),
            radial_max_ratio=float(plate_cfg.get("radial_max_ratio", 1.25)),
            n_angles=int(plate_cfg.get("n_angles", 720)),
            limit=plate_cfg.get("limit"),
            debug_limit=5,
        )

    intensity_csv = intensity_dir / "intensity_metrics.csv"
    intensity_cfg = config.get("intensity", {})
    if force_intensity or not intensity_csv.exists():
        intensity_module.process_dataset(
            input_root=plate_dir,
            detections_path=detection_csv,
            output_root=intensity_dir,
            input_variant=str(intensity_cfg.get("input_variant", "normalized_raw")),
            safe_erode_ratio=float(intensity_cfg.get("safe_erode_ratio", 0.025)),
            outlier_mad_multiplier=float(intensity_cfg.get("outlier_mad", 3.5)),
            safe_minimum_fraction=float(
                intensity_cfg.get("safe_minimum_fraction", 0.15)
            ),
            flatfield_sigma_ratio=float(
                intensity_cfg.get("flatfield_sigma_ratio", 0.04)
            ),
            percentile_low=float(intensity_cfg.get("percentile_low", 1.0)),
            percentile_high=float(intensity_cfg.get("percentile_high", 99.0)),
            limit=intensity_cfg.get("limit"),
            debug_limit=5,
        )

    visual_root = output_dir / "visual_samples"
    _copy_first_five(plate_dir / "debug", visual_root / "01_outer_plate", ("*.jpg", "*.png"))
    _copy_first_five(
        plate_dir / "plate_crop_normalized",
        visual_root / "02_normalized_roi",
        ("*.jpg", "*.png"),
    )
    _copy_first_five(
        intensity_dir / "preview",
        visual_root / "03_intensity_flatfield",
        ("*.jpg", "*.png"),
    )
    _generate_preprocess_visuals(
        dataset_root=dataset_root,
        metadata_dir=metadata_dir,
        plate_dir=plate_dir,
        intensity_dir=intensity_dir,
        visual_root=visual_root,
    )

    detection_frame = pd.read_csv(detection_csv)
    status_summary = (
        detection_frame["processing_status"].value_counts(dropna=False).to_dict()
    )
    summary = {
        "manifest": str(manifest_path),
        "objects": str(object_path),
        "plate_detection": str(detection_csv),
        "normalized_annotations": str(normalized_annotations),
        "intensity_metrics": str(intensity_csv),
        "plate_processing_status": status_summary,
    }
    save_json(output_dir / "metadata" / "preprocessing_summary.json", summary)
    return {key: str(value) for key, value in summary.items() if isinstance(value, str)}
