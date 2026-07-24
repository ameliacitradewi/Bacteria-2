"""Finalize box-guided pseudo-masks automatically, without manual review.

This script consumes the workspace produced by prepare_pixel_ground_truth.py.
It automatically classifies every annotation as accepted_high, accepted_medium,
or ignored. Accepted annotation masks are merged into one full-resolution binary
pseudo-label per image. Ignored boxes are excluded from training through a
valid-region mask and a pixel-weight map.

Important terminology:
- Outputs are pseudo-labels / weak labels, not manually verified ground truth.
- Dice/IoU measured against these masks evaluate agreement with pseudo-labels.

Default usage from the Bacteria-2 project root:
    python auto_finalize_pseudolabels.py \
        --workspace pixel_ground_truth_workspace \
        --policy balanced

Policies:
- strict: only high-consensus candidates are accepted.
- balanced: high-consensus and plausible medium-confidence candidates accepted.
- permissive: every non-empty, non-box-like plausible candidate accepted.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

Policy = Literal["strict", "balanced", "permissive"]
SCRIPT_VERSION = "2026-07-24-auto-pseudolabel-v1"


@dataclass(frozen=True)
class Config:
    project_root: Path
    workspace: Path
    output_dir: Path
    policy: Policy = "balanced"

    min_area_ratio: float = 0.001
    max_area_ratio: float = 0.90

    strict_min_iou: float = 0.30
    strict_min_confidence: float = 0.50
    strict_min_component_score: float = 0.50

    balanced_min_confidence: float = 0.45
    balanced_min_component_score: float = 0.50
    balanced_min_methods: int = 2

    high_weight: int = 255
    medium_weight: int = 160
    outside_boxes_are_background: bool = True
    save_overlays: bool = True


def safe_filename(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._") or "annotation"


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def finite_int(value: Any, default: int = 0) -> int:
    return int(round(finite_float(value, default)))


def read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise RuntimeError(f"Gagal membaca gambar: {path}")
    return image


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 5]
    )
    if not ok:
        raise RuntimeError(f"Gagal meng-encode gambar: {path}")
    encoded.tofile(str(path))


def to_binary(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image > 0


def to_display_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype == np.uint8:
        return image.copy()
    return np.clip(image, 0, 255).astype(np.uint8)


def validate_config(cfg: Config) -> None:
    required = [
        cfg.workspace / "annotation_mask_manifest.csv",
        cfg.workspace / "image_mask_manifest.csv",
        cfg.workspace / "annotation_crops" / "masks",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Workspace pseudo-mask belum lengkap:\n- " + "\n- ".join(missing)
        )
    if not 0 <= cfg.min_area_ratio < cfg.max_area_ratio <= 2.0:
        raise ValueError("Rentang area ratio tidak valid.")
    for name, value in (
        ("high_weight", cfg.high_weight),
        ("medium_weight", cfg.medium_weight),
    ):
        if not 0 <= value <= 255:
            raise ValueError(f"{name} harus berada pada 0..255.")


def auto_decision(row: pd.Series, cfg: Config) -> tuple[str, str, int]:
    """Return status, reason, and pixel training weight."""
    area = finite_int(row.get("mask_area"))
    ratio = finite_float(row.get("mask_area_ratio_to_box"))
    full_box_like = parse_bool(row.get("full_box_like"))
    selected_method = str(row.get("selected_method", "none"))
    confidence = finite_float(row.get("confidence"))
    pair_iou = finite_float(row.get("mean_pairwise_iou"))
    component_score = finite_float(row.get("mean_component_score"))
    n_methods = finite_int(row.get("n_methods_nonempty"))
    used_fallback = parse_bool(row.get("used_single_method_fallback"))
    review_status = str(row.get("review_status", ""))

    if area <= 0:
        return "ignored", "empty_mask", 0
    if selected_method == "none":
        return "ignored", "no_selected_method", 0
    if full_box_like:
        return "ignored", "full_box_like", 0
    if ratio < cfg.min_area_ratio:
        return "ignored", "area_ratio_too_small", 0
    if ratio > cfg.max_area_ratio:
        return "ignored", "area_ratio_too_large", 0

    high = (
        not used_fallback
        and n_methods >= 2
        and pair_iou >= cfg.strict_min_iou
        and confidence >= cfg.strict_min_confidence
        and component_score >= cfg.strict_min_component_score
    )
    if high:
        return "accepted_high", "high_consensus", cfg.high_weight

    if cfg.policy == "strict":
        return "ignored", "below_strict_threshold", 0

    medium = (
        n_methods >= cfg.balanced_min_methods
        and confidence >= cfg.balanced_min_confidence
        and component_score >= cfg.balanced_min_component_score
    )
    if cfg.policy == "balanced" and medium:
        reason = "balanced_candidate"
        if review_status == "auto_candidate":
            reason = "auto_candidate_balanced"
        return "accepted_medium", reason, cfg.medium_weight

    if cfg.policy == "permissive":
        return "accepted_medium", "permissive_plausible_candidate", cfg.medium_weight

    return "ignored", "below_balanced_threshold", 0


def annotation_crop_path(workspace: Path, row: pd.Series) -> Path:
    image_id = safe_filename(row["image_id"])
    annotation_id = safe_filename(row["annotation_id"])
    instance_id = finite_int(row["instance_id"])
    split = str(row.get("split", "unknown"))
    background = str(row.get("background", "unknown"))
    filename = f"{image_id}__{instance_id:04d}__{annotation_id}.png"
    return (
        workspace
        / "annotation_crops"
        / "masks"
        / split
        / background
        / image_id
        / filename
    )


def resolve_project_path(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else project_root / path


def draw_overlay(
    image: np.ndarray,
    binary_mask: np.ndarray,
    decisions: pd.DataFrame,
) -> np.ndarray:
    overlay = to_display_bgr(image)
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1, cv2.LINE_AA)

    for _, row in decisions.iterrows():
        x0 = finite_int(row.get("x"))
        y0 = finite_int(row.get("y"))
        x1 = x0 + max(finite_int(row.get("width")), 1)
        y1 = y0 + max(finite_int(row.get("height")), 1)
        status = str(row.get("auto_status", "ignored"))
        if status == "accepted_high":
            color = (0, 200, 0)
        elif status == "accepted_medium":
            color = (0, 200, 255)
        else:
            color = (0, 0, 255)
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, 1)
    return overlay


def process_one_image(
    image_row: pd.Series,
    annotations: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    image_id = str(image_row["image_id"])
    height = finite_int(image_row.get("height"))
    width = finite_int(image_row.get("width"))
    if height <= 0 or width <= 0:
        raise ValueError(f"Ukuran gambar tidak valid untuk {image_id}")

    source_image = resolve_project_path(cfg.project_root, image_row["source_image"])
    source_counting = resolve_project_path(
        cfg.project_root, image_row["source_counting_mask"]
    )
    raw_image = read_image(source_image, cv2.IMREAD_COLOR)
    counting = to_binary(read_image(source_counting, cv2.IMREAD_UNCHANGED))
    if counting.shape != (height, width):
        raise ValueError(
            f"Ukuran counting mask {counting.shape} tidak sama dengan "
            f"{(height, width)} untuk {image_id}"
        )

    binary = np.zeros((height, width), dtype=bool)
    instance = np.zeros((height, width), dtype=np.uint16)
    owner_confidence = np.full((height, width), -1.0, dtype=np.float32)

    if cfg.outside_boxes_are_background:
        weights = counting.astype(np.uint8) * 255
    else:
        weights = np.zeros((height, width), dtype=np.uint8)

    decisions: list[dict[str, Any]] = []
    image_annotations = annotations[annotations["image_id"].astype(str).eq(image_id)]

    for _, row in image_annotations.iterrows():
        status, reason, training_weight = auto_decision(row, cfg)
        crop_path = annotation_crop_path(cfg.workspace, row)

        cx0 = finite_int(row.get("crop_x"))
        cy0 = finite_int(row.get("crop_y"))
        crop_width = finite_int(row.get("crop_width"))
        crop_height = finite_int(row.get("crop_height"))
        cx1 = min(cx0 + crop_width, width)
        cy1 = min(cy0 + crop_height, height)

        x0 = max(finite_int(row.get("x")), 0)
        y0 = max(finite_int(row.get("y")), 0)
        x1 = min(x0 + max(finite_int(row.get("width")), 1), width)
        y1 = min(y0 + max(finite_int(row.get("height")), 1), height)

        crop_mask: np.ndarray | None = None
        if not crop_path.exists():
            status, reason, training_weight = "ignored", "crop_mask_missing", 0
        else:
            crop_mask = to_binary(read_image(crop_path, cv2.IMREAD_UNCHANGED))
            expected_shape = (max(cy1 - cy0, 0), max(cx1 - cx0, 0))
            if crop_mask.shape != expected_shape:
                status, reason, training_weight = (
                    "ignored",
                    f"crop_shape_mismatch_{crop_mask.shape}_{expected_shape}",
                    0,
                )
            elif not np.any(crop_mask):
                status, reason, training_weight = "ignored", "empty_crop_mask", 0

        if status == "ignored":
            weights[y0:y1, x0:x1] = 0
        else:
            # The full bounding-box region receives the pseudo-label confidence.
            box_view = weights[y0:y1, x0:x1]
            if cfg.outside_boxes_are_background:
                box_view[:] = np.minimum(box_view, training_weight)
            else:
                box_view[:] = training_weight

            assert crop_mask is not None
            confidence = finite_float(row.get("confidence"))
            global_slice = np.s_[cy0:cy1, cx0:cx1]
            replace = crop_mask & (confidence > owner_confidence[global_slice])
            binary_view = binary[global_slice]
            instance_view = instance[global_slice]
            owner_view = owner_confidence[global_slice]
            binary_view[replace] = True
            instance_view[replace] = finite_int(row.get("instance_id"))
            owner_view[replace] = confidence

        record = row.to_dict()
        record.update(
            {
                "auto_status": status,
                "auto_reason": reason,
                "training_weight": training_weight,
                "annotation_crop_mask": str(crop_path),
            }
        )
        decisions.append(record)

    # Never train outside the counting mask.
    binary &= counting
    instance[~counting] = 0
    weights[~counting] = 0
    valid_region = weights > 0

    split = str(image_row.get("split", "unknown"))
    background = str(image_row.get("background", "unknown"))
    relative = Path(split) / background / f"{safe_filename(image_id)}.png"

    binary_path = cfg.output_dir / "binary" / relative
    instance_path = cfg.output_dir / "instance" / relative
    valid_path = cfg.output_dir / "valid_regions" / relative
    weight_path = cfg.output_dir / "weights" / relative
    overlay_path = cfg.output_dir / "overlays" / relative

    save_png(binary_path, binary.astype(np.uint8) * 255)
    save_png(instance_path, instance)
    save_png(valid_path, valid_region.astype(np.uint8) * 255)
    save_png(weight_path, weights)

    decisions_frame = pd.DataFrame(decisions)
    if cfg.save_overlays:
        save_png(overlay_path, draw_overlay(raw_image, binary, decisions_frame))

    accepted_high = int((decisions_frame["auto_status"] == "accepted_high").sum())
    accepted_medium = int(
        (decisions_frame["auto_status"] == "accepted_medium").sum()
    )
    ignored = int((decisions_frame["auto_status"] == "ignored").sum())

    image_record = {
        "image_id": image_id,
        "split": split,
        "background": background,
        "source_image": str(source_image),
        "source_counting_mask": str(source_counting),
        "auto_binary_mask": str(binary_path),
        "auto_instance_mask": str(instance_path),
        "auto_valid_region": str(valid_path),
        "auto_weight_map": str(weight_path),
        "auto_overlay": str(overlay_path) if cfg.save_overlays else "",
        "height": height,
        "width": width,
        "n_annotations": len(decisions_frame),
        "n_accepted_high": accepted_high,
        "n_accepted_medium": accepted_medium,
        "n_ignored": ignored,
        "foreground_pixels": int(binary.sum()),
        "valid_training_pixels": int(valid_region.sum()),
    }
    return decisions_frame, image_record


def write_readme(cfg: Config) -> None:
    text = f"""AUTOMATIC PSEUDO-LABEL DATASET

Script version: {SCRIPT_VERSION}
Policy: {cfg.policy}

Outputs:
- binary/: automatic binary pseudo-labels (0 background, 255 colony)
- instance/: automatic instance IDs
- valid_regions/: pixels allowed to contribute to training loss
- weights/: per-pixel training weight in the range 0..255
- overlays/: automatic quality-control visualization
- annotation_auto_decisions.csv: accepted/ignored decision per bounding box
- image_auto_manifest.csv: paths and counts per image

Status meaning:
- accepted_high: strong consensus; full training weight
- accepted_medium: plausible candidate; reduced training weight
- ignored: excluded automatically; its bounding-box region has weight 0

These masks are pseudo-labels, not manually verified ground truth. Use the
valid-region and weight maps during U-Net/U2-Net training. Report the method as
weakly supervised / pseudo-label training.
"""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "README_AUTO_PSEUDOLABELS.txt").write_text(
        text, encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize pseudo-masks fully automatically without review."
    )
    parser.add_argument(
        "--workspace", type=Path, default=Path("pixel_ground_truth_workspace")
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--policy",
        choices=("strict", "balanced", "permissive"),
        default="balanced",
    )
    parser.add_argument(
        "--outside-boxes-ignore",
        action="store_true",
        help=(
            "Ignore all pixels outside annotated boxes. Default assumes the "
            "bounding-box annotations are exhaustive and uses them as background."
        ),
    )
    parser.add_argument("--no-overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    workspace = (
        args.workspace.resolve()
        if args.workspace.is_absolute()
        else (project_root / args.workspace).resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir and args.output_dir.is_absolute()
        else (
            (project_root / args.output_dir).resolve()
            if args.output_dir
            else workspace / "auto_pseudolabels"
        )
    )
    cfg = Config(
        project_root=project_root,
        workspace=workspace,
        output_dir=output_dir,
        policy=args.policy,
        outside_boxes_are_background=not args.outside_boxes_ignore,
        save_overlays=not args.no_overlays,
    )
    validate_config(cfg)

    print(f"auto_finalize_pseudolabels.py version: {SCRIPT_VERSION}")
    print(f"Workspace : {cfg.workspace}")
    print(f"Policy    : {cfg.policy}")
    print(f"Output    : {cfg.output_dir}")

    annotations = pd.read_csv(cfg.workspace / "annotation_mask_manifest.csv")
    images = pd.read_csv(cfg.workspace / "image_mask_manifest.csv")
    annotations["image_id"] = annotations["image_id"].astype(str)
    images["image_id"] = images["image_id"].astype(str)

    all_decisions: list[pd.DataFrame] = []
    image_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for _, image_row in tqdm(
        images.iterrows(), total=len(images), desc="Auto-finalize", unit="gambar"
    ):
        image_id = str(image_row["image_id"])
        try:
            decisions, image_record = process_one_image(
                image_row=image_row,
                annotations=annotations,
                cfg=cfg,
            )
            all_decisions.append(decisions)
            image_records.append(image_record)
        except Exception as exc:  # continue other images and report failures
            failures.append({"image_id": image_id, "error": repr(exc)})
            tqdm.write(f"Gagal {image_id}: {exc}")

    if not image_records:
        raise RuntimeError("Tidak ada gambar yang berhasil difinalisasi.")

    decisions_frame = pd.concat(all_decisions, ignore_index=True)
    image_frame = pd.DataFrame(image_records)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    decisions_frame.to_csv(
        cfg.output_dir / "annotation_auto_decisions.csv", index=False
    )
    image_frame.to_csv(cfg.output_dir / "image_auto_manifest.csv", index=False)

    summary = (
        decisions_frame.groupby(["auto_status", "auto_reason"], dropna=False)
        .agg(
            n_annotations=("annotation_id", "count"),
            mean_confidence=("confidence", "mean"),
            mean_pairwise_iou=("mean_pairwise_iou", "mean"),
            mean_area_ratio=("mask_area_ratio_to_box", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(cfg.output_dir / "auto_decision_summary.csv", index=False)
    ignored = decisions_frame[decisions_frame["auto_status"].eq("ignored")]
    ignored.to_csv(cfg.output_dir / "auto_ignored_annotations.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(
            cfg.output_dir / "failed_images.csv", index=False
        )
    write_readme(cfg)

    counts = decisions_frame["auto_status"].value_counts()
    print("\nSelesai.")
    print(f"Gambar berhasil   : {len(image_frame)}")
    print(f"Anotasi total     : {len(decisions_frame)}")
    print(f"Accepted high     : {int(counts.get('accepted_high', 0))}")
    print(f"Accepted medium   : {int(counts.get('accepted_medium', 0))}")
    print(f"Ignored otomatis  : {int(counts.get('ignored', 0))}")
    print(f"Gambar gagal      : {len(failures)}")
    print(f"Binary masks      : {cfg.output_dir / 'binary'}")
    print(f"Valid regions     : {cfg.output_dir / 'valid_regions'}")
    print(f"Weight maps       : {cfg.output_dir / 'weights'}")
    print("Tidak ada tahap review manual pada pipeline ini.")


if __name__ == "__main__":
    main()
