from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


EPSILON = 1e-6


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image safely, including paths containing non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise RuntimeError(f"Gagal membaca gambar: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    """Save PNG/JPEG safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".png":
        success, encoded = cv2.imencode(
            ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 4]
        )
    elif suffix in {".jpg", ".jpeg"}:
        success, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
    else:
        raise ValueError(f"Format output tidak didukung: {suffix}")

    if not success:
        raise RuntimeError(f"Gagal meng-encode gambar: {path}")
    encoded.tofile(str(path))


def resolve_relative(root: Path, path_value: Any) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else root / path


def ensure_binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a mask if needed and return uint8 values 0/255."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape[:2] != shape:
        mask = cv2.resize(
            mask,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def robust_percentile_scale(
    values_image: np.ndarray,
    valid_mask: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> tuple[np.ndarray, float, float]:
    """Scale a float image to [0, 1] using percentiles inside valid_mask."""
    valid_values = values_image[valid_mask > 0]
    valid_values = valid_values[np.isfinite(valid_values)]
    if valid_values.size < 32:
        raise RuntimeError("Piksel valid terlalu sedikit untuk percentile scaling.")

    low = float(np.percentile(valid_values, low_percentile))
    high = float(np.percentile(valid_values, high_percentile))
    if high - low < EPSILON:
        high = low + EPSILON

    scaled = np.clip((values_image - low) / (high - low), 0.0, 1.0)
    return scaled.astype(np.float32), low, high


def robust_inside_color(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = image_bgr[mask > 0]
    if pixels.size == 0:
        return np.array([127, 127, 127], dtype=np.uint8)
    return np.rint(np.median(pixels, axis=0)).astype(np.uint8)


def fill_outside_mask(
    image_bgr: np.ndarray,
    counting_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fill_color = robust_inside_color(image_bgr, counting_mask)
    output = image_bgr.copy()
    output[counting_mask == 0] = fill_color
    return output, fill_color


def make_safe_medium_mask(
    image_bgr: np.ndarray,
    counting_mask: np.ndarray,
    erode_ratio: float,
    outlier_mad_multiplier: float,
    minimum_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Create a robust mask used only to estimate the agar background.

    This mask is intentionally conservative. It is NOT the final counting mask.
    It removes the plate rim by erosion and rejects extreme luminance/chroma
    pixels caused by text, colonies, glare, and leaked background.
    """
    height, width = counting_mask.shape
    radius = max(1, int(round(min(height, width) * erode_ratio)))
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    eroded = cv2.erode(counting_mask, kernel, iterations=1)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidate = eroded > 0
    if candidate.sum() < 100:
        candidate = counting_mask > 0

    vectors = lab[candidate]
    median_lab = np.median(vectors, axis=0)
    absolute_deviation = np.abs(vectors - median_lab)
    mad_lab = np.median(absolute_deviation, axis=0)
    mad_lab = np.maximum(mad_lab, np.array([2.0, 1.0, 1.0], dtype=np.float32))

    z_like = absolute_deviation / mad_lab
    robust_keep = np.all(z_like <= outlier_mad_multiplier, axis=1)

    safe = np.zeros_like(counting_mask, dtype=np.uint8)
    candidate_y, candidate_x = np.where(candidate)
    safe[candidate_y[robust_keep], candidate_x[robust_keep]] = 255

    counting_area = int((counting_mask > 0).sum())
    minimum_pixels = int(max(100, minimum_fraction * counting_area))

    # If robust rejection becomes too aggressive, relax luminance/chroma jointly.
    if int((safe > 0).sum()) < minimum_pixels:
        distance = np.sqrt(np.sum(z_like**2, axis=1))
        keep_count = min(len(distance), max(minimum_pixels, 100))
        chosen = np.argpartition(distance, keep_count - 1)[:keep_count]
        safe[:] = 0
        safe[candidate_y[chosen], candidate_x[chosen]] = 255

    stats = {
        "safe_erode_radius": float(radius),
        "safe_mask_pixels": float((safe > 0).sum()),
        "safe_mask_fraction_of_plate": float(
            (safe > 0).sum() / max(counting_area, 1)
        ),
        "safe_median_l": float(median_lab[0]),
        "safe_median_a": float(median_lab[1]),
        "safe_median_b": float(median_lab[2]),
    }
    return safe, stats


def normalized_gaussian_background(
    luminance: np.ndarray,
    valid_background_mask: np.ndarray,
    sigma: float,
    fallback_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a smooth illumination field using normalized convolution."""
    weights = (valid_background_mask > 0).astype(np.float32)
    weighted_signal = luminance * weights

    numerator = cv2.GaussianBlur(
        weighted_signal,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    denominator = cv2.GaussianBlur(
        weights,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )

    background = numerator / np.maximum(denominator, EPSILON)
    weak_support = denominator < 0.01
    background[weak_support] = fallback_value
    background = np.clip(background, 1.0 / 255.0, 1.0)
    return background.astype(np.float32), denominator.astype(np.float32)


def intensity_statistics(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "cv": math.nan,
            "p01": math.nan,
            "p99": math.nan,
        }

    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "mean": mean,
        "median": float(np.median(values)),
        "std": std,
        "cv": float(std / max(abs(mean), EPSILON)),
        "p01": float(np.percentile(values, 1)),
        "p99": float(np.percentile(values, 99)),
    }


def create_preview(
    raw_bgr: np.ndarray,
    safe_mask: np.ndarray,
    global_gray: np.ndarray,
    flatfield_gray: np.ndarray,
    max_side: int = 1600,
) -> np.ndarray:
    """Create a 2x2 preview montage."""
    raw_overlay = raw_bgr.copy()
    boundary, _ = cv2.findContours(
        safe_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(raw_overlay, boundary, -1, (0, 0, 255), 3)

    global_bgr = cv2.cvtColor(global_gray, cv2.COLOR_GRAY2BGR)
    flat_bgr = cv2.cvtColor(flatfield_gray, cv2.COLOR_GRAY2BGR)
    safe_view = np.zeros_like(raw_bgr)
    safe_view[safe_mask > 0] = raw_bgr[safe_mask > 0]

    panels = [raw_overlay, safe_view, global_bgr, flat_bgr]
    panel_h, panel_w = raw_bgr.shape[:2]
    montage = np.zeros((panel_h * 2, panel_w * 2, 3), dtype=np.uint8)
    montage[:panel_h, :panel_w] = panels[0]
    montage[:panel_h, panel_w:] = panels[1]
    montage[panel_h:, :panel_w] = panels[2]
    montage[panel_h:, panel_w:] = panels[3]

    labels = [
        ("raw + safe-mask boundary", 20, 40),
        ("safe medium pixels", panel_w + 20, 40),
        ("global log", 20, panel_h + 40),
        ("local flat-field", panel_w + 20, panel_h + 40),
    ]
    for label, x, y in labels:
        cv2.putText(
            montage,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    largest = max(montage.shape[:2])
    if largest > max_side:
        scale = max_side / largest
        montage = cv2.resize(
            montage,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    return montage


def process_one(
    image_bgr: np.ndarray,
    counting_mask: np.ndarray,
    safe_erode_ratio: float,
    outlier_mad_multiplier: float,
    safe_minimum_fraction: float,
    flatfield_sigma_ratio: float,
    percentile_low: float,
    percentile_high: float,
) -> dict[str, Any]:
    height, width = image_bgr.shape[:2]
    counting_mask = ensure_binary_mask(counting_mask, (height, width))
    if int((counting_mask > 0).sum()) < 100:
        raise RuntimeError("Counting mask kosong atau terlalu kecil.")

    raw_masked, fill_color = fill_outside_mask(image_bgr, counting_mask)
    safe_mask, safe_stats = make_safe_medium_mask(
        image_bgr=raw_masked,
        counting_mask=counting_mask,
        erode_ratio=safe_erode_ratio,
        outlier_mad_multiplier=outlier_mad_multiplier,
        minimum_fraction=safe_minimum_fraction,
    )

    lab = cv2.cvtColor(raw_masked, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0
    safe_values = luminance[safe_mask > 0]
    if safe_values.size < 100:
        raise RuntimeError("Safe medium mask terlalu kecil setelah filtering.")

    medium_i0 = float(np.median(safe_values))

    # Faithful global-log variant from the paper's intensity formula.
    global_log = np.log(medium_i0 + EPSILON) - np.log(luminance + EPSILON)
    global_scaled, global_low, global_high = robust_percentile_scale(
        global_log,
        counting_mask,
        percentile_low,
        percentile_high,
    )
    global_gray = np.rint(global_scaled * 255.0).astype(np.uint8)

    sigma = max(3.0, min(height, width) * flatfield_sigma_ratio)
    background_field, background_support = normalized_gaussian_background(
        luminance=luminance,
        valid_background_mask=safe_mask,
        sigma=sigma,
        fallback_value=medium_i0,
    )
    flatfield_log = np.log(background_field + EPSILON) - np.log(
        luminance + EPSILON
    )
    flatfield_scaled, flatfield_low, flatfield_high = robust_percentile_scale(
        flatfield_log,
        counting_mask,
        percentile_low,
        percentile_high,
    )
    flatfield_gray = np.rint(flatfield_scaled * 255.0).astype(np.uint8)

    # Ensure no artificial black boundary outside the plate.
    global_fill = int(round(float(np.median(global_gray[safe_mask > 0]))))
    flatfield_fill = int(round(float(np.median(flatfield_gray[safe_mask > 0]))))
    global_gray[counting_mask == 0] = global_fill
    flatfield_gray[counting_mask == 0] = flatfield_fill

    raw_stats = intensity_statistics(luminance[safe_mask > 0])
    global_residual_stats = intensity_statistics(global_log[safe_mask > 0])
    flatfield_residual_stats = intensity_statistics(flatfield_log[safe_mask > 0])
    background_stats = intensity_statistics(background_field[safe_mask > 0])

    return {
        "raw_masked": raw_masked,
        "safe_mask": safe_mask,
        "global_gray": global_gray,
        "flatfield_gray": flatfield_gray,
        "background_field": background_field,
        "background_support": background_support,
        "fill_color": fill_color,
        "medium_i0": medium_i0,
        "sigma": sigma,
        "global_low": global_low,
        "global_high": global_high,
        "flatfield_low": flatfield_low,
        "flatfield_high": flatfield_high,
        "raw_stats": raw_stats,
        "global_stats": global_residual_stats,
        "flatfield_stats": flatfield_residual_stats,
        "background_stats": background_stats,
        **safe_stats,
    }


def process_dataset(
    input_root: Path,
    detections_path: Path,
    output_root: Path,
    input_variant: str,
    safe_erode_ratio: float,
    outlier_mad_multiplier: float,
    safe_minimum_fraction: float,
    flatfield_sigma_ratio: float,
    percentile_low: float,
    percentile_high: float,
    limit: int | None,
    debug_limit: int,
) -> None:
    input_root = input_root.resolve()
    detections_path = detections_path.resolve()
    output_root = output_root.resolve()

    detections = pd.read_csv(
        detections_path,
        dtype={"image_id": "string"},
    )
    detections = detections[
        detections["processing_status"].astype(str) == "success"
    ].copy()
    if limit is not None:
        detections = detections.head(limit).copy()

    image_column = {
        "normalized_raw": "normalized_raw_path",
        "normalized_masked": "normalized_masked_path",
    }[input_variant]
    required_columns = {
        "image_id",
        "background",
        "processing_status",
        image_column,
        "normalized_counting_mask_path",
    }
    missing = required_columns.difference(detections.columns)
    if missing:
        raise KeyError(
            f"Kolom yang dibutuhkan tidak ada di detection CSV: {sorted(missing)}"
        )

    records: list[dict[str, Any]] = []

    for row_number, (_, row) in enumerate(
        tqdm(
            detections.iterrows(),
            total=len(detections),
            desc="Normalisasi intensitas",
        )
    ):
        image_id = str(row["image_id"])
        background = str(row.get("background", "unknown"))
        image_path = resolve_relative(input_root, row[image_column])
        mask_path = resolve_relative(
            input_root,
            row["normalized_counting_mask_path"],
        )

        record: dict[str, Any] = {
            "image_id": image_id,
            "background": background,
            "geometry_status": row.get("detection_status", "unknown"),
            "input_variant": input_variant,
            "input_path": str(image_path),
            "counting_mask_path": str(mask_path),
        }

        try:
            image = read_image(image_path, cv2.IMREAD_COLOR)
            mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
            outputs = process_one(
                image_bgr=image,
                counting_mask=mask,
                safe_erode_ratio=safe_erode_ratio,
                outlier_mad_multiplier=outlier_mad_multiplier,
                safe_minimum_fraction=safe_minimum_fraction,
                flatfield_sigma_ratio=flatfield_sigma_ratio,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
            )

            raw_path = output_root / "raw_masked" / background / f"{image_id}.png"
            safe_mask_path = (
                output_root / "safe_medium_mask" / background / f"{image_id}.png"
            )
            global_path = (
                output_root / "global_log" / background / f"{image_id}.png"
            )
            flatfield_path = (
                output_root / "local_flatfield" / background / f"{image_id}.png"
            )

            save_image(raw_path, outputs["raw_masked"])
            save_image(safe_mask_path, outputs["safe_mask"])
            save_image(global_path, outputs["global_gray"])
            save_image(flatfield_path, outputs["flatfield_gray"])

            preview_path: Path | None = None
            if row_number < debug_limit:
                preview = create_preview(
                    raw_bgr=outputs["raw_masked"],
                    safe_mask=outputs["safe_mask"],
                    global_gray=outputs["global_gray"],
                    flatfield_gray=outputs["flatfield_gray"],
                )
                preview_path = (
                    output_root / "preview" / background / f"{image_id}.jpg"
                )
                save_image(preview_path, preview)

            raw_stats = outputs["raw_stats"]
            global_stats = outputs["global_stats"]
            flatfield_stats = outputs["flatfield_stats"]
            background_stats = outputs["background_stats"]

            record.update(
                {
                    "processing_status": "success",
                    "medium_i0": outputs["medium_i0"],
                    "safe_erode_radius": outputs["safe_erode_radius"],
                    "safe_mask_pixels": outputs["safe_mask_pixels"],
                    "safe_mask_fraction_of_plate": outputs[
                        "safe_mask_fraction_of_plate"
                    ],
                    "flatfield_sigma": outputs["sigma"],
                    "fill_color_b": int(outputs["fill_color"][0]),
                    "fill_color_g": int(outputs["fill_color"][1]),
                    "fill_color_r": int(outputs["fill_color"][2]),
                    "raw_medium_mean": raw_stats["mean"],
                    "raw_medium_median": raw_stats["median"],
                    "raw_medium_std": raw_stats["std"],
                    "raw_medium_cv": raw_stats["cv"],
                    "raw_medium_p01": raw_stats["p01"],
                    "raw_medium_p99": raw_stats["p99"],
                    "global_log_residual_mean": global_stats["mean"],
                    "global_log_residual_median": global_stats["median"],
                    "global_log_residual_std": global_stats["std"],
                    "global_log_residual_p01": global_stats["p01"],
                    "global_log_residual_p99": global_stats["p99"],
                    "flatfield_residual_mean": flatfield_stats["mean"],
                    "flatfield_residual_median": flatfield_stats["median"],
                    "flatfield_residual_std": flatfield_stats["std"],
                    "flatfield_residual_p01": flatfield_stats["p01"],
                    "flatfield_residual_p99": flatfield_stats["p99"],
                    "background_field_mean": background_stats["mean"],
                    "background_field_std": background_stats["std"],
                    "global_scale_low": outputs["global_low"],
                    "global_scale_high": outputs["global_high"],
                    "flatfield_scale_low": outputs["flatfield_low"],
                    "flatfield_scale_high": outputs["flatfield_high"],
                    "raw_output_path": raw_path.relative_to(output_root).as_posix(),
                    "safe_mask_output_path": safe_mask_path.relative_to(
                        output_root
                    ).as_posix(),
                    "global_log_output_path": global_path.relative_to(
                        output_root
                    ).as_posix(),
                    "local_flatfield_output_path": flatfield_path.relative_to(
                        output_root
                    ).as_posix(),
                    "preview_output_path": (
                        preview_path.relative_to(output_root).as_posix()
                        if preview_path is not None
                        else ""
                    ),
                    "error_message": "",
                }
            )
        except Exception as exc:
            record.update(
                {
                    "processing_status": "failed",
                    "error_message": str(exc),
                }
            )

        records.append(record)

    output_root.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(records)
    metrics_path = output_root / "intensity_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print(f"\nHasil metrik: {metrics_path}")
    print("\nProcessing status:")
    print(metrics["processing_status"].value_counts(dropna=False))

    successful = metrics[metrics["processing_status"] == "success"]
    if not successful.empty:
        summary_columns = [
            "raw_medium_cv",
            "global_log_residual_std",
            "flatfield_residual_std",
            "safe_mask_fraction_of_plate",
        ]
        print("\nRingkasan metrik:")
        print(successful[summary_columns].describe())

        if "background" in successful.columns:
            print("\nMedian metrik per subset:")
            print(
                successful.groupby("background")[summary_columns]
                .median(numeric_only=True)
                .sort_index()
            )


def main() -> None:
    project_root = Path(__file__).resolve().parent
    default_input_root = project_root / "processed_plate_strategy_b_circle"

    parser = argparse.ArgumentParser(
        description=(
            "Normalisasi intensitas AGAR setelah normalisasi geometri V3: "
            "raw masked, global-log, dan local flat-field."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=default_input_root,
        help="Folder output V3.",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=None,
        help=(
            "CSV deteksi V3. Default: "
            "<input-root>/plate_detection_strategy_b.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "preprocessed_intensity",
    )
    parser.add_argument(
        "--input-variant",
        choices=["normalized_raw", "normalized_masked"],
        default="normalized_raw",
        help="Input utama. normalized_raw disarankan untuk eksperimen.",
    )
    parser.add_argument(
        "--safe-erode-ratio",
        type=float,
        default=0.025,
        help=(
            "Radius erosi relatif terhadap sisi gambar untuk estimasi medium. "
            "0.025 pada 2048 px kira-kira radius 51 px."
        ),
    )
    parser.add_argument(
        "--outlier-mad",
        type=float,
        default=3.5,
        help="Batas outlier LAB dalam satuan robust MAD.",
    )
    parser.add_argument(
        "--safe-minimum-fraction",
        type=float,
        default=0.15,
        help="Fraksi minimum area plate untuk safe medium mask.",
    )
    parser.add_argument(
        "--flatfield-sigma-ratio",
        type=float,
        default=0.04,
        help=(
            "Sigma Gaussian relatif terhadap sisi gambar. "
            "0.04 pada 2048 px kira-kira sigma 82 px."
        ),
    )
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=100,
        help="Jumlah preview montage yang disimpan.",
    )

    args = parser.parse_args()
    detections_path = (
        args.detections
        if args.detections is not None
        else args.input_root / "plate_detection_strategy_b.csv"
    )

    if not 0.0 <= args.safe_erode_ratio < 0.25:
        raise ValueError("--safe-erode-ratio harus berada pada [0, 0.25).")
    if not 0.0 < args.safe_minimum_fraction < 1.0:
        raise ValueError("--safe-minimum-fraction harus berada antara 0 dan 1.")
    if not 0.0 < args.flatfield_sigma_ratio < 0.5:
        raise ValueError("--flatfield-sigma-ratio harus berada antara 0 dan 0.5.")
    if not 0 <= args.percentile_low < args.percentile_high <= 100:
        raise ValueError("Percentile harus memenuhi 0 <= low < high <= 100.")

    process_dataset(
        input_root=args.input_root,
        detections_path=detections_path,
        output_root=args.output,
        input_variant=args.input_variant,
        safe_erode_ratio=args.safe_erode_ratio,
        outlier_mad_multiplier=args.outlier_mad,
        safe_minimum_fraction=args.safe_minimum_fraction,
        flatfield_sigma_ratio=args.flatfield_sigma_ratio,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        limit=args.limit,
        debug_limit=args.debug_limit,
    )


if __name__ == "__main__":
    main()
