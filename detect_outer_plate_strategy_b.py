from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Baca gambar dengan aman, termasuk path non-ASCII."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise RuntimeError(f"Gagal membaca gambar: {path}")
    return image


def save_image(path: Path, image: np.ndarray, jpeg_quality: int = 95) -> None:
    """Simpan JPG/PNG dengan aman."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
    elif suffix == ".png":
        ok, encoded = cv2.imencode(
            ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 5]
        )
    else:
        raise ValueError(f"Format tidak didukung: {suffix}")

    if not ok:
        raise RuntimeError(f"Gagal meng-encode: {path}")

    encoded.tofile(str(path))


def resolve_image_path(dataset_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else dataset_root / path


def resize_for_detection(
    image: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, float]:
    """Resize hanya untuk deteksi. `scale` = ukuran kecil / ukuran asli."""
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))

    if scale == 1.0:
        return image.copy(), 1.0

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def robust_normalize(channel: np.ndarray) -> np.ndarray:
    channel = channel.astype(np.float32)
    low, high = np.percentile(channel, [1, 99])

    if high <= low:
        return np.zeros_like(channel, dtype=np.float32)

    normalized = (channel - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def build_gradient_map(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Membuat grayscale dan gradient gabungan grayscale + LAB.
    LAB membantu saat plastik transparan lemah di grayscale tetapi berbeda warna.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    channels = [gray, lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]]
    gradients: list[np.ndarray] = []

    for channel in channels:
        channel = cv2.GaussianBlur(channel, (5, 5), 1.0)
        gx = cv2.Scharr(channel, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(channel, cv2.CV_32F, 0, 1)
        gradients.append(robust_normalize(cv2.magnitude(gx, gy)))

    combined = np.max(np.stack(gradients, axis=0), axis=0)
    gradient_u8 = np.round(combined * 255).astype(np.uint8)
    return gray, gradient_u8


def deduplicate_circles(
    circles: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    unique: list[tuple[float, float, float]] = []

    for cx, cy, radius in sorted(circles, key=lambda item: item[2], reverse=True):
        is_duplicate = any(
            math.hypot(cx - ux, cy - uy) < 12 and abs(radius - uradius) < 12
            for ux, uy, uradius in unique
        )
        if not is_duplicate:
            unique.append((cx, cy, radius))

    return unique


def circle_support(
    edge_distance: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    tolerance: int = 4,
    n_points: int = 720,
) -> float:
    height, width = edge_distance.shape
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    xs = np.round(cx + radius * np.cos(angles)).astype(np.int32)
    ys = np.round(cy + radius * np.sin(angles)).astype(np.int32)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if valid.sum() == 0:
        return 0.0

    return float(np.mean(edge_distance[ys[valid], xs[valid]] <= tolerance))


def hough_candidates(gray: np.ndarray) -> list[tuple[float, float, float]]:
    """Hough hanya dipakai sebagai perkiraan awal pusat dan skala."""
    height, width = gray.shape
    min_dimension = min(height, width)

    enhanced = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8),
    ).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (9, 9), 2.0)

    candidates: list[tuple[float, float, float]] = []

    for param2 in (48, 40, 34, 28, 22):
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(50, int(0.20 * min_dimension)),
            param1=120,
            param2=param2,
            minRadius=int(0.15 * min_dimension),
            maxRadius=int(0.62 * min_dimension),
        )

        if circles is None:
            continue

        for cx, cy, radius in circles[0]:
            candidates.append((float(cx), float(cy), float(radius)))

    return deduplicate_circles(candidates)


def contour_candidates(gradient_u8: np.ndarray) -> list[tuple[float, float, float]]:
    """Fallback apabila Hough tidak menemukan kandidat."""
    height, width = gradient_u8.shape
    min_dimension = min(height, width)
    image_area = height * width

    _, binary = cv2.threshold(
        gradient_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[tuple[float, float, float]] = []

    for contour in contours:
        if cv2.contourArea(contour) < 0.06 * image_area:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if 0.15 * min_dimension <= radius <= 0.68 * min_dimension:
            candidates.append((float(cx), float(cy), float(radius)))

    return deduplicate_circles(candidates)


def select_coarse_circle(
    gray: np.ndarray,
    gradient_u8: np.ndarray,
) -> dict[str, float]:
    edges = cv2.Canny(gray, 40, 120)
    edge_distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)

    candidates = hough_candidates(gray)
    if not candidates:
        candidates = contour_candidates(gradient_u8)
    if not candidates:
        raise RuntimeError("Tidak ditemukan kandidat awal cawan.")

    height, width = gray.shape
    min_dimension = min(height, width)
    image_center = (width / 2.0, height / 2.0)

    evaluated: list[dict[str, float]] = []

    for cx, cy, radius in candidates:
        support = circle_support(edge_distance, cx, cy, radius)
        radius_ratio = radius / min_dimension
        center_distance = math.hypot(
            cx - image_center[0],
            cy - image_center[1],
        ) / min_dimension

        size_score = min(radius_ratio / 0.50, 1.0)
        score = 0.60 * support + 0.30 * size_score - 0.10 * center_distance

        evaluated.append(
            {
                "center_x": cx,
                "center_y": cy,
                "radius": radius,
                "support": support,
                "score": score,
            }
        )

    best_support = max(item["support"] for item in evaluated)
    eligible = [
        item
        for item in evaluated
        if item["support"] >= max(0.05, 0.60 * best_support)
    ]

    # Prioritaskan kandidat besar agar tidak berhenti pada ring bagian dalam.
    return max(eligible, key=lambda item: (item["radius"], item["score"]))


def smooth_profile(profile: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(
        profile.reshape(1, -1).astype(np.float32),
        (kernel_size, 1),
        sigmaX=2.0,
    ).ravel()


def circular_median(values: np.ndarray, window: int = 21) -> np.ndarray:
    if window % 2 == 0:
        window += 1

    half = window // 2
    padded = np.concatenate([values[-half:], values, values[:half]])
    output = np.empty_like(values)

    for index in range(len(values)):
        output[index] = np.median(padded[index : index + window])

    return output


def ellipse_residuals(
    points: np.ndarray,
    ellipse: tuple[tuple[float, float], tuple[float, float], float],
) -> np.ndarray:
    (cx, cy), (diameter_a, diameter_b), angle_deg = ellipse
    a = max(diameter_a / 2.0, 1.0)
    b = max(diameter_b / 2.0, 1.0)

    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    shifted = points - np.array([cx, cy], dtype=np.float32)
    x_rot = shifted[:, 0] * cos_t + shifted[:, 1] * sin_t
    y_rot = -shifted[:, 0] * sin_t + shifted[:, 1] * cos_t

    normalized_radius = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)
    return np.abs(normalized_radius - 1.0) * ((a + b) / 2.0)


def robust_fit_ellipse(
    points: np.ndarray,
) -> tuple[
    tuple[tuple[float, float], tuple[float, float], float],
    np.ndarray,
    np.ndarray,
]:
    if len(points) < 20:
        raise RuntimeError("Titik boundary terlalu sedikit untuk fit ellipse.")

    current = points.astype(np.float32)

    for _ in range(4):
        ellipse = cv2.fitEllipseAMS(current.reshape(-1, 1, 2))
        residuals = ellipse_residuals(current, ellipse)

        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        threshold = max(4.0, median + 3.5 * max(mad, 1.0))
        keep = residuals <= threshold

        if keep.all():
            break

        current = current[keep]
        if len(current) < 20:
            raise RuntimeError("Titik valid terlalu sedikit setelah filtering.")

    ellipse = cv2.fitEllipseAMS(current.reshape(-1, 1, 2))
    residuals = ellipse_residuals(current, ellipse)
    return ellipse, current, residuals


def refine_outer_boundary(
    gradient_u8: np.ndarray,
    coarse: dict[str, float],
    n_angles: int,
    radial_min_ratio: float,
    radial_max_ratio: float,
) -> dict[str, Any]:
    """
    Pada setiap arah, cari peak gradient valid yang paling luar.
    Kontinuitas radius dan robust ellipse fitting membuang edge objek luar.
    """
    gradient = gradient_u8.astype(np.float32) / 255.0
    height, width = gradient.shape

    coarse_center_x = coarse["center_x"]
    coarse_center_y = coarse["center_y"]
    coarse_radius = coarse["radius"]

    min_radius = max(5, int(round(coarse_radius * radial_min_ratio)))
    max_radius = int(round(coarse_radius * radial_max_ratio))
    radii = np.arange(min_radius, max_radius + 1, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    selected_radii = np.full(n_angles, np.nan, dtype=np.float32)

    for angle_index, angle in enumerate(angles):
        cos_a = math.cos(float(angle))
        sin_a = math.sin(float(angle))

        xs = np.round(coarse_center_x + radii * cos_a).astype(np.int32)
        ys = np.round(coarse_center_y + radii * sin_a).astype(np.int32)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)

        if valid.sum() < 12:
            continue

        valid_radii = radii[valid]
        profile = smooth_profile(gradient[ys[valid], xs[valid]], 9)

        median = float(np.median(profile))
        mad = float(np.median(np.abs(profile - median)))
        threshold = max(
            float(np.percentile(profile, 80)),
            median + 1.5 * max(mad, 0.01),
            0.08,
        )

        local_maxima = (
            (profile[1:-1] > profile[:-2])
            & (profile[1:-1] >= profile[2:])
            & (profile[1:-1] >= threshold)
        )
        peaks = np.where(local_maxima)[0] + 1

        if len(peaks) > 0:
            selected_radii[angle_index] = valid_radii[int(peaks[-1])]

    observed = np.isfinite(selected_radii)
    if observed.sum() < max(60, int(0.25 * n_angles)):
        raise RuntimeError("Boundary radial ditemukan pada terlalu sedikit sudut.")

    # Interpolasi circular untuk mendapatkan baseline radius lokal.
    indices = np.arange(n_angles)
    valid_indices = indices[observed]
    valid_values = selected_radii[observed]
    extended_indices = np.concatenate(
        [valid_indices - n_angles, valid_indices, valid_indices + n_angles]
    )
    extended_values = np.concatenate([valid_values, valid_values, valid_values])
    interpolated = np.interp(indices, extended_indices, extended_values)

    local_median = circular_median(interpolated, window=21)
    deviation = np.abs(interpolated - local_median)
    allowed_deviation = max(6.0, 0.025 * coarse_radius)
    accepted = observed & (deviation <= allowed_deviation)

    accepted_angles = angles[accepted]
    accepted_radii = selected_radii[accepted]
    if len(accepted_radii) < 40:
        raise RuntimeError("Titik boundary tersisa terlalu sedikit.")

    points = np.column_stack(
        [
            coarse_center_x + accepted_radii * np.cos(accepted_angles),
            coarse_center_y + accepted_radii * np.sin(accepted_angles),
        ]
    ).astype(np.float32)

    ellipse, inlier_points, residuals = robust_fit_ellipse(points)
    (cx, cy), (axis_1, axis_2), angle = ellipse

    major = max(axis_1, axis_2)
    minor = min(axis_1, axis_2)
    axis_ratio = minor / max(major, 1.0)

    if axis_ratio < 0.65:
        raise RuntimeError(f"Ellipse terlalu pipih: axis_ratio={axis_ratio:.3f}")

    return {
        "ellipse_center_x": float(cx),
        "ellipse_center_y": float(cy),
        "ellipse_axis_1": float(axis_1),
        "ellipse_axis_2": float(axis_2),
        "ellipse_angle": float(angle),
        "boundary_coverage": float(observed.mean()),
        "accepted_coverage": float(accepted.mean()),
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
        "axis_ratio": float(axis_ratio),
        "boundary_points": inlier_points,
    }


def map_refined_to_original(
    refined_small: dict[str, Any],
    scale: float,
) -> dict[str, Any]:
    inverse = 1.0 / scale
    result = dict(refined_small)

    for key in (
        "ellipse_center_x",
        "ellipse_center_y",
        "ellipse_axis_1",
        "ellipse_axis_2",
        "mean_residual",
        "median_residual",
    ):
        result[key] = float(result[key]) * inverse

    result["boundary_points"] = refined_small["boundary_points"] * inverse
    return result


def detection_status(
    refined_small: dict[str, Any],
    small_shape: tuple[int, int],
) -> str:
    height, width = small_shape
    major_axis = max(
        refined_small["ellipse_axis_1"],
        refined_small["ellipse_axis_2"],
    )

    center_ok = (
        -0.10 * width <= refined_small["ellipse_center_x"] <= 1.10 * width
        and -0.10 * height <= refined_small["ellipse_center_y"] <= 1.10 * height
    )
    size_ok = 0.25 * min(height, width) <= major_axis <= 1.45 * max(height, width)
    axis_ok = refined_small["axis_ratio"] >= 0.75
    coverage_ok = refined_small["accepted_coverage"] >= 0.40
    residual_ratio = refined_small["mean_residual"] / max(major_axis, 1.0)
    residual_ok = residual_ratio <= 0.020

    return (
        "ok"
        if all([center_ok, size_ok, axis_ok, coverage_ok, residual_ok])
        else "low_confidence"
    )


def scale_ellipse(
    center: tuple[float, float],
    axes: tuple[float, float],
    angle: float,
    factor: float,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    return center, (axes[0] * factor, axes[1] * factor), angle


def ellipse_mask(
    shape: tuple[int, int],
    ellipse: tuple[tuple[float, float], tuple[float, float], float],
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    (cx, cy), (axis_1, axis_2), angle = ellipse

    cv2.ellipse(
        mask,
        (int(round(cx)), int(round(cy))),
        (
            max(1, int(round(axis_1 / 2.0))),
            max(1, int(round(axis_2 / 2.0))),
        ),
        angle,
        0,
        360,
        255,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    return mask


def image_corner_color(image: np.ndarray, patch: int = 32) -> tuple[int, int, int]:
    height, width = image.shape[:2]
    patch = min(patch, height, width)
    corners = np.concatenate(
        [
            image[:patch, :patch].reshape(-1, 3),
            image[:patch, width - patch :].reshape(-1, 3),
            image[height - patch :, :patch].reshape(-1, 3),
            image[height - patch :, width - patch :].reshape(-1, 3),
        ],
        axis=0,
    )
    median = np.median(corners, axis=0)
    return tuple(int(round(value)) for value in median)


def crop_square_with_padding(
    image: np.ndarray,
    crop_x0: int,
    crop_y0: int,
    crop_size: int,
    fill_value: tuple[int, int, int] | int,
) -> np.ndarray:
    if image.ndim == 2:
        fill = fill_value if isinstance(fill_value, int) else int(fill_value[0])
        output = np.full((crop_size, crop_size), fill, dtype=image.dtype)
    else:
        output = np.full(
            (crop_size, crop_size, image.shape[2]),
            fill_value,
            dtype=image.dtype,
        )

    height, width = image.shape[:2]
    source_x0 = max(0, crop_x0)
    source_y0 = max(0, crop_y0)
    source_x1 = min(width, crop_x0 + crop_size)
    source_y1 = min(height, crop_y0 + crop_size)

    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return output

    destination_x0 = source_x0 - crop_x0
    destination_y0 = source_y0 - crop_y0
    destination_x1 = destination_x0 + source_x1 - source_x0
    destination_y1 = destination_y0 + source_y1 - source_y0

    output[destination_y0:destination_y1, destination_x0:destination_x1] = image[
        source_y0:source_y1,
        source_x0:source_x1,
    ]
    return output


def robust_inside_color(image: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    pixels = image[mask > 0]
    if len(pixels) == 0:
        return (0, 0, 0)

    median = np.median(pixels, axis=0)
    return tuple(int(round(value)) for value in median)


def create_crop_outputs(
    image: np.ndarray,
    physical_ellipse: tuple[tuple[float, float], tuple[float, float], float],
    counting_scale: float,
    crop_padding_ratio: float,
    target_size: int,
) -> dict[str, Any]:
    center, axes, angle = physical_ellipse
    crop_size = max(32, int(math.ceil(max(axes) * crop_padding_ratio)))
    crop_x0 = int(round(center[0] - crop_size / 2.0))
    crop_y0 = int(round(center[1] - crop_size / 2.0))

    initial_fill = image_corner_color(image)
    crop_raw = crop_square_with_padding(
        image,
        crop_x0,
        crop_y0,
        crop_size,
        initial_fill,
    )

    ellipse_in_crop = (
        (center[0] - crop_x0, center[1] - crop_y0),
        axes,
        angle,
    )
    counting_ellipse_in_crop = scale_ellipse(
        ellipse_in_crop[0],
        ellipse_in_crop[1],
        ellipse_in_crop[2],
        counting_scale,
    )

    physical_mask = ellipse_mask(crop_raw.shape[:2], ellipse_in_crop)
    counting_mask = ellipse_mask(crop_raw.shape[:2], counting_ellipse_in_crop)

    fill_color = robust_inside_color(crop_raw, physical_mask)
    crop_masked = crop_raw.copy()
    crop_masked[physical_mask == 0] = fill_color

    image_interpolation = (
        cv2.INTER_AREA if crop_size > target_size else cv2.INTER_CUBIC
    )
    normalized_raw = cv2.resize(
        crop_raw,
        (target_size, target_size),
        interpolation=image_interpolation,
    )
    normalized_masked = cv2.resize(
        crop_masked,
        (target_size, target_size),
        interpolation=image_interpolation,
    )
    normalized_physical_mask = cv2.resize(
        physical_mask,
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST,
    )
    normalized_counting_mask = cv2.resize(
        counting_mask,
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST,
    )

    return {
        "crop_raw": crop_raw,
        "normalized_raw": normalized_raw,
        "normalized_masked": normalized_masked,
        "normalized_physical_mask": normalized_physical_mask,
        "normalized_counting_mask": normalized_counting_mask,
        "crop_x0": crop_x0,
        "crop_y0": crop_y0,
        "crop_size": crop_size,
        "normalization_scale": target_size / crop_size,
        "fill_color_b": fill_color[0],
        "fill_color_g": fill_color[1],
        "fill_color_r": fill_color[2],
    }


def create_debug_overlay(
    image: np.ndarray,
    detection_scale: float,
    coarse_small: dict[str, float],
    refined_original: dict[str, Any],
    physical_expansion: float,
    counting_scale: float,
    status: str,
    max_side: int = 1400,
) -> np.ndarray:
    debug, debug_scale = resize_for_detection(image, max_side)

    # Coarse circle: cyan.
    coarse_to_original = 1.0 / detection_scale
    coarse_center_original = (
        coarse_small["center_x"] * coarse_to_original,
        coarse_small["center_y"] * coarse_to_original,
    )
    coarse_radius_original = coarse_small["radius"] * coarse_to_original

    cv2.circle(
        debug,
        (
            int(round(coarse_center_original[0] * debug_scale)),
            int(round(coarse_center_original[1] * debug_scale)),
        ),
        int(round(coarse_radius_original * debug_scale)),
        (255, 255, 0),
        2,
    )

    center = (
        refined_original["ellipse_center_x"] * debug_scale,
        refined_original["ellipse_center_y"] * debug_scale,
    )
    axes = (
        refined_original["ellipse_axis_1"] * physical_expansion * debug_scale,
        refined_original["ellipse_axis_2"] * physical_expansion * debug_scale,
    )
    angle = refined_original["ellipse_angle"]

    # Final physical boundary: green.
    cv2.ellipse(
        debug,
        (int(round(center[0])), int(round(center[1]))),
        (int(round(axes[0] / 2.0)), int(round(axes[1] / 2.0))),
        angle,
        0,
        360,
        (0, 255, 0),
        4,
    )

    # Hanya digambar bila counting mask berbeda dari physical mask.
    if abs(counting_scale - 1.0) > 1e-6:
        cv2.ellipse(
            debug,
            (int(round(center[0])), int(round(center[1]))),
            (
                int(round(axes[0] * counting_scale / 2.0)),
                int(round(axes[1] * counting_scale / 2.0)),
            ),
            angle,
            0,
            360,
            (0, 0, 255),
            3,
        )

    points = refined_original["boundary_points"]
    step = max(1, len(points) // 120)
    for x, y in points[::step]:
        cv2.circle(
            debug,
            (int(round(x * debug_scale)), int(round(y * debug_scale))),
            2,
            (255, 0, 255),
            -1,
        )

    cv2.circle(
        debug,
        (int(round(center[0])), int(round(center[1]))),
        6,
        (255, 0, 0),
        -1,
    )

    major = max(
        refined_original["ellipse_axis_1"],
        refined_original["ellipse_axis_2"],
    )
    residual_ratio = refined_original["mean_residual"] / max(major, 1.0)
    label = (
        f"{status} | coverage={refined_original['accepted_coverage']:.2f} "
        f"| residual={residual_ratio:.4f} | axis={refined_original['axis_ratio']:.3f}"
    )
    cv2.putText(
        debug,
        label,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return debug


def transform_annotations(
    objects_path: Path,
    detections: pd.DataFrame,
    output_root: Path,
) -> None:
    objects = pd.read_csv(objects_path)
    if objects.empty:
        return

    columns = [
        "image_id",
        "processing_status",
        "crop_x0",
        "crop_y0",
        "normalization_scale",
        "normalized_counting_mask_path",
    ]
    merged = objects.merge(detections[columns], on="image_id", how="left")
    valid = merged["processing_status"] == "success"

    merged["x_normalized"] = np.nan
    merged["y_normalized"] = np.nan
    merged["width_normalized"] = np.nan
    merged["height_normalized"] = np.nan
    merged["center_inside_counting_mask"] = False

    merged.loc[valid, "x_normalized"] = (
        merged.loc[valid, "x"] - merged.loc[valid, "crop_x0"]
    ) * merged.loc[valid, "normalization_scale"]
    merged.loc[valid, "y_normalized"] = (
        merged.loc[valid, "y"] - merged.loc[valid, "crop_y0"]
    ) * merged.loc[valid, "normalization_scale"]
    merged.loc[valid, "width_normalized"] = (
        merged.loc[valid, "width"] * merged.loc[valid, "normalization_scale"]
    )
    merged.loc[valid, "height_normalized"] = (
        merged.loc[valid, "height"] * merged.loc[valid, "normalization_scale"]
    )

    for _, detection in detections[
        detections["processing_status"] == "success"
    ].iterrows():
        image_id = detection["image_id"]
        group_indices = merged.index[(merged["image_id"] == image_id) & valid]
        if len(group_indices) == 0:
            continue

        mask_path = output_root / str(detection["normalized_counting_mask_path"])
        mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)

        for index in group_indices:
            row = merged.loc[index]
            cx = row["x_normalized"] + row["width_normalized"] / 2.0
            cy = row["y_normalized"] + row["height_normalized"] / 2.0
            ix = int(round(cx))
            iy = int(round(cy))

            inside = (
                0 <= ix < mask.shape[1]
                and 0 <= iy < mask.shape[0]
                and mask[iy, ix] > 0
            )
            merged.at[index, "center_inside_counting_mask"] = bool(inside)

    output_path = output_root / "object_annotations_normalized.csv"
    merged.to_csv(output_path, index=False)
    print(f"Anotasi hasil transformasi: {output_path}")


def process_dataset(
    dataset_root: Path,
    manifest_path: Path,
    objects_path: Path | None,
    output_root: Path,
    detection_max_side: int,
    target_size: int,
    physical_expansion: float,
    counting_scale: float,
    crop_padding_ratio: float,
    radial_min_ratio: float,
    radial_max_ratio: float,
    n_angles: int,
    limit: int | None,
    debug_limit: int,
) -> None:
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    manifest = pd.read_csv(manifest_path)

    if limit is not None:
        manifest = manifest.head(limit).copy()

    records: list[dict[str, Any]] = []

    for row_number, (_, row) in enumerate(
        tqdm(
            manifest.iterrows(),
            total=len(manifest),
            desc="Deteksi outer physical plate",
        )
    ):
        image_id = str(row["image_id"])
        background = str(row.get("background", "unknown"))
        image_path = resolve_image_path(dataset_root, row["image_path"])

        record: dict[str, Any] = {
            "image_id": image_id,
            "sample_id": row.get("sample_id"),
            "background": background,
            "image_path": str(image_path),
        }

        try:
            image = read_image(image_path)
            small, detection_scale = resize_for_detection(
                image,
                detection_max_side,
            )
            gray_small, gradient_small = build_gradient_map(small)
            coarse_small = select_coarse_circle(gray_small, gradient_small)
            refined_small = refine_outer_boundary(
                gradient_u8=gradient_small,
                coarse=coarse_small,
                n_angles=n_angles,
                radial_min_ratio=radial_min_ratio,
                radial_max_ratio=radial_max_ratio,
            )
            status = detection_status(refined_small, small.shape[:2])
            refined_original = map_refined_to_original(
                refined_small,
                detection_scale,
            )

            refined_center = (
                refined_original["ellipse_center_x"],
                refined_original["ellipse_center_y"],
            )
            refined_axes = (
                refined_original["ellipse_axis_1"],
                refined_original["ellipse_axis_2"],
            )
            physical_ellipse = scale_ellipse(
                refined_center,
                refined_axes,
                refined_original["ellipse_angle"],
                physical_expansion,
            )

            outputs = create_crop_outputs(
                image=image,
                physical_ellipse=physical_ellipse,
                counting_scale=counting_scale,
                crop_padding_ratio=crop_padding_ratio,
                target_size=target_size,
            )

            raw_crop_path = (
                output_root / "plate_crop_raw" / background / f"{image_id}.jpg"
            )
            normalized_raw_path = (
                output_root
                / "plate_crop_normalized"
                / background
                / f"{image_id}.jpg"
            )
            normalized_masked_path = (
                output_root
                / "plate_crop_masked"
                / background
                / f"{image_id}.jpg"
            )
            physical_mask_path = (
                output_root
                / "physical_plate_mask"
                / background
                / f"{image_id}.png"
            )
            counting_mask_path = (
                output_root / "counting_mask" / background / f"{image_id}.png"
            )

            save_image(raw_crop_path, outputs["crop_raw"])
            save_image(normalized_raw_path, outputs["normalized_raw"])
            save_image(normalized_masked_path, outputs["normalized_masked"])
            save_image(physical_mask_path, outputs["normalized_physical_mask"])
            save_image(counting_mask_path, outputs["normalized_counting_mask"])

            if row_number < debug_limit or status != "ok":
                debug = create_debug_overlay(
                    image=image,
                    detection_scale=detection_scale,
                    coarse_small=coarse_small,
                    refined_original=refined_original,
                    physical_expansion=physical_expansion,
                    counting_scale=counting_scale,
                    status=status,
                )
                debug_path = output_root / "debug" / background / f"{image_id}.jpg"
                save_image(debug_path, debug)

            major_original = max(
                refined_original["ellipse_axis_1"],
                refined_original["ellipse_axis_2"],
            )

            record.update(
                {
                    "processing_status": "success",
                    "detection_status": status,
                    "detection_scale": detection_scale,
                    "coarse_center_x_small": coarse_small["center_x"],
                    "coarse_center_y_small": coarse_small["center_y"],
                    "coarse_radius_small": coarse_small["radius"],
                    "coarse_support": coarse_small["support"],
                    "ellipse_center_x": refined_original["ellipse_center_x"],
                    "ellipse_center_y": refined_original["ellipse_center_y"],
                    "ellipse_axis_1": refined_original["ellipse_axis_1"],
                    "ellipse_axis_2": refined_original["ellipse_axis_2"],
                    "ellipse_angle": refined_original["ellipse_angle"],
                    "physical_expansion": physical_expansion,
                    "counting_scale": counting_scale,
                    "boundary_coverage": refined_original["boundary_coverage"],
                    "accepted_coverage": refined_original["accepted_coverage"],
                    "mean_residual": refined_original["mean_residual"],
                    "mean_residual_ratio": refined_original["mean_residual"]
                    / max(major_original, 1.0),
                    "axis_ratio": refined_original["axis_ratio"],
                    "crop_x0": outputs["crop_x0"],
                    "crop_y0": outputs["crop_y0"],
                    "crop_size": outputs["crop_size"],
                    "target_size": target_size,
                    "normalization_scale": outputs["normalization_scale"],
                    "fill_color_b": outputs["fill_color_b"],
                    "fill_color_g": outputs["fill_color_g"],
                    "fill_color_r": outputs["fill_color_r"],
                    "raw_crop_path": raw_crop_path.relative_to(output_root).as_posix(),
                    "normalized_raw_path": normalized_raw_path.relative_to(
                        output_root
                    ).as_posix(),
                    "normalized_masked_path": normalized_masked_path.relative_to(
                        output_root
                    ).as_posix(),
                    "normalized_physical_mask_path": physical_mask_path.relative_to(
                        output_root
                    ).as_posix(),
                    "normalized_counting_mask_path": counting_mask_path.relative_to(
                        output_root
                    ).as_posix(),
                    "error_message": "",
                }
            )

        except Exception as exc:
            record.update(
                {
                    "processing_status": "failed",
                    "detection_status": "failed",
                    "error_message": str(exc),
                }
            )

        records.append(record)

    output_root.mkdir(parents=True, exist_ok=True)
    detections = pd.DataFrame(records)
    detection_path = output_root / "plate_detection_strategy_b.csv"
    detections.to_csv(detection_path, index=False)

    if objects_path is not None and objects_path.exists():
        transform_annotations(objects_path, detections, output_root)

    print(f"Hasil deteksi: {detection_path}")
    print("\nProcessing status:")
    print(detections["processing_status"].value_counts(dropna=False))
    print("\nDetection status:")
    print(detections["detection_status"].value_counts(dropna=False))

    successful = detections[detections["processing_status"] == "success"]
    if not successful.empty:
        print("\nStatistik kualitas:")
        print(
            successful[
                ["accepted_coverage", "mean_residual_ratio", "axis_ratio"]
            ].describe()
        )


def main() -> None:
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Strategi B: pakai seluruh batas fisik cawan untuk counting, "
            "dengan deteksi ukuran cawan yang dinamis."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "AGAR_dataset" / "dataset",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "agar_metadata" / "image_manifest.csv",
    )
    parser.add_argument(
        "--objects",
        type=Path,
        default=project_root / "agar_metadata" / "object_annotations.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "processed_plate_strategy_b",
    )
    parser.add_argument("--detection-max-side", type=int, default=1400)
    parser.add_argument("--target-size", type=int, default=2048)
    parser.add_argument(
        "--physical-expansion",
        type=float,
        default=1.01,
        help="Ekspansi kecil agar ellipse mencapai tepi plastik terluar.",
    )
    parser.add_argument(
        "--counting-scale",
        type=float,
        default=1.0,
        help="1.0 berarti seluruh physical plate digunakan untuk counting.",
    )
    parser.add_argument("--crop-padding-ratio", type=float, default=1.04)
    parser.add_argument("--radial-min-ratio", type=float, default=0.82)
    parser.add_argument("--radial-max-ratio", type=float, default=1.25)
    parser.add_argument("--n-angles", type=int, default=720)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug-limit", type=int, default=200)
    args = parser.parse_args()

    if args.target_size <= 0:
        raise ValueError("--target-size harus lebih besar dari 0.")
    if not 0.90 <= args.counting_scale <= 1.05:
        raise ValueError("--counting-scale harus berada pada 0.90–1.05.")
    if args.radial_min_ratio >= args.radial_max_ratio:
        raise ValueError("radial-min-ratio harus lebih kecil dari radial-max-ratio.")

    process_dataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        objects_path=args.objects,
        output_root=args.output,
        detection_max_side=args.detection_max_side,
        target_size=args.target_size,
        physical_expansion=args.physical_expansion,
        counting_scale=args.counting_scale,
        crop_padding_ratio=args.crop_padding_ratio,
        radial_min_ratio=args.radial_min_ratio,
        radial_max_ratio=args.radial_max_ratio,
        n_angles=args.n_angles,
        limit=args.limit,
        debug_limit=args.debug_limit,
    )


if __name__ == "__main__":
    main()
