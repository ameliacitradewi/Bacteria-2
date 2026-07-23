# run: python detect_plate_mask.py --limit 100 --debug-limit 100
# result: processed_plate

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_image(image_path: Path) -> np.ndarray:
    """
    Membaca gambar dengan aman, termasuk jika path mengandung karakter khusus.
    """
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Gagal membaca gambar: {image_path}")

    return image


def save_image(
    output_path: Path,
    image: np.ndarray,
    jpeg_quality: int = 95,
) -> None:
    """
    Menyimpan JPG/PNG dengan aman.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
    elif suffix == ".png":
        success, encoded = cv2.imencode(
            ".png",
            image,
            [cv2.IMWRITE_PNG_COMPRESSION, 5],
        )
    else:
        raise ValueError(f"Format output tidak didukung: {suffix}")

    if not success:
        raise RuntimeError(f"Gagal mengencode gambar: {output_path}")

    encoded.tofile(str(output_path))


def resize_for_detection(
    image: np.ndarray,
    max_side: int = 1200,
) -> tuple[np.ndarray, float]:
    """
    Mengecilkan gambar hanya untuk proses deteksi.

    Hasil akhir tetap dikonversi kembali ke koordinat resolusi asli.
    """
    height, width = image.shape[:2]
    largest_side = max(height, width)

    if largest_side <= max_side:
        return image.copy(), 1.0

    scale = max_side / largest_side

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def prepare_detection_image(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Membuat citra grayscale dan edge map untuk deteksi cawan.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(
        enhanced,
        (9, 9),
        sigmaX=2.0,
    )

    median_intensity = float(np.median(blurred))

    lower = int(max(0, 0.60 * median_intensity))
    upper = int(min(255, 1.40 * median_intensity))

    if upper <= lower:
        lower = 40
        upper = 140

    edges = cv2.Canny(
        blurred,
        lower,
        upper,
    )

    return blurred, edges


def circle_edge_support(
    distance_to_edge: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    tolerance: int = 4,
    n_points: int = 720,
) -> float:
    """
    Mengukur seberapa besar keliling kandidat lingkaran berimpit dengan edge.
    """
    height, width = distance_to_edge.shape

    angles = np.linspace(
        0,
        2 * np.pi,
        n_points,
        endpoint=False,
    )

    best_distances = np.full(
        n_points,
        np.inf,
        dtype=np.float32,
    )

    for radius_offset in (-tolerance, 0, tolerance):
        current_radius = radius + radius_offset

        xs = np.round(
            cx + current_radius * np.cos(angles)
        ).astype(np.int32)

        ys = np.round(
            cy + current_radius * np.sin(angles)
        ).astype(np.int32)

        valid = (
            (xs >= 0)
            & (xs < width)
            & (ys >= 0)
            & (ys < height)
        )

        distances = np.full(
            n_points,
            np.inf,
            dtype=np.float32,
        )

        distances[valid] = distance_to_edge[
            ys[valid],
            xs[valid],
        ]

        best_distances = np.minimum(
            best_distances,
            distances,
        )

    valid_points = np.isfinite(best_distances)

    if valid_points.sum() == 0:
        return 0.0

    return float(
        np.mean(best_distances[valid_points] <= tolerance)
    )


def score_circle(
    cx: float,
    cy: float,
    radius: float,
    edge_support: float,
    image_width: int,
    image_height: int,
) -> float:
    """
    Menilai kandidat berdasarkan:
    - dukungan edge;
    - ukuran radius;
    - kedekatan ke pusat gambar;
    - apakah lingkaran keluar dari gambar.
    """
    min_dimension = min(image_width, image_height)

    center_x = image_width / 2
    center_y = image_height / 2

    center_distance = math.sqrt(
        (cx - center_x) ** 2
        + (cy - center_y) ** 2
    )

    normalized_center_distance = (
        center_distance / min_dimension
    )

    normalized_radius = (
        radius / (0.5 * min_dimension)
    )

    outside_left = max(0.0, radius - cx)
    outside_top = max(0.0, radius - cy)
    outside_right = max(
        0.0,
        cx + radius - image_width,
    )
    outside_bottom = max(
        0.0,
        cy + radius - image_height,
    )

    outside_fraction = (
        outside_left
        + outside_top
        + outside_right
        + outside_bottom
    ) / max(radius, 1.0)

    return float(
        edge_support
        + 0.15 * normalized_radius
        - 0.10 * normalized_center_distance
        - 0.30 * outside_fraction
    )


def deduplicate_circles(
    circles: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """
    Menghapus kandidat lingkaran yang hampir sama.
    """
    unique: list[tuple[float, float, float]] = []

    for cx, cy, radius in circles:
        duplicate = False

        for ux, uy, uradius in unique:
            center_distance = math.sqrt(
                (cx - ux) ** 2
                + (cy - uy) ** 2
            )

            if (
                center_distance < 15
                and abs(radius - uradius) < 15
            ):
                duplicate = True
                break

        if not duplicate:
            unique.append((cx, cy, radius))

    return unique


def hough_candidates(
    blurred: np.ndarray,
) -> list[tuple[float, float, float]]:
    """
    Menghasilkan beberapa kandidat lingkaran dari Hough Circle.
    """
    height, width = blurred.shape
    min_dimension = min(height, width)

    min_radius = int(0.25 * min_dimension)
    max_radius = int(0.50 * min_dimension)

    candidates: list[tuple[float, float, float]] = []

    # Beberapa param2 dipakai karena kontras tiap subset berbeda.
    for accumulator_threshold in (42, 34, 28, 22):
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(50, min_dimension // 4),
            param1=120,
            param2=accumulator_threshold,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is None:
            continue

        for circle in circles[0]:
            cx, cy, radius = map(float, circle)
            candidates.append((cx, cy, radius))

    return deduplicate_circles(candidates)


def contour_candidates(
    edges: np.ndarray,
) -> list[tuple[float, float, float]]:
    """
    Fallback jika Hough Circle tidak menemukan kandidat yang baik.
    """
    height, width = edges.shape
    image_area = height * width
    min_dimension = min(height, width)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11),
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[tuple[float, float, float]] = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)

        if contour_area < 0.10 * image_area:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        if not (
            0.22 * min_dimension
            <= radius
            <= 0.55 * min_dimension
        ):
            continue

        candidates.append(
            (float(cx), float(cy), float(radius))
        )

    return candidates


def detect_plate_circle(
    image: np.ndarray,
    detection_max_side: int = 1200,
) -> dict[str, float | str]:
    """
    Mendeteksi cawan dan mengembalikan koordinat pada resolusi asli.
    """
    small_image, scale = resize_for_detection(
        image,
        max_side=detection_max_side,
    )

    blurred, edges = prepare_detection_image(small_image)

    inverse_edges = 255 - edges

    distance_to_edge = cv2.distanceTransform(
        inverse_edges,
        cv2.DIST_L2,
        3,
    )

    candidates = hough_candidates(blurred)

    if not candidates:
        candidates = contour_candidates(edges)

    if not candidates:
        raise RuntimeError(
            "Tidak ditemukan kandidat lingkaran cawan."
        )

    small_height, small_width = blurred.shape

    evaluated_candidates: list[dict[str, float]] = []

    for cx, cy, radius in candidates:
        support = circle_edge_support(
            distance_to_edge,
            cx,
            cy,
            radius,
        )

        score = score_circle(
            cx=cx,
            cy=cy,
            radius=radius,
            edge_support=support,
            image_width=small_width,
            image_height=small_height,
        )

        evaluated_candidates.append(
            {
                "cx": cx,
                "cy": cy,
                "radius": radius,
                "edge_support": support,
                "score": score,
            }
        )

    best = max(
        evaluated_candidates,
        key=lambda item: item["score"],
    )

    confidence_status = (
        "ok"
        if best["edge_support"] >= 0.18
        else "low_confidence"
    )

    inverse_scale = 1.0 / scale

    return {
        "center_x": best["cx"] * inverse_scale,
        "center_y": best["cy"] * inverse_scale,
        "outer_radius": best["radius"] * inverse_scale,
        "edge_support": best["edge_support"],
        "detection_score": best["score"],
        "detection_status": confidence_status,
    }


def crop_square_with_padding(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    half_size: int,
) -> tuple[np.ndarray, int, int]:
    """
    Membuat crop persegi. Jika crop melewati batas gambar,
    bagian di luar gambar diisi hitam.
    """
    image_height, image_width = image.shape[:2]

    crop_x0 = int(round(center_x - half_size))
    crop_y0 = int(round(center_y - half_size))

    crop_size = 2 * half_size

    output = np.zeros(
        (crop_size, crop_size, 3),
        dtype=image.dtype,
    )

    source_x0 = max(0, crop_x0)
    source_y0 = max(0, crop_y0)
    source_x1 = min(image_width, crop_x0 + crop_size)
    source_y1 = min(image_height, crop_y0 + crop_size)

    destination_x0 = source_x0 - crop_x0
    destination_y0 = source_y0 - crop_y0

    destination_x1 = (
        destination_x0 + source_x1 - source_x0
    )
    destination_y1 = (
        destination_y0 + source_y1 - source_y0
    )

    if source_x1 > source_x0 and source_y1 > source_y0:
        output[
            destination_y0:destination_y1,
            destination_x0:destination_x1,
        ] = image[
            source_y0:source_y1,
            source_x0:source_x1,
        ]

    return output, crop_x0, crop_y0


def create_inner_plate_mask(
    crop_shape: tuple[int, int],
    center_x_in_crop: float,
    center_y_in_crop: float,
    inner_radius: float,
) -> np.ndarray:
    """
    Mask bernilai 255 pada medium agar bagian dalam.
    """
    height, width = crop_shape

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.circle(
        mask,
        (
            int(round(center_x_in_crop)),
            int(round(center_y_in_crop)),
        ),
        int(round(inner_radius)),
        255,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    return mask


def create_debug_overlay(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    outer_radius: float,
    inner_radius: float,
    status: str,
    max_side: int = 1400,
) -> np.ndarray:
    """
    Visualisasi outer circle dan inner plate mask.
    """
    debug, scale = resize_for_detection(
        image,
        max_side=max_side,
    )

    cx = int(round(center_x * scale))
    cy = int(round(center_y * scale))
    outer = int(round(outer_radius * scale))
    inner = int(round(inner_radius * scale))

    cv2.circle(
        debug,
        (cx, cy),
        outer,
        (0, 255, 0),
        thickness=4,
    )

    cv2.circle(
        debug,
        (cx, cy),
        inner,
        (0, 0, 255),
        thickness=4,
    )

    cv2.circle(
        debug,
        (cx, cy),
        8,
        (255, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        debug,
        f"status: {status}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    return debug


def resolve_image_path(
    dataset_root: Path,
    image_path_value: Any,
) -> Path:
    image_path = Path(str(image_path_value))

    if image_path.is_absolute():
        return image_path

    return dataset_root / image_path


def process_dataset(
    dataset_root: Path,
    manifest_path: Path,
    output_root: Path,
    inner_ratio: float,
    crop_ratio: float,
    detection_max_side: int,
    limit: int | None,
    debug_limit: int,
) -> None:
    dataset_root = dataset_root.resolve()
    manifest_path = manifest_path.resolve()
    output_root = output_root.resolve()

    manifest = pd.read_csv(manifest_path)

    if limit is not None:
        manifest = manifest.head(limit).copy()

    crop_root = output_root / "plate_crop_rgb"
    mask_root = output_root / "plate_mask"
    debug_root = output_root / "debug"

    records: list[dict[str, Any]] = []

    for row_number, (_, row) in enumerate(
        tqdm(
            manifest.iterrows(),
            total=len(manifest),
            desc="Deteksi cawan",
        )
    ):
        image_id = str(row["image_id"])
        background = str(row.get("background", "unknown"))

        image_path = resolve_image_path(
            dataset_root,
            row["image_path"],
        )

        record: dict[str, Any] = {
            "image_id": image_id,
            "sample_id": row.get("sample_id"),
            "background": background,
            "image_path": str(image_path),
        }

        try:
            image = read_image(image_path)

            image_height, image_width = image.shape[:2]

            detection = detect_plate_circle(
                image,
                detection_max_side=detection_max_side,
            )

            center_x = float(detection["center_x"])
            center_y = float(detection["center_y"])
            outer_radius = float(detection["outer_radius"])

            inner_radius = outer_radius * inner_ratio
            crop_half_size = int(
                math.ceil(outer_radius * crop_ratio)
            )

            crop, crop_x0, crop_y0 = crop_square_with_padding(
                image=image,
                center_x=center_x,
                center_y=center_y,
                half_size=crop_half_size,
            )

            center_x_in_crop = center_x - crop_x0
            center_y_in_crop = center_y - crop_y0

            mask = create_inner_plate_mask(
                crop_shape=crop.shape[:2],
                center_x_in_crop=center_x_in_crop,
                center_y_in_crop=center_y_in_crop,
                inner_radius=inner_radius,
            )

            crop_output = (
                crop_root
                / background
                / f"{image_id}.jpg"
            )

            mask_output = (
                mask_root
                / background
                / f"{image_id}.png"
            )

            save_image(crop_output, crop)
            save_image(mask_output, mask)

            if row_number < debug_limit:
                debug_image = create_debug_overlay(
                    image=image,
                    center_x=center_x,
                    center_y=center_y,
                    outer_radius=outer_radius,
                    inner_radius=inner_radius,
                    status=str(
                        detection["detection_status"]
                    ),
                )

                debug_output = (
                    debug_root
                    / background
                    / f"{image_id}.jpg"
                )

                save_image(debug_output, debug_image)

            record.update(
                {
                    "processing_status": "success",
                    "detection_status": detection[
                        "detection_status"
                    ],
                    "edge_support": detection[
                        "edge_support"
                    ],
                    "detection_score": detection[
                        "detection_score"
                    ],
                    "original_width": image_width,
                    "original_height": image_height,
                    "center_x": center_x,
                    "center_y": center_y,
                    "outer_radius": outer_radius,
                    "inner_radius": inner_radius,
                    "inner_ratio": inner_ratio,
                    "crop_ratio": crop_ratio,
                    "crop_x0": crop_x0,
                    "crop_y0": crop_y0,
                    "crop_size": crop.shape[0],
                    "center_x_in_crop": center_x_in_crop,
                    "center_y_in_crop": center_y_in_crop,
                    "crop_path": crop_output.relative_to(
                        output_root
                    ).as_posix(),
                    "mask_path": mask_output.relative_to(
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

    result = pd.DataFrame(records)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = output_root / "plate_detection.csv"

    result.to_csv(
        result_path,
        index=False,
    )

    print("\nSelesai.")
    print(f"Hasil deteksi: {result_path}")

    print("\nProcessing status:")
    print(
        result["processing_status"].value_counts(
            dropna=False
        )
    )

    print("\nDetection status:")
    print(
        result["detection_status"].value_counts(
            dropna=False
        )
    )

    if "edge_support" in result.columns:
        print("\nStatistik edge support:")
        print(result["edge_support"].describe())


def main() -> None:
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Deteksi cawan Petri dan pembuatan inner plate mask."
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
        default=project_root
        / "agar_metadata"
        / "image_manifest.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "processed_plate",
    )

    parser.add_argument(
        "--inner-ratio",
        type=float,
        default=0.90,
        help=(
            "Proporsi radius area agar terhadap radius luar cawan."
        ),
    )

    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=1.03,
        help=(
            "Ukuran setengah crop relatif terhadap radius luar."
        ),
    )

    parser.add_argument(
        "--detection-max-side",
        type=int,
        default=1200,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batasi jumlah gambar untuk pengujian.",
    )

    parser.add_argument(
        "--debug-limit",
        type=int,
        default=100,
        help="Jumlah debug overlay yang disimpan.",
    )

    args = parser.parse_args()

    if not 0.5 < args.inner_ratio < 1.0:
        raise ValueError(
            "--inner-ratio harus berada antara 0.5 dan 1.0"
        )

    process_dataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        output_root=args.output,
        inner_ratio=args.inner_ratio,
        crop_ratio=args.crop_ratio,
        detection_max_side=args.detection_max_side,
        limit=args.limit,
        debug_limit=args.debug_limit,
    )


if __name__ == "__main__":
    main()
    
    
    
import pandas as pd

detections = pd.read_csv(
    "processed_plate/plate_detection.csv"
)

low_confidence = detections[
    detections["detection_status"] != "ok"
]

print(
    low_confidence[
        [
            "image_id",
            "background",
            "edge_support",
            "detection_status",
            "error_message",
        ]
    ]
)

low_confidence.to_csv(
    "processed_plate/low_confidence.csv",
    index=False,
)