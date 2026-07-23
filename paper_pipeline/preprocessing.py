from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


EPSILON = 1e-6


@dataclass(frozen=True)
class PreprocessResult:
    denoised: np.ndarray
    approximate_plate_mask: np.ndarray
    approximate_medium_mask: np.ndarray
    corrected_float: np.ndarray
    corrected_u8: np.ndarray
    medium_intensity: float


def adaptive_bilateral_denoise(
    image_bgr: np.ndarray,
    diameter: int = 9,
    sigma_color: float | None = None,
    sigma_space: float = 9.0,
) -> np.ndarray:
    """Range-adaptive bilateral approximation.

    Paper menyebut range-based adaptive bilateral filter tetapi tidak
    menerbitkan parameter implementasinya. Sigma warna diestimasi dari robust
    spread luminance agar filter menyesuaikan kontras setiap gambar.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if sigma_color is None:
        q25, q75 = np.percentile(gray, (25, 75))
        robust_sigma = max(float(q75 - q25) / 1.349, 5.0)
        sigma_color = float(np.clip(2.5 * robust_sigma, 15.0, 100.0))
    return cv2.bilateralFilter(
        image_bgr,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )


def approximate_plate_mask(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """Approximate ROI dengan differential transform + threshold.

    Ini disediakan untuk raw-image preprocessing. Dataset proyek juga memiliki
    counting mask yang lebih stabil dari tahap deteksi cawan sebelumnya.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=7.0)
    differential = cv2.absdiff(gray, blurred)
    differential = cv2.normalize(
        differential,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    _, thresholded = cv2.threshold(
        differential,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    thresholded = cv2.morphologyEx(
        thresholded,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=2,
    )

    contours, _ = cv2.findContours(
        thresholded,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    mask = np.zeros_like(gray)
    if not contours:
        return mask
    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask


def approximate_medium_mask(
    image_bgr: np.ndarray,
    plate_mask: np.ndarray,
) -> np.ndarray:
    """Pisahkan medium secara kasar melalui edge rejection."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    edge_zone = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    medium = (plate_mask > 0) & (edge_zone == 0)
    if medium.sum() < max(100, int(0.1 * (plate_mask > 0).sum())):
        medium = plate_mask > 0
    return np.where(medium, 255, 0).astype(np.uint8)


def logarithmic_intensity_correction(
    image_bgr: np.ndarray,
    medium_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Implementasi lg(I0)-lg(Ii) dari paper.

    Output float mempertahankan nilai absorbance. Output uint8 memakai robust
    percentile hanya untuk penyimpanan dan input jaringan pada proyek ini.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    intensity = lab[:, :, 0].astype(np.float32) / 255.0
    medium_values = intensity[medium_mask > 0]
    if medium_values.size < 100:
        raise RuntimeError("Area medium terlalu kecil untuk koreksi intensitas.")
    i0 = float(np.mean(medium_values))
    corrected = np.log10(i0 + EPSILON) - np.log10(intensity + EPSILON)

    valid_values = corrected[valid_mask > 0]
    if valid_values.size < 100:
        raise RuntimeError("Area valid terlalu kecil untuk normalisasi.")
    low, high = np.percentile(valid_values, (1.0, 99.0))
    if high - low < EPSILON:
        high = low + EPSILON
    scaled = np.clip((corrected - low) / (high - low), 0.0, 1.0)
    fill = float(np.median(scaled[medium_mask > 0]))
    scaled[valid_mask == 0] = fill
    return corrected.astype(np.float32), np.rint(scaled * 255).astype(np.uint8), i0


def preprocess_raw_image(
    image_bgr: np.ndarray,
    known_plate_mask: np.ndarray | None = None,
) -> PreprocessResult:
    denoised = adaptive_bilateral_denoise(image_bgr)
    plate_mask = (
        known_plate_mask
        if known_plate_mask is not None
        else approximate_plate_mask(denoised)
    )
    plate_mask = np.where(plate_mask > 0, 255, 0).astype(np.uint8)
    medium_mask = approximate_medium_mask(denoised, plate_mask)
    corrected, corrected_u8, i0 = logarithmic_intensity_correction(
        denoised,
        medium_mask,
        plate_mask,
    )
    return PreprocessResult(
        denoised=denoised,
        approximate_plate_mask=plate_mask,
        approximate_medium_mask=medium_mask,
        corrected_float=corrected,
        corrected_u8=corrected_u8,
        medium_intensity=i0,
    )


def make_edge_ring_label(
    plate_mask: np.ndarray,
    ring_width: int,
) -> np.ndarray:
    """Buat proxy label edge dari mask geometri cawan.

    Paper memakai label manual berupa area di antara lingkaran dalam dan luar.
    AGAR tidak menyediakan label tersebut. Mask ini adalah proxy eksplisit,
    bukan pengganti label manual untuk reproduksi ilmiah final.
    """
    binary = np.where(plate_mask > 0, 255, 0).astype(np.uint8)
    radius = max(1, int(round(ring_width / 2)))
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    outer = cv2.dilate(binary, kernel, iterations=1)
    inner = cv2.erode(binary, kernel, iterations=1)
    return cv2.subtract(outer, inner)


def inner_roi_from_edge_mask(edge_mask: np.ndarray) -> np.ndarray:
    """Ambil area di dalam kontur edge U2-Net."""
    edge = np.where(edge_mask > 0, 255, 0).astype(np.uint8)
    edge = cv2.morphologyEx(
        edge,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    contours, hierarchy = cv2.findContours(
        edge,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    roi = np.zeros_like(edge)
    if hierarchy is not None and contours:
        hierarchy = hierarchy[0]
        candidates: list[tuple[float, np.ndarray]] = []
        for index, contour in enumerate(contours):
            parent = int(hierarchy[index][3])
            if parent >= 0:
                candidates.append((cv2.contourArea(contour), contour))
        if candidates:
            contour = max(candidates, key=lambda item: item[0])[1]
            cv2.drawContours(roi, [contour], -1, 255, cv2.FILLED)
            return roi

    # Fallback: komponen background yang mengandung pusat gambar.
    inverse = np.where(edge == 0, 1, 0).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(inverse, connectivity=8)
    if n_labels <= 1:
        return roi
    center_label = int(labels[labels.shape[0] // 2, labels.shape[1] // 2])
    if center_label == 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        center_label = int(np.argmax(sizes))
    roi[labels == center_label] = 255
    return roi

