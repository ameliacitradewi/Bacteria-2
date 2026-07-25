from __future__ import annotations

import importlib
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import nms
from tqdm import tqdm

from .common import read_image, resolve_device, save_json, write_image
from .metrics import (
    decode_centernet,
    detection_match_counts,
    largest_component,
    precision_recall_f1,
    summarize_count_metrics,
)
from .models import ResNet50FPNCenterNet, U2NETP, load_state_dict_flexible


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_coordinates(height: int, width: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    return [
        (x0, y0)
        for y0 in _axis_starts(height, tile_size, overlap)
        for x0 in _axis_starts(width, tile_size, overlap)
    ]


def _import_project_module(project_root: Path, module_name: str):
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return importlib.import_module(module_name)


def _image_tensor(image_bgr: np.ndarray, size: int | None = None) -> torch.Tensor:
    if size is not None:
        image_bgr = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_rgb = (image_rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1))).float()


def _clean_plate_mask(mask: np.ndarray) -> np.ndarray:
    binary = largest_component(mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return binary
    radius = max(2, int(round(0.006 * min(binary.shape))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(binary)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, thickness=-1)
    return filled


def infer_plate_ellipse(
    image_bgr: np.ndarray,
    model: U2NETP,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    plate_cfg = config["plate"]
    image_size = int(plate_cfg["image_size"])
    threshold = float(plate_cfg.get("threshold", 0.5))
    tensor = _image_tensor(image_bgr, image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        probability_small = torch.sigmoid(model(tensor)[0])[0, 0].cpu().numpy()
    mask_small = _clean_plate_mask(probability_small >= threshold)
    if mask_small.sum() == 0:
        raise RuntimeError("U2-NetP tidak menghasilkan mask cawan.")

    height, width = image_bgr.shape[:2]
    mask_original = cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_NEAREST)
    mask_original = _clean_plate_mask(mask_original)
    contours, _ = cv2.findContours(
        (mask_original * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise RuntimeError("Kontur cawan tidak ditemukan dari mask U2-NetP.")
    contour = max(contours, key=cv2.contourArea)
    area_fraction = cv2.contourArea(contour) / max(float(height * width), 1.0)
    if area_fraction < 0.05:
        raise RuntimeError(f"Mask cawan terlalu kecil: area_fraction={area_fraction:.4f}")

    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipseAMS(contour)
        except cv2.error:
            ellipse = cv2.fitEllipse(contour)
    else:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        ellipse = ((float(cx), float(cy)), (2.0 * float(radius), 2.0 * float(radius)), 0.0)
    (cx, cy), (axis_1, axis_2), angle = ellipse
    axis_ratio = min(axis_1, axis_2) / max(axis_1, axis_2, 1.0)
    if axis_ratio < 0.55:
        raise RuntimeError(f"Prediksi cawan terlalu pipih: axis_ratio={axis_ratio:.3f}")
    return {
        "ellipse": ((float(cx), float(cy)), (float(axis_1), float(axis_2)), float(angle)),
        "probability_small": probability_small,
        "mask_small": mask_small,
        "mask_original": mask_original,
        "area_fraction": float(area_fraction),
        "axis_ratio": float(axis_ratio),
    }


def normalize_raw_photo(
    image_bgr: np.ndarray,
    plate_model: U2NETP,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    project_root = Path(config["project_root"])
    plate_module = _import_project_module(project_root, "detect_outer_plate_strategy_b")
    intensity_module = _import_project_module(project_root, "normalize_agar_intensity_v1")

    plate_result = infer_plate_ellipse(image_bgr, plate_model, config, device)
    classical_cfg = config.get("classical_plate", {})
    crop = plate_module.create_crop_outputs(
        image=image_bgr,
        physical_ellipse=plate_result["ellipse"],
        counting_scale=float(classical_cfg.get("counting_scale", 1.0)),
        crop_padding_ratio=float(classical_cfg.get("crop_padding_ratio", 1.04)),
        target_size=int(classical_cfg.get("target_size", 2048)),
    )
    intensity_cfg = config.get("intensity", {})
    intensity = intensity_module.process_one(
        image_bgr=crop["normalized_raw"],
        counting_mask=crop["normalized_counting_mask"],
        safe_erode_ratio=float(intensity_cfg.get("safe_erode_ratio", 0.025)),
        outlier_mad_multiplier=float(intensity_cfg.get("outlier_mad", 3.5)),
        safe_minimum_fraction=float(intensity_cfg.get("safe_minimum_fraction", 0.15)),
        flatfield_sigma_ratio=float(intensity_cfg.get("flatfield_sigma_ratio", 0.04)),
        percentile_low=float(intensity_cfg.get("percentile_low", 1.0)),
        percentile_high=float(intensity_cfg.get("percentile_high", 99.0)),
    )
    return {**plate_result, **crop, "flatfield_gray": intensity["flatfield_gray"]}


def transform_ground_truth_boxes(
    objects: pd.DataFrame,
    crop: dict[str, Any],
) -> np.ndarray:
    if objects.empty:
        return np.empty((0, 4), dtype=np.float32)
    scale = float(crop["normalization_scale"])
    x0 = float(crop["crop_x0"])
    y0 = float(crop["crop_y0"])
    target_size = int(crop["normalized_counting_mask"].shape[0])
    mask = crop["normalized_counting_mask"]
    output: list[list[float]] = []
    for _, row in objects.iterrows():
        x1 = (float(row["x"]) - x0) * scale
        y1 = (float(row["y"]) - y0) * scale
        x2 = x1 + float(row["width"]) * scale
        y2 = y1 + float(row["height"]) * scale
        cx = int(round(0.5 * (x1 + x2)))
        cy = int(round(0.5 * (y1 + y2)))
        if not (0 <= cx < target_size and 0 <= cy < target_size and mask[cy, cx] > 0):
            continue
        x1 = float(np.clip(x1, 0, target_size))
        y1 = float(np.clip(y1, 0, target_size))
        x2 = float(np.clip(x2, 0, target_size))
        y2 = float(np.clip(y2, 0, target_size))
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            output.append([x1, y1, x2, y2])
    return np.asarray(output, dtype=np.float32).reshape(-1, 4)


def infer_colonies(
    flatfield_gray: np.ndarray,
    counting_mask: np.ndarray,
    model: ResNet50FPNCenterNet,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    colony_cfg = config["colony"]
    tile_size = int(colony_cfg["tile_size"])
    overlap = int(colony_cfg["tile_overlap"])
    edge_margin = int(colony_cfg.get("tile_edge_margin", 16))
    batch_size = int(colony_cfg.get("batch_size", 8))
    height, width = flatfield_gray.shape
    coordinates = tile_coordinates(height, width, tile_size, overlap)
    bgr = cv2.cvtColor(flatfield_gray, cv2.COLOR_GRAY2BGR)
    all_boxes: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []

    for start in range(0, len(coordinates), batch_size):
        current = coordinates[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        for x0, y0 in current:
            crop = bgr[y0 : y0 + tile_size, x0 : x0 + tile_size]
            if crop.shape[:2] != (tile_size, tile_size):
                crop = cv2.copyMakeBorder(
                    crop,
                    0,
                    tile_size - crop.shape[0],
                    0,
                    tile_size - crop.shape[1],
                    cv2.BORDER_REFLECT_101,
                )
            tensors.append(_image_tensor(crop))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            outputs = model(batch)
            decoded = decode_centernet(
                outputs,
                stride=int(colony_cfg.get("output_stride", 4)),
                score_threshold=float(colony_cfg.get("score_threshold", 0.25)),
                nms_iou_threshold=float(colony_cfg.get("tile_nms_iou", 0.30)),
                topk=int(colony_cfg.get("topk", 500)),
            )
        for (x0, y0), prediction in zip(current, decoded, strict=False):
            boxes = prediction["boxes"].detach().cpu()
            scores = prediction["scores"].detach().cpu()
            if len(boxes) == 0:
                continue
            cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
            cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
            keep = torch.ones(len(boxes), dtype=torch.bool)
            if x0 > 0:
                keep &= cx >= edge_margin
            if y0 > 0:
                keep &= cy >= edge_margin
            if x0 + tile_size < width:
                keep &= cx < tile_size - edge_margin
            if y0 + tile_size < height:
                keep &= cy < tile_size - edge_margin
            boxes = boxes[keep]
            scores = scores[keep]
            if len(boxes):
                boxes[:, 0::2] += x0
                boxes[:, 1::2] += y0
                all_boxes.append(boxes)
                all_scores.append(scores)

    if not all_boxes:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
        }
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    keep = nms(boxes, scores, float(colony_cfg.get("global_nms_iou", 0.25)))
    boxes = boxes[keep]
    scores = scores[keep]
    center_x = ((boxes[:, 0] + boxes[:, 2]) * 0.5).round().long()
    center_y = ((boxes[:, 1] + boxes[:, 3]) * 0.5).round().long()
    valid = (
        (center_x >= 0)
        & (center_x < counting_mask.shape[1])
        & (center_y >= 0)
        & (center_y < counting_mask.shape[0])
    )
    inside = torch.zeros_like(valid)
    valid_indices = torch.where(valid)[0]
    if len(valid_indices):
        values = counting_mask[
            center_y[valid_indices].numpy(), center_x[valid_indices].numpy()
        ] > 0
        inside[valid_indices] = torch.from_numpy(values)
    return {"boxes": boxes[inside], "scores": scores[inside]}


def _draw_overlay(
    image_bgr: np.ndarray,
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    scores: np.ndarray,
    text: str,
) -> np.ndarray:
    output = image_bgr.copy()
    for box in ground_truth:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for box, score in zip(predicted, scores, strict=False):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(output, f"{score:.2f}", (x1, max(y1 - 3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.rectangle(output, (0, 0), (min(output.shape[1] - 1, 1100), 58), (0, 0, 0), -1)
    cv2.putText(output, text, (15, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def _numeric_runtime_summary(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    row: dict[str, Any] = {
        "split": split,
        "n_images": int(len(frame)),
        "plate_localization_success_rate": float(frame["plate_success"].mean()) if len(frame) else math.nan,
    }
    for column in (
        "plate_seconds",
        "normalization_seconds",
        "detector_seconds",
        "total_seconds",
    ):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        row[f"{column}_mean"] = float(values.mean()) if len(values) else math.nan
        row[f"{column}_median"] = float(values.median()) if len(values) else math.nan
        row[f"{column}_p90"] = float(values.quantile(0.90)) if len(values) else math.nan
    return pd.DataFrame([row])


def evaluate_production_pipeline(
    config: dict[str, Any],
    split: str = "test",
) -> dict[str, str]:
    device = resolve_device(str(config.get("device", "auto")))
    output_dir = Path(config["output_dir"])
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    plate_model = U2NETP().to(device).eval()
    load_state_dict_flexible(
        plate_model,
        str(output_dir / "checkpoints" / "u2netp_plate_best.pt"),
        map_location=device,
    )
    colony_cfg = config["colony"]
    colony_model = ResNet50FPNCenterNet(
        fpn_channels=int(colony_cfg.get("fpn_channels", 128)), pretrained=False
    ).to(device).eval()
    load_state_dict_flexible(
        colony_model,
        str(output_dir / "checkpoints" / "resnet50_fpn_centernet_best.pt"),
        map_location=device,
    )

    manifest = pd.read_csv(
        Path(config["metadata_dir"]) / "image_manifest.csv", dtype={"image_id": "string"}
    )
    aliases = [split, "validation"] if split == "val" else [split]
    manifest = manifest[
        manifest["split"].isin(aliases)
        & manifest["plate_condition"].isin(["countable", "empty"])
    ].copy()
    max_eval_images = config.get("evaluation", {}).get("max_images")
    if max_eval_images:
        manifest = manifest.sample(
            n=min(int(max_eval_images), len(manifest)),
            random_state=int(config.get("seed", 42)),
        ).sort_values("image_id")
    objects = pd.read_csv(
        Path(config["metadata_dir"]) / "object_annotations.csv", dtype={"image_id": "string"}
    )
    objects = objects[objects["object_type"].astype(str) == "colony"].copy()
    grouped = {key: value for key, value in objects.groupby("image_id")}

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    total_tp = total_fp = total_fn = 0
    rows: list[dict[str, Any]] = []
    visual_dir = output_dir / "visual_samples" / "07_production_e2e_predictions"
    visual_count = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"Production E2E {split}"):
        image_id = str(row["image_id"])
        raw = read_image(Path(config["dataset_root"]) / str(row["image_path"]))
        true_count = int(row["colonies_number"])
        original_group = grouped.get(image_id, pd.DataFrame())
        plate_seconds = normalization_seconds = detector_seconds = math.nan
        status = "success"
        error_message = ""
        started_total = time.perf_counter()
        try:
            started = time.perf_counter()
            plate_result = infer_plate_ellipse(raw, plate_model, config, device)
            plate_seconds = time.perf_counter() - started

            started = time.perf_counter()
            # Avoid running U2-Net twice: reuse the predicted ellipse here.
            project_root = Path(config["project_root"])
            plate_module = _import_project_module(project_root, "detect_outer_plate_strategy_b")
            intensity_module = _import_project_module(project_root, "normalize_agar_intensity_v1")
            classical_cfg = config.get("classical_plate", {})
            crop = plate_module.create_crop_outputs(
                image=raw,
                physical_ellipse=plate_result["ellipse"],
                counting_scale=float(classical_cfg.get("counting_scale", 1.0)),
                crop_padding_ratio=float(classical_cfg.get("crop_padding_ratio", 1.04)),
                target_size=int(classical_cfg.get("target_size", 2048)),
            )
            intensity_cfg = config.get("intensity", {})
            normalized = intensity_module.process_one(
                image_bgr=crop["normalized_raw"],
                counting_mask=crop["normalized_counting_mask"],
                safe_erode_ratio=float(intensity_cfg.get("safe_erode_ratio", 0.025)),
                outlier_mad_multiplier=float(intensity_cfg.get("outlier_mad", 3.5)),
                safe_minimum_fraction=float(intensity_cfg.get("safe_minimum_fraction", 0.15)),
                flatfield_sigma_ratio=float(intensity_cfg.get("flatfield_sigma_ratio", 0.04)),
                percentile_low=float(intensity_cfg.get("percentile_low", 1.0)),
                percentile_high=float(intensity_cfg.get("percentile_high", 99.0)),
            )
            gt_boxes_np = transform_ground_truth_boxes(original_group, crop)
            normalization_seconds = time.perf_counter() - started

            started = time.perf_counter()
            prediction = infer_colonies(
                normalized["flatfield_gray"],
                crop["normalized_counting_mask"],
                colony_model,
                config,
                device,
            )
            detector_seconds = time.perf_counter() - started
            pred_boxes = prediction["boxes"].cpu()
            pred_scores = prediction["scores"].cpu()
            gt_boxes = torch.from_numpy(gt_boxes_np)
            visual_base = cv2.cvtColor(normalized["flatfield_gray"], cv2.COLOR_GRAY2BGR)
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            pred_boxes = torch.empty((0, 4), dtype=torch.float32)
            pred_scores = torch.empty((0,), dtype=torch.float32)
            if original_group.empty:
                gt_boxes = torch.empty((0, 4), dtype=torch.float32)
            else:
                gt_boxes = torch.tensor(
                    np.stack(
                        [
                            original_group["x"].to_numpy(np.float32),
                            original_group["y"].to_numpy(np.float32),
                            (original_group["x"] + original_group["width"]).to_numpy(np.float32),
                            (original_group["y"] + original_group["height"]).to_numpy(np.float32),
                        ],
                        axis=1,
                    )
                )
            visual_base = raw

        total_seconds = time.perf_counter() - started_total
        tp, fp, fn = detection_match_counts(pred_boxes, gt_boxes, iou_threshold=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        metric.update(
            [{"boxes": pred_boxes, "scores": pred_scores, "labels": torch.ones(len(pred_boxes), dtype=torch.int64)}],
            [{"boxes": gt_boxes, "labels": torch.ones(len(gt_boxes), dtype=torch.int64)}],
        )
        pred_count = int(len(pred_boxes))
        rows.append(
            {
                "image_id": image_id,
                "split": split,
                "plate_condition": str(row["plate_condition"]),
                "plate_success": int(status == "success"),
                "status": status,
                "error_message": error_message,
                "true_count": true_count,
                "true_count_boxes": int(row["n_colony_boxes"]),
                "pred_count": pred_count,
                "error": pred_count - true_count,
                "absolute_error": abs(pred_count - true_count),
                "relative_error": abs(pred_count - true_count) / max(true_count, 1),
                "tp_iou50": tp,
                "fp_iou50": fp,
                "fn_iou50": fn,
                "plate_seconds": plate_seconds,
                "normalization_seconds": normalization_seconds,
                "detector_seconds": detector_seconds,
                "total_seconds": total_seconds,
            }
        )
        if visual_count < 5:
            display = visual_base
            gt_display = gt_boxes.numpy().copy()
            pred_display = pred_boxes.numpy().copy()
            if max(display.shape[:2]) > 2048:
                scale = 2048 / max(display.shape[:2])
                display = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                gt_display *= scale
                pred_display *= scale
            overlay = _draw_overlay(
                display,
                gt_display,
                pred_display,
                pred_scores.numpy(),
                f"{image_id} | {status} | GT={true_count} Pred={pred_count} Error={pred_count-true_count:+d}",
            )
            write_image(visual_dir / f"{visual_count + 1:02d}_{image_id}.jpg", overlay)
            visual_count += 1

    per_image = pd.DataFrame(rows)
    per_image_path = metrics_dir / f"production_e2e_{split}_per_image.csv"
    per_image.to_csv(per_image_path, index=False)

    count_summary = summarize_count_metrics(per_image)
    count_summary.insert(0, "split", split)
    count_path = metrics_dir / f"production_e2e_{split}_count_summary.csv"
    count_summary.to_csv(count_path, index=False)

    map_result = metric.compute()
    prf = precision_recall_f1(total_tp, total_fp, total_fn)
    detection_summary = pd.DataFrame(
        [
            {
                "split": split,
                "map": float(map_result["map"]),
                "map_50": float(map_result["map_50"]),
                "map_75": float(map_result["map_75"]),
                "precision_iou50": prf["precision"],
                "recall_iou50": prf["recall"],
                "f1_iou50": prf["f1"],
                "tp_iou50": total_tp,
                "fp_iou50": total_fp,
                "fn_iou50": total_fn,
            }
        ]
    )
    detection_path = metrics_dir / f"production_e2e_{split}_detection_summary.csv"
    detection_summary.to_csv(detection_path, index=False)

    runtime_summary = _numeric_runtime_summary(per_image, split)
    runtime_path = metrics_dir / f"production_e2e_{split}_runtime_summary.csv"
    runtime_summary.to_csv(runtime_path, index=False)
    return {
        "per_image": str(per_image_path),
        "count_summary": str(count_path),
        "detection_summary": str(detection_path),
        "runtime_summary": str(runtime_path),
    }


def predict_single_image(
    config: dict[str, Any],
    image_path: str | Path,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    device = resolve_device(str(config.get("device", "auto")))
    output_dir = Path(config["output_dir"])
    plate_model = U2NETP().to(device).eval()
    load_state_dict_flexible(
        plate_model,
        str(output_dir / "checkpoints" / "u2netp_plate_best.pt"),
        map_location=device,
    )
    colony_cfg = config["colony"]
    colony_model = ResNet50FPNCenterNet(
        fpn_channels=int(colony_cfg.get("fpn_channels", 128)), pretrained=False
    ).to(device).eval()
    load_state_dict_flexible(
        colony_model,
        str(output_dir / "checkpoints" / "resnet50_fpn_centernet_best.pt"),
        map_location=device,
    )

    image_path = Path(image_path)
    raw = read_image(image_path)
    started = time.perf_counter()
    normalized = normalize_raw_photo(raw, plate_model, config, device)
    prediction = infer_colonies(
        normalized["flatfield_gray"],
        normalized["normalized_counting_mask"],
        colony_model,
        config,
        device,
    )
    elapsed = time.perf_counter() - started
    count = int(len(prediction["boxes"]))
    overlay = _draw_overlay(
        cv2.cvtColor(normalized["flatfield_gray"], cv2.COLOR_GRAY2BGR),
        np.empty((0, 4), dtype=np.float32),
        prediction["boxes"].numpy(),
        prediction["scores"].numpy(),
        f"Predicted colonies={count} | elapsed={elapsed:.3f}s",
    )
    destination = Path(destination) if destination else output_dir / "inference" / image_path.stem
    destination.mkdir(parents=True, exist_ok=True)
    overlay_path = destination / "colony_overlay.jpg"
    flatfield_path = destination / "flatfield.png"
    mask_path = destination / "counting_mask.png"
    write_image(overlay_path, overlay)
    write_image(flatfield_path, normalized["flatfield_gray"])
    write_image(mask_path, normalized["normalized_counting_mask"])
    result = {
        "image_path": str(image_path),
        "predicted_count": count,
        "elapsed_seconds": elapsed,
        "boxes": prediction["boxes"].tolist(),
        "scores": prediction["scores"].tolist(),
        "overlay_path": str(overlay_path),
        "flatfield_path": str(flatfield_path),
        "counting_mask_path": str(mask_path),
    }
    save_json(destination / "prediction.json", result)
    return result
