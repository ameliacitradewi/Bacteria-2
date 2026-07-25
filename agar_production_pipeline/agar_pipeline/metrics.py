from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, nms


def binary_mask_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-7,
) -> dict[str, float]:
    pred = prediction.astype(bool)
    true = target.astype(bool)
    tp = float(np.logical_and(pred, true).sum())
    fp = float(np.logical_and(pred, ~true).sum())
    fn = float(np.logical_and(~pred, true).sum())
    intersection = tp
    union = float(np.logical_or(pred, true).sum())
    return {
        "dice": (2 * intersection + eps) / (pred.sum() + true.sum() + eps),
        "iou": (intersection + eps) / (union + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
    }


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n_labels <= 1:
        return np.zeros_like(binary)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == index).astype(np.uint8)


def mask_geometry(mask: np.ndarray) -> dict[str, float]:
    component = largest_component(mask)
    ys, xs = np.where(component > 0)
    if len(xs) == 0:
        return {
            "center_x": math.nan,
            "center_y": math.nan,
            "area": 0.0,
            "equivalent_radius": 0.0,
        }
    area = float(len(xs))
    return {
        "center_x": float(xs.mean()),
        "center_y": float(ys.mean()),
        "area": area,
        "equivalent_radius": float(math.sqrt(area / math.pi)),
    }


def plate_geometry_errors(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    pred = mask_geometry(prediction)
    true = mask_geometry(target)
    height, width = target.shape
    diagonal = math.hypot(height, width)
    if math.isnan(pred["center_x"]):
        center_error = 1.0
    else:
        center_error = math.hypot(
            pred["center_x"] - true["center_x"],
            pred["center_y"] - true["center_y"],
        ) / max(diagonal, 1.0)
    area_error = abs(pred["area"] - true["area"]) / max(true["area"], 1.0)
    radius_error = abs(pred["equivalent_radius"] - true["equivalent_radius"]) / max(
        true["equivalent_radius"], 1.0
    )
    return {
        "center_error_normalized": center_error,
        "area_relative_error": area_error,
        "radius_relative_error": radius_error,
        "prediction_empty": float(pred["area"] == 0),
    }


def centernet_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    predictions = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = targets.eq(1).float()
    negative = targets.lt(1).float()
    negative_weights = torch.pow(1 - targets, beta)
    positive_loss = -torch.log(predictions) * torch.pow(1 - predictions, alpha) * positive
    negative_loss = (
        -torch.log(1 - predictions)
        * torch.pow(predictions, alpha)
        * negative_weights
        * negative
    )
    n_positive = positive.sum()
    if n_positive.item() == 0:
        return negative_loss.sum()
    return (positive_loss.sum() + negative_loss.sum()) / n_positive


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    expanded = mask.expand_as(prediction)
    denominator = expanded.sum().clamp(min=1.0)
    return (torch.abs(prediction - target) * expanded).sum() / denominator


def colony_detection_loss(
    outputs: dict[str, torch.Tensor],
    heatmap: torch.Tensor,
    size: torch.Tensor,
    offset: torch.Tensor,
    reg_mask: torch.Tensor,
    size_weight: float = 0.1,
    offset_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    heatmap_loss = centernet_focal_loss(outputs["heatmap_logits"], heatmap)
    size_prediction = F.softplus(outputs["size_raw"])
    offset_prediction = torch.sigmoid(outputs["offset_raw"])
    size_loss = masked_l1_loss(size_prediction, size, reg_mask)
    offset_loss = masked_l1_loss(offset_prediction, offset, reg_mask)
    total = heatmap_loss + size_weight * size_loss + offset_weight * offset_loss
    return {
        "loss": total,
        "heatmap_loss": heatmap_loss,
        "size_loss": size_loss,
        "offset_loss": offset_loss,
    }


def decode_centernet(
    outputs: dict[str, torch.Tensor],
    stride: int,
    score_threshold: float,
    nms_iou_threshold: float,
    topk: int = 500,
) -> list[dict[str, torch.Tensor]]:
    heatmap = torch.sigmoid(outputs["heatmap_logits"])
    size = F.softplus(outputs["size_raw"])
    offset = torch.sigmoid(outputs["offset_raw"])
    pooled = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    heatmap = heatmap * pooled.eq(heatmap)
    batch, _, height, width = heatmap.shape
    results: list[dict[str, torch.Tensor]] = []
    for batch_index in range(batch):
        flat = heatmap[batch_index, 0].reshape(-1)
        current_topk = min(int(topk), flat.numel())
        scores, indices = torch.topk(flat, k=current_topk)
        keep = scores >= score_threshold
        scores = scores[keep]
        indices = indices[keep]
        if scores.numel() == 0:
            results.append(
                {
                    "boxes": torch.empty((0, 4), device=heatmap.device),
                    "scores": torch.empty((0,), device=heatmap.device),
                }
            )
            continue
        ys = torch.div(indices, width, rounding_mode="floor")
        xs = indices % width
        sizes = size[batch_index, :, ys, xs].transpose(0, 1)
        offsets = offset[batch_index, :, ys, xs].transpose(0, 1)
        centers_x = (xs.float() + offsets[:, 0]) * stride
        centers_y = (ys.float() + offsets[:, 1]) * stride
        box_widths = sizes[:, 0] * stride
        box_heights = sizes[:, 1] * stride
        boxes = torch.stack(
            [
                centers_x - box_widths / 2,
                centers_y - box_heights / 2,
                centers_x + box_widths / 2,
                centers_y + box_heights / 2,
            ],
            dim=1,
        )
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width * stride)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height * stride)
        valid = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
        boxes = boxes[valid]
        scores = scores[valid]
        if boxes.numel() == 0:
            results.append(
                {
                    "boxes": torch.empty((0, 4), device=heatmap.device),
                    "scores": torch.empty((0,), device=heatmap.device),
                }
            )
            continue
        selected = nms(boxes, scores, nms_iou_threshold)
        results.append({"boxes": boxes[selected], "scores": scores[selected]})
    return results


def detection_match_counts(
    predicted_boxes: torch.Tensor,
    ground_truth_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    if len(predicted_boxes) == 0:
        return 0, 0, int(len(ground_truth_boxes))
    if len(ground_truth_boxes) == 0:
        return 0, int(len(predicted_boxes)), 0
    ious = box_iou(predicted_boxes.cpu(), ground_truth_boxes.cpu())
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    candidates: list[tuple[float, int, int]] = []
    for pred_index in range(ious.shape[0]):
        for target_index in range(ious.shape[1]):
            value = float(ious[pred_index, target_index])
            if value >= iou_threshold:
                candidates.append((value, pred_index, target_index))
    candidates.sort(reverse=True)
    for _, pred_index, target_index in candidates:
        if pred_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(pred_index)
        used_targets.add(target_index)
    tp = len(used_predictions)
    fp = len(predicted_boxes) - tp
    fn = len(ground_truth_boxes) - tp
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize_count_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    true = frame["true_count"].to_numpy(np.float64)
    pred = frame["pred_count"].to_numpy(np.float64)
    error = pred - true
    absolute = np.abs(error)
    nonzero = true > 0
    relative = np.zeros_like(true)
    relative[nonzero] = absolute[nonzero] / true[nonzero]
    smape = 2 * absolute / np.maximum(np.abs(true) + np.abs(pred), 1.0)
    summary = {
        "n_images": len(frame),
        "mae": float(np.mean(absolute)),
        "median_absolute_error": float(np.median(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_bias": float(np.mean(error)),
        "mape_nonzero": float(np.mean(relative[nonzero])) if nonzero.any() else math.nan,
        "smape": float(np.mean(smape)),
        "within_5_colonies": float(np.mean(absolute <= 5)),
        "within_10_colonies": float(np.mean(absolute <= 10)),
        "within_10_percent_or_2": float(
            np.mean(absolute <= np.maximum(2.0, true * 0.10))
        ),
        "pearson_correlation": float(np.corrcoef(true, pred)[0, 1])
        if len(frame) > 1 and np.std(true) > 0 and np.std(pred) > 0
        else math.nan,
    }
    return pd.DataFrame([summary])


def summarize_numeric_metrics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": column,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "median": float(values.median()),
                "p10": float(values.quantile(0.10)),
                "p90": float(values.quantile(0.90)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)
