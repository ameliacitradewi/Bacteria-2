from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import nms
from tqdm import tqdm

from .common import read_image, resolve_device, save_json, seed_everything, write_image
from .data import (
    ColonyTileDataset,
    PlateSegmentationDataset,
    build_plate_training_frame,
    collate_colony_batch,
    materialize_colony_tiles,
)
from .metrics import (
    binary_mask_metrics,
    colony_detection_loss,
    decode_centernet,
    detection_match_counts,
    plate_geometry_errors,
    precision_recall_f1,
    summarize_count_metrics,
    summarize_numeric_metrics,
)
from .models import (
    ColonyProductionWrapper,
    ResNet50FPNCenterNet,
    U2NETP,
    U2NetProductionWrapper,
    load_state_dict_flexible,
    total_parameter_count,
    trainable_parameter_count,
)


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + 1.0) / (denominator + 1.0)).mean()


def deep_supervision_loss(outputs: tuple[torch.Tensor, ...], target: torch.Tensor) -> torch.Tensor:
    weights = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1]
    total = torch.zeros((), device=target.device)
    weight_sum = 0.0
    for weight, logits in zip(weights, outputs, strict=False):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        total = total + weight * (bce + _dice_loss(logits, target))
        weight_sum += weight
    return total / weight_sum


def _loader_kwargs(config: dict[str, Any], workers: int) -> dict[str, Any]:
    use_cuda = resolve_device(str(config.get("device", "auto"))).type == "cuda"
    return {
        "num_workers": workers,
        "pin_memory": use_cuda,
        "persistent_workers": workers > 0,
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


def _load_original_resized(path: str, size: int) -> np.ndarray:
    image = read_image(path)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _plate_visual(
    image_bgr: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    label: str,
) -> np.ndarray:
    target_u8 = (target > 0).astype(np.uint8) * 255
    prediction_u8 = (prediction > 0).astype(np.uint8) * 255
    overlay = image_bgr.copy()
    target_contours, _ = cv2.findContours(target_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    prediction_contours, _ = cv2.findContours(
        prediction_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, target_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(overlay, prediction_contours, -1, (0, 0, 255), 2)
    probability_u8 = np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8)
    probability_bgr = cv2.applyColorMap(probability_u8, cv2.COLORMAP_VIRIDIS)
    mask_panel = np.zeros_like(image_bgr)
    mask_panel[:, :, 1] = target_u8
    mask_panel[:, :, 2] = prediction_u8
    montage = np.concatenate([image_bgr, mask_panel, probability_bgr, overlay], axis=1)
    cv2.putText(
        montage,
        label,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return montage


@torch.no_grad()
def evaluate_plate_model(
    model: U2NETP,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    output_dir: Path,
    split: str,
    save_visuals: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    visual_count = 0
    visual_dir = output_dir / "visual_samples" / "05_u2netp_predictions"
    for batch in tqdm(loader, desc=f"Evaluasi plate {split}", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].cpu().numpy()[:, 0] > 0.5
        probabilities = torch.sigmoid(model(images)[0]).cpu().numpy()[:, 0]
        predictions = probabilities >= threshold
        for index in range(len(images)):
            pixel = binary_mask_metrics(predictions[index], targets[index])
            geometry = plate_geometry_errors(predictions[index], targets[index])
            row = {
                "image_id": str(batch["image_id"][index]),
                "split": split,
                **pixel,
                **geometry,
            }
            rows.append(row)
            if save_visuals and visual_count < 5:
                size = targets[index].shape[0]
                image_bgr = _load_original_resized(str(batch["image_path"][index]), size)
                montage = _plate_visual(
                    image_bgr,
                    targets[index],
                    predictions[index],
                    probabilities[index],
                    f"{row['image_id']} | Dice={row['dice']:.3f} IoU={row['iou']:.3f}",
                )
                write_image(visual_dir / f"{visual_count + 1:02d}_{row['image_id']}.jpg", montage)
                visual_count += 1
    frame = pd.DataFrame(rows)
    summary = summarize_numeric_metrics(
        frame,
        [
            "dice",
            "iou",
            "precision",
            "recall",
            "center_error_normalized",
            "area_relative_error",
            "radius_relative_error",
            "prediction_empty",
        ],
    )
    return frame, summary


def train_plate_model(config: dict[str, Any]) -> dict[str, str]:
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    output_dir = Path(config["output_dir"])
    metrics_dir = output_dir / "metrics"
    checkpoints_dir = output_dir / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    frame = build_plate_training_frame(config)
    plate_cfg = config["plate"]
    image_size = int(plate_cfg["image_size"])
    batch_size = int(plate_cfg["batch_size"])
    workers = int(plate_cfg.get("workers", 4))
    train_frame = frame[frame["split"] == "train"].copy()
    val_frame = frame[frame["split"].isin(["val", "validation"])].copy()
    test_frame = frame[frame["split"] == "test"].copy()
    max_train = plate_cfg.get("max_train_images")
    if max_train:
        train_frame = train_frame.sample(
            n=min(int(max_train), len(train_frame)),
            random_state=int(config.get("seed", 42)),
        )

    train_dataset = PlateSegmentationDataset(
        train_frame, config["dataset_root"], image_size, training=True
    )
    val_dataset = PlateSegmentationDataset(
        val_frame, config["dataset_root"], image_size, training=False
    )
    test_dataset = PlateSegmentationDataset(
        test_frame, config["dataset_root"], image_size, training=False
    )
    loader_extra = _loader_kwargs(config, workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **loader_extra,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_extra,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_extra,
    )

    model = U2NETP().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(plate_cfg["learning_rate"]),
        weight_decay=float(plate_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(plate_cfg["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    threshold = float(plate_cfg.get("threshold", 0.5))
    patience = int(plate_cfg.get("early_stopping_patience", 10))
    best_dice = -math.inf
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    best_path = checkpoints_dir / "u2netp_plate_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        started = time.perf_counter()
        for batch in tqdm(train_loader, desc=f"Plate epoch {epoch}/{epochs}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(images)
                loss = deep_supervision_loss(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item()) * len(images)
            train_samples += len(images)
        scheduler.step()

        val_per_image, _ = evaluate_plate_model(
            model,
            val_loader,
            device,
            threshold,
            output_dir,
            split="val",
            save_visuals=False,
        )
        val_dice = float(val_per_image["dice"].mean()) if not val_per_image.empty else 0.0
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_samples, 1),
            "val_dice": val_dice,
            "val_iou": float(val_per_image["iou"].mean()) if not val_per_image.empty else 0.0,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(metrics_dir / "plate_training_history.csv", index=False)
        if val_dice > best_dice:
            best_dice = val_dice
            stale_epochs = 0
            _save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_dice, config)
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    load_state_dict_flexible(model, str(best_path), map_location=device)
    for split, loader in (("val", val_loader), ("test", test_loader)):
        per_image, summary = evaluate_plate_model(
            model,
            loader,
            device,
            threshold,
            output_dir,
            split=split,
            save_visuals=split == "test",
        )
        per_image.to_csv(metrics_dir / f"plate_{split}_per_image.csv", index=False)
        summary.to_csv(metrics_dir / f"plate_{split}_summary.csv", index=False)

    model_info = {
        "model": "U2NETP",
        "total_parameters": total_parameter_count(model),
        "trainable_parameters": trainable_parameter_count(model),
        "image_size": image_size,
        "threshold": threshold,
        "best_val_dice": best_dice,
        "checkpoint": str(best_path),
        "device": str(device),
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "test_images": len(test_dataset),
    }
    save_json(output_dir / "metadata" / "plate_model_info.json", model_info)
    return {
        "checkpoint": str(best_path),
        "history": str(metrics_dir / "plate_training_history.csv"),
        "test_metrics": str(metrics_dir / "plate_test_per_image.csv"),
    }


def _colony_epoch(
    model: ResNet50FPNCenterNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    colony_cfg: dict[str, Any],
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    n_samples = 0
    tp = fp = fn = 0
    for batch in tqdm(loader, desc="Train colony" if training else "Validasi colony", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        heatmap = batch["heatmap"].to(device, non_blocking=True)
        size = batch["size"].to(device, non_blocking=True)
        offset = batch["offset"].to(device, non_blocking=True)
        reg_mask = batch["reg_mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(images)
                losses = colony_detection_loss(
                    outputs,
                    heatmap,
                    size,
                    offset,
                    reg_mask,
                    size_weight=float(colony_cfg.get("size_loss_weight", 0.1)),
                    offset_weight=float(colony_cfg.get("offset_loss_weight", 1.0)),
                )
            if training:
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        batch_size = len(images)
        n_samples += batch_size
        for key, value in losses.items():
            totals[key] += float(value.item()) * batch_size
        with torch.no_grad():
            decoded = decode_centernet(
                outputs,
                stride=int(colony_cfg.get("output_stride", 4)),
                score_threshold=float(colony_cfg.get("score_threshold", 0.25)),
                nms_iou_threshold=float(colony_cfg.get("tile_nms_iou", 0.30)),
                topk=int(colony_cfg.get("topk", 500)),
            )
            for prediction, target_boxes in zip(decoded, batch["boxes"], strict=False):
                current_tp, current_fp, current_fn = detection_match_counts(
                    prediction["boxes"].cpu(), target_boxes.cpu(), iou_threshold=0.5
                )
                tp += current_tp
                fp += current_fp
                fn += current_fn
    output = {key: value / max(n_samples, 1) for key, value in totals.items()}
    output.update(precision_recall_f1(tp, fp, fn))
    return output


def train_colony_model(config: dict[str, Any]) -> dict[str, str]:
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    output_dir = Path(config["output_dir"])
    metrics_dir = output_dir / "metrics"
    checkpoints_dir = output_dir / "checkpoints"
    tile_manifest_path = materialize_colony_tiles(config, overwrite=False)
    tile_frame = pd.read_csv(tile_manifest_path, dtype={"image_id": "string"})
    colony_cfg = config["colony"]
    tile_size = int(colony_cfg["tile_size"])
    stride = int(colony_cfg.get("output_stride", 4))
    train_frame = tile_frame[tile_frame["split"] == "train"].copy()
    val_frame = tile_frame[tile_frame["split"].isin(["val", "validation"])].copy()
    max_train_tiles = colony_cfg.get("max_train_tiles")
    if max_train_tiles:
        train_frame = train_frame.sample(
            n=min(int(max_train_tiles), len(train_frame)),
            random_state=int(config.get("seed", 42)),
        )

    train_dataset = ColonyTileDataset(
        train_frame, output_dir, tile_size, stride, training=True
    )
    val_dataset = ColonyTileDataset(val_frame, output_dir, tile_size, stride, training=False)
    workers = int(colony_cfg.get("workers", 4))
    loader_extra = _loader_kwargs(config, workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(colony_cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
        collate_fn=collate_colony_batch,
        **loader_extra,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(colony_cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_colony_batch,
        **loader_extra,
    )

    model = ResNet50FPNCenterNet(
        fpn_channels=int(colony_cfg.get("fpn_channels", 128)),
        pretrained=bool(colony_cfg.get("pretrained", True)),
        freeze_stem=bool(colony_cfg.get("freeze_stem", False)),
    ).to(device)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(colony_cfg["learning_rate"]),
        weight_decay=float(colony_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(colony_cfg["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    patience = int(colony_cfg.get("early_stopping_patience", 8))
    best_loss = math.inf
    stale_epochs = 0
    best_path = checkpoints_dir / "resnet50_fpn_centernet_best.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_metrics = _colony_epoch(
            model, train_loader, device, optimizer, scaler, colony_cfg
        )
        with torch.no_grad():
            val_metrics = _colony_epoch(
                model, val_loader, device, None, scaler, colony_cfg
            )
        scheduler.step()
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(metrics_dir / "colony_training_history.csv", index=False)
        val_loss = val_metrics.get("loss", math.inf)
        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            _save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_loss, config)
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    model_info = {
        "model": "ResNet50FPNCenterNet",
        "total_parameters": total_parameter_count(model),
        "trainable_parameters": trainable_parameter_count(model),
        "tile_size": tile_size,
        "output_stride": stride,
        "checkpoint": str(best_path),
        "best_val_loss": best_loss,
        "device": str(device),
        "train_tiles": len(train_dataset),
        "val_tiles": len(val_dataset),
    }
    save_json(output_dir / "metadata" / "colony_model_info.json", model_info)
    return {
        "checkpoint": str(best_path),
        "history": str(metrics_dir / "colony_training_history.csv"),
    }


def _filter_tile_edge_predictions(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    meta: dict[str, Any],
    tile_size: int,
    edge_margin: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(boxes) == 0 or edge_margin <= 0:
        return boxes, scores
    center_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    center_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
    x0, y0 = int(meta["x0"]), int(meta["y0"])
    source_width, source_height = int(meta["source_width"]), int(meta["source_height"])
    keep = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
    if x0 > 0:
        keep &= center_x >= edge_margin
    if y0 > 0:
        keep &= center_y >= edge_margin
    if x0 + tile_size < source_width:
        keep &= center_x < tile_size - edge_margin
    if y0 + tile_size < source_height:
        keep &= center_y < tile_size - edge_margin
    return boxes[keep], scores[keep]


def _draw_colony_overlay(
    image_bgr: np.ndarray,
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    scores: np.ndarray,
    true_count: int,
    pred_count: int,
) -> np.ndarray:
    output = image_bgr.copy()
    for box in ground_truth:
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for box, score in zip(predicted, scores, strict=False):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            output,
            f"{score:.2f}",
            (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(output, (0, 0), (620, 55), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"GT={true_count}  Pred={pred_count}  Error={pred_count - true_count:+d}",
        (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


@torch.no_grad()
def evaluate_colony_model(config: dict[str, Any], split: str = "test") -> dict[str, str]:
    device = resolve_device(str(config.get("device", "auto")))
    output_dir = Path(config["output_dir"])
    metrics_dir = output_dir / "metrics"
    colony_cfg = config["colony"]
    tile_size = int(colony_cfg["tile_size"])
    stride = int(colony_cfg.get("output_stride", 4))
    tile_frame = pd.read_csv(
        output_dir / "metadata" / "tile_manifest.csv", dtype={"image_id": "string"}
    )
    split_aliases = [split, "validation"] if split == "val" else [split]
    eval_frame = tile_frame[tile_frame["split"].isin(split_aliases)].copy()
    max_eval_images = config.get("evaluation", {}).get("max_images")
    if max_eval_images:
        selected_ids = (
            eval_frame[["image_id"]]
            .drop_duplicates()
            .sample(
                n=min(int(max_eval_images), eval_frame["image_id"].nunique()),
                random_state=int(config.get("seed", 42)),
            )["image_id"]
            .tolist()
        )
        eval_frame = eval_frame[eval_frame["image_id"].isin(selected_ids)].copy()
    dataset = ColonyTileDataset(eval_frame, output_dir, tile_size, stride, training=False)
    workers = int(colony_cfg.get("workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=int(colony_cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate_colony_batch,
        **_loader_kwargs(config, workers),
    )
    model = ResNet50FPNCenterNet(
        fpn_channels=int(colony_cfg.get("fpn_channels", 128)),
        pretrained=False,
    ).to(device)
    checkpoint = output_dir / "checkpoints" / "resnet50_fpn_centernet_best.pt"
    load_state_dict_flexible(model, str(checkpoint), map_location=device)
    model.eval()

    tile_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    tile_tp = tile_fp = tile_fn = 0
    global_boxes: dict[str, list[torch.Tensor]] = defaultdict(list)
    global_scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    image_meta: dict[str, dict[str, Any]] = {}
    edge_margin = int(colony_cfg.get("tile_edge_margin", 16))

    for batch in tqdm(loader, desc=f"Evaluasi colony {split}"):
        images = batch["image"].to(device, non_blocking=True)
        outputs = model(images)
        decoded = decode_centernet(
            outputs,
            stride=stride,
            score_threshold=float(colony_cfg.get("score_threshold", 0.25)),
            nms_iou_threshold=float(colony_cfg.get("tile_nms_iou", 0.30)),
            topk=int(colony_cfg.get("topk", 500)),
        )
        predictions_for_map: list[dict[str, torch.Tensor]] = []
        targets_for_map: list[dict[str, torch.Tensor]] = []
        for prediction, target_boxes, meta in zip(
            decoded, batch["boxes"], batch["meta"], strict=False
        ):
            boxes = prediction["boxes"].detach().cpu()
            scores = prediction["scores"].detach().cpu()
            current_tp, current_fp, current_fn = detection_match_counts(
                boxes, target_boxes.cpu(), iou_threshold=0.5
            )
            tile_tp += current_tp
            tile_fp += current_fp
            tile_fn += current_fn
            predictions_for_map.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": torch.ones(len(boxes), dtype=torch.int64),
                }
            )
            targets_for_map.append(
                {
                    "boxes": target_boxes.cpu(),
                    "labels": torch.ones(len(target_boxes), dtype=torch.int64),
                }
            )
            boxes, scores = _filter_tile_edge_predictions(
                boxes, scores, meta, tile_size, edge_margin
            )
            if len(boxes):
                shifted = boxes.clone()
                shifted[:, 0::2] += int(meta["x0"])
                shifted[:, 1::2] += int(meta["y0"])
                global_boxes[str(meta["image_id"])].append(shifted)
                global_scores[str(meta["image_id"])].append(scores)
            image_meta[str(meta["image_id"])] = meta
        tile_map.update(predictions_for_map, targets_for_map)

    tile_map_result = tile_map.compute()
    tile_prf = precision_recall_f1(tile_tp, tile_fp, tile_fn)
    tile_summary = {
        "split": split,
        "tile_map": float(tile_map_result["map"]),
        "tile_map_50": float(tile_map_result["map_50"]),
        "tile_map_75": float(tile_map_result["map_75"]),
        "tile_precision_iou50": tile_prf["precision"],
        "tile_recall_iou50": tile_prf["recall"],
        "tile_f1_iou50": tile_prf["f1"],
        "tile_tp": tile_tp,
        "tile_fp": tile_fp,
        "tile_fn": tile_fn,
    }
    pd.DataFrame([tile_summary]).to_csv(
        metrics_dir / f"colony_{split}_tile_detection_summary.csv", index=False
    )

    annotations = pd.read_csv(
        Path(config["classical_plate_dir"]) / "object_annotations_normalized.csv",
        dtype={"image_id": "string"},
    )
    if "center_inside_counting_mask" in annotations.columns:
        annotations = annotations[
            annotations["center_inside_counting_mask"].fillna(False).astype(bool)
        ]
    if "processing_status" in annotations.columns:
        annotations = annotations[
            annotations["processing_status"].astype(str) == "success"
        ]
    annotations = annotations[annotations["object_type"].astype(str) == "colony"].copy()
    grouped_gt = {key: value for key, value in annotations.groupby("image_id")}
    detections = pd.read_csv(
        Path(config["classical_plate_dir"]) / "plate_detection_strategy_b.csv",
        dtype={"image_id": "string"},
    ).set_index("image_id", drop=False)
    intensity = pd.read_csv(
        Path(config["intensity_dir"]) / "intensity_metrics.csv",
        dtype={"image_id": "string"},
    ).set_index("image_id", drop=False)

    full_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    per_image_rows: list[dict[str, Any]] = []
    visual_count = 0
    visual_dir = output_dir / "visual_samples" / "06_colony_predictions"
    full_tp = full_fp = full_fn = 0

    for image_id in sorted(image_meta):
        meta = image_meta[image_id]
        if global_boxes.get(image_id):
            boxes = torch.cat(global_boxes[image_id], dim=0)
            scores = torch.cat(global_scores[image_id], dim=0)
            keep = nms(
                boxes,
                scores,
                float(colony_cfg.get("global_nms_iou", 0.25)),
            )
            boxes = boxes[keep]
            scores = scores[keep]
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            scores = torch.empty((0,), dtype=torch.float32)

        if image_id in detections.index and len(boxes):
            mask_path = Path(config["classical_plate_dir"]) / str(
                detections.loc[image_id, "normalized_counting_mask_path"]
            )
            counting_mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
            center_x = ((boxes[:, 0] + boxes[:, 2]) / 2).round().long()
            center_y = ((boxes[:, 1] + boxes[:, 3]) / 2).round().long()
            valid = (
                (center_x >= 0)
                & (center_x < counting_mask.shape[1])
                & (center_y >= 0)
                & (center_y < counting_mask.shape[0])
            )
            inside = torch.zeros_like(valid)
            valid_indices = torch.where(valid)[0]
            if len(valid_indices):
                inside_values = counting_mask[
                    center_y[valid_indices].numpy(), center_x[valid_indices].numpy()
                ] > 0
                inside[valid_indices] = torch.from_numpy(inside_values)
            boxes = boxes[inside]
            scores = scores[inside]

        group = grouped_gt.get(image_id)
        if group is None:
            gt_boxes = torch.empty((0, 4), dtype=torch.float32)
        else:
            gt_boxes = torch.tensor(
                np.stack(
                    [
                        group["x_normalized"].to_numpy(np.float32),
                        group["y_normalized"].to_numpy(np.float32),
                        (
                            group["x_normalized"] + group["width_normalized"]
                        ).to_numpy(np.float32),
                        (
                            group["y_normalized"] + group["height_normalized"]
                        ).to_numpy(np.float32),
                    ],
                    axis=1,
                )
            )
        current_tp, current_fp, current_fn = detection_match_counts(
            boxes, gt_boxes, iou_threshold=0.5
        )
        full_tp += current_tp
        full_fp += current_fp
        full_fn += current_fn
        full_map.update(
            [
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": torch.ones(len(boxes), dtype=torch.int64),
                }
            ],
            [
                {
                    "boxes": gt_boxes,
                    "labels": torch.ones(len(gt_boxes), dtype=torch.int64),
                }
            ],
        )
        true_count = int(meta["true_count_metadata"])
        pred_count = int(len(boxes))
        per_image_rows.append(
            {
                "image_id": image_id,
                "split": split,
                "plate_condition": meta["plate_condition"],
                "true_count": true_count,
                "true_count_boxes": int(meta["true_count_boxes"]),
                "pred_count": pred_count,
                "error": pred_count - true_count,
                "absolute_error": abs(pred_count - true_count),
                "relative_error": abs(pred_count - true_count) / max(true_count, 1),
                "tp_iou50": current_tp,
                "fp_iou50": current_fp,
                "fn_iou50": current_fn,
            }
        )
        if visual_count < 5 and image_id in intensity.index:
            image_path = Path(config["intensity_dir"]) / str(
                intensity.loc[image_id, "local_flatfield_output_path"]
            )
            gray = read_image(image_path, cv2.IMREAD_GRAYSCALE)
            image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            overlay = _draw_colony_overlay(
                image_bgr,
                gt_boxes.numpy(),
                boxes.numpy(),
                scores.numpy(),
                true_count,
                pred_count,
            )
            write_image(visual_dir / f"{visual_count + 1:02d}_{image_id}.jpg", overlay)
            visual_count += 1

    per_image = pd.DataFrame(per_image_rows)
    per_image.to_csv(metrics_dir / f"colony_{split}_per_image.csv", index=False)
    count_summary = summarize_count_metrics(per_image)
    count_summary.insert(0, "split", split)
    count_summary.to_csv(metrics_dir / f"colony_{split}_count_summary.csv", index=False)
    full_map_result = full_map.compute()
    full_prf = precision_recall_f1(full_tp, full_fp, full_fn)
    full_detection_summary = pd.DataFrame(
        [
            {
                "split": split,
                "full_map": float(full_map_result["map"]),
                "full_map_50": float(full_map_result["map_50"]),
                "full_map_75": float(full_map_result["map_75"]),
                "precision_iou50": full_prf["precision"],
                "recall_iou50": full_prf["recall"],
                "f1_iou50": full_prf["f1"],
                "tp_iou50": full_tp,
                "fp_iou50": full_fp,
                "fn_iou50": full_fn,
            }
        ]
    )
    full_detection_summary.to_csv(
        metrics_dir / f"colony_{split}_full_detection_summary.csv", index=False
    )
    return {
        "per_image": str(metrics_dir / f"colony_{split}_per_image.csv"),
        "count_summary": str(metrics_dir / f"colony_{split}_count_summary.csv"),
        "detection_summary": str(
            metrics_dir / f"colony_{split}_full_detection_summary.csv"
        ),
    }


def export_models(config: dict[str, Any]) -> dict[str, str]:
    output_dir = Path(config["output_dir"])
    export_dir = output_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    statuses: list[dict[str, Any]] = []

    plate_cfg = config["plate"]
    plate_model = U2NETP().to(device).eval()
    plate_checkpoint = output_dir / "checkpoints" / "u2netp_plate_best.pt"
    load_state_dict_flexible(plate_model, str(plate_checkpoint), map_location=device)
    plate_wrapper = U2NetProductionWrapper(plate_model).eval()
    plate_example = torch.randn(
        1, 3, int(plate_cfg["image_size"]), int(plate_cfg["image_size"])
    )
    plate_trace = torch.jit.trace(plate_wrapper, plate_example, strict=True)
    plate_torchscript = export_dir / "PlateU2NetP.torchscript.pt"
    plate_trace.save(str(plate_torchscript))
    statuses.append(
        {
            "artifact": "PlateU2NetP.torchscript.pt",
            "format": "TorchScript",
            "status": "success",
            "message": "",
        }
    )

    colony_cfg = config["colony"]
    colony_model = ResNet50FPNCenterNet(
        fpn_channels=int(colony_cfg.get("fpn_channels", 128)), pretrained=False
    ).to(device).eval()
    colony_checkpoint = output_dir / "checkpoints" / "resnet50_fpn_centernet_best.pt"
    load_state_dict_flexible(colony_model, str(colony_checkpoint), map_location=device)
    colony_wrapper = ColonyProductionWrapper(colony_model).eval()
    colony_example = torch.randn(
        1, 3, int(colony_cfg["tile_size"]), int(colony_cfg["tile_size"])
    )
    colony_trace = torch.jit.trace(colony_wrapper, colony_example, strict=True)
    colony_torchscript = export_dir / "ColonyResNet50FPN.torchscript.pt"
    colony_trace.save(str(colony_torchscript))
    statuses.append(
        {
            "artifact": "ColonyResNet50FPN.torchscript.pt",
            "format": "TorchScript",
            "status": "success",
            "message": "",
        }
    )

    try:
        import coremltools as ct

        target_name = str(config.get("coreml", {}).get("minimum_target", "macOS13"))
        target = getattr(ct.target, target_name)
        plate_mlmodel = ct.convert(
            plate_trace,
            inputs=[
                ct.TensorType(
                    name="image",
                    shape=plate_example.shape,
                    dtype=np.float32,
                )
            ],
            outputs=[ct.TensorType(name="plate_probability")],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=target,
        )
        plate_coreml = export_dir / "PlateU2NetP.mlpackage"
        plate_mlmodel.save(str(plate_coreml))
        statuses.append(
            {
                "artifact": "PlateU2NetP.mlpackage",
                "format": "Core ML ML Program FP16",
                "status": "success",
                "message": "",
            }
        )

        colony_mlmodel = ct.convert(
            colony_trace,
            inputs=[
                ct.TensorType(
                    name="image",
                    shape=colony_example.shape,
                    dtype=np.float32,
                )
            ],
            outputs=[
                ct.TensorType(name="heatmap"),
                ct.TensorType(name="size"),
                ct.TensorType(name="offset"),
            ],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=target,
        )
        colony_coreml = export_dir / "ColonyResNet50FPN.mlpackage"
        colony_mlmodel.save(str(colony_coreml))
        statuses.append(
            {
                "artifact": "ColonyResNet50FPN.mlpackage",
                "format": "Core ML ML Program FP16",
                "status": "success",
                "message": "Decoder dan NMS dijalankan di Swift.",
            }
        )
    except Exception as exc:
        statuses.append(
            {
                "artifact": "Core ML packages",
                "format": "Core ML",
                "status": "failed",
                "message": str(exc),
            }
        )

    status_path = export_dir / "export_status.csv"
    pd.DataFrame(statuses).to_csv(status_path, index=False)
    save_json(
        export_dir / "production_parameters.json",
        {
            "plate_input_size": int(plate_cfg["image_size"]),
            "plate_threshold": float(plate_cfg.get("threshold", 0.5)),
            "colony_tile_size": int(colony_cfg["tile_size"]),
            "colony_tile_overlap": int(colony_cfg["tile_overlap"]),
            "colony_output_stride": int(colony_cfg.get("output_stride", 4)),
            "colony_score_threshold": float(colony_cfg.get("score_threshold", 0.25)),
            "tile_nms_iou": float(colony_cfg.get("tile_nms_iou", 0.30)),
            "global_nms_iou": float(colony_cfg.get("global_nms_iou", 0.25)),
            "tile_edge_margin": int(colony_cfg.get("tile_edge_margin", 16)),
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
            "plate_roi_target_size": int(
                config.get("classical_plate", {}).get("target_size", 2048)
            ),
            "plate_crop_padding_ratio": float(
                config.get("classical_plate", {}).get("crop_padding_ratio", 1.04)
            ),
            "plate_counting_scale": float(
                config.get("classical_plate", {}).get("counting_scale", 1.0)
            ),
            "flatfield_sigma_ratio": float(
                config.get("intensity", {}).get("flatfield_sigma_ratio", 0.04)
            ),
            "safe_erode_ratio": float(
                config.get("intensity", {}).get("safe_erode_ratio", 0.025)
            ),
            "safe_minimum_fraction": float(
                config.get("intensity", {}).get("safe_minimum_fraction", 0.15)
            ),
            "flatfield_percentile_low": float(
                config.get("intensity", {}).get("percentile_low", 1.0)
            ),
            "flatfield_percentile_high": float(
                config.get("intensity", {}).get("percentile_high", 99.0)
            ),
        },
    )
    return {
        "plate_torchscript": str(plate_torchscript),
        "colony_torchscript": str(colony_torchscript),
        "status": str(status_path),
    }


def compile_all_metrics(config: dict[str, Any]) -> Path:
    metrics_dir = Path(config["output_dir"]) / "metrics"
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(metrics_dir.glob("*_summary.csv")):
        frame = pd.read_csv(csv_path)
        for _, row in frame.iterrows():
            for key, value in row.items():
                if key in {"metric", "split", "n"}:
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)) and pd.notna(value):
                    rows.append(
                        {
                            "source_file": csv_path.name,
                            "split": row.get("split", ""),
                            "metric_group": row.get("metric", "summary"),
                            "metric": key,
                            "value": float(value),
                        }
                    )
    output_path = metrics_dir / "all_evaluation_metrics.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
