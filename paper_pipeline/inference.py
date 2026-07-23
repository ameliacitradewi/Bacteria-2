from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .components import extract_colony_components
from .config import PaperConfig
from .io_utils import as_three_channel, list_images, read_image, save_image
from .models import U2Net, build_resnet50_counter, load_checkpoint
from .preprocessing import inner_roi_from_edge_mask
from .training import select_device


def _image_tensor(
    image: np.ndarray,
    device: torch.device,
    resize_long_side: int | None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    bgr = as_three_channel(image)
    original_shape = bgr.shape[:2]
    if resize_long_side:
        scale = resize_long_side / float(max(original_shape))
        new_size = (
            max(1, int(round(original_shape[1] * scale))),
            max(1, int(round(original_shape[0] * scale))),
        )
        bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(
        rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    )[None, :, :, :]
    return tensor.to(device), original_shape


def predict_u2net(
    model: U2Net,
    image: np.ndarray,
    device: torch.device,
    resize_long_side: int | None = None,
) -> np.ndarray:
    tensor, original_shape = _image_tensor(
        image,
        device,
        resize_long_side,
    )
    with torch.no_grad():
        logits = model(tensor)[0]
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    if probability.shape != original_shape:
        probability = cv2.resize(
            probability,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.clip(probability, 0.0, 1.0).astype(np.float32)


def _predict_component_counts(
    model: torch.nn.Module,
    crops: list[np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    confidences: list[float] = []
    for start in range(0, len(crops), batch_size):
        batch = crops[start : start + batch_size]
        tensors = []
        for crop in batch:
            rgb = cv2.cvtColor(as_three_channel(crop), cv2.COLOR_BGR2RGB)
            tensors.append(
                torch.from_numpy(
                    rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
                )
            )
        inputs = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(inputs)
            probabilities = F.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
        labels.extend(int(value) for value in prediction.cpu().tolist())
        confidences.extend(float(value) for value in confidence.cpu().tolist())
    return labels, confidences


def _save_probability(path: Path, probability: np.ndarray) -> None:
    save_image(path, np.rint(np.clip(probability, 0.0, 1.0) * 255).astype(np.uint8))


def _component_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    component_rows: list[dict[str, object]],
) -> np.ndarray:
    overlay = as_three_channel(image).copy()
    for row in component_rows:
        source_label = int(row["source_label"])
        binary = np.where(labels == source_label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        x = int(row["bbox_x"])
        y = int(row["bbox_y"])
        text = f"{row['predicted_count']} ({float(row['confidence']):.2f})"
        cv2.putText(
            overlay,
            text,
            (x, max(y - 4, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return overlay


def run_full_inference(
    cfg: PaperConfig,
    edge_weights: Path,
    colony_weights: Path,
    resnet_weights: Path,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    device_name: str = "auto",
    resize_long_side: int | None = None,
    limit: int | None = None,
    min_component_area: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for name, path in {
        "edge U2-Net": edge_weights,
        "colony U2-Net": colony_weights,
        "ResNet50": resnet_weights,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"Weight {name} tidak ditemukan: {path}")

    device = select_device(device_name)
    edge_model = U2Net().to(device)
    colony_model = U2Net().to(device)
    counter_model = build_resnet50_counter(num_classes=10).to(device)
    load_checkpoint(edge_model, str(edge_weights), device)
    load_checkpoint(colony_model, str(colony_weights), device)
    load_checkpoint(counter_model, str(resnet_weights), device)
    edge_model.eval()
    colony_model.eval()
    counter_model.eval()

    source = (input_dir or cfg.preprocessed_dir).resolve()
    destination = (output_dir or cfg.output_root).resolve()
    images = list_images(source)
    if limit is not None:
        images = images[:limit]
    area_limit = (
        cfg.min_component_area
        if min_component_area is None
        else min_component_area
    )

    count_records: list[dict[str, object]] = []
    component_records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for image_path in tqdm(images, desc="Paper pipeline inference"):
        image_id = image_path.stem
        relative_parent = image_path.relative_to(source).parent
        try:
            image = read_image(image_path, cv2.IMREAD_UNCHANGED)
            three_channel = as_three_channel(image)

            edge_probability = predict_u2net(
                edge_model,
                three_channel,
                device,
                resize_long_side,
            )
            edge_mask = np.where(
                edge_probability >= cfg.edge_threshold,
                255,
                0,
            ).astype(np.uint8)
            roi_mask = inner_roi_from_edge_mask(edge_mask)
            if int((roi_mask > 0).sum()) < 100:
                raise RuntimeError(
                    "U2-Net edge tidak menghasilkan ROI dalam yang valid."
                )
            roi_image = three_channel.copy()
            roi_image[roi_mask == 0] = 0

            colony_probability = predict_u2net(
                colony_model,
                roi_image,
                device,
                resize_long_side,
            )
            colony_probability[roi_mask == 0] = 0.0
            colony_mask = np.where(
                colony_probability >= cfg.colony_threshold,
                255,
                0,
            ).astype(np.uint8)

            components, label_map = extract_colony_components(
                image=roi_image,
                colony_mask=colony_mask,
                target_size=cfg.component_size,
                min_area=area_limit,
            )
            predicted_counts, confidences = _predict_component_counts(
                counter_model,
                [component.crop for component in components],
                device,
            ) if components else ([], [])

            image_component_rows: list[dict[str, object]] = []
            for component, predicted_count, confidence in zip(
                components,
                predicted_counts,
                confidences,
            ):
                crop_path = (
                    destination
                    / "components"
                    / relative_parent
                    / image_id
                    / f"{image_id}_cc{component.component_id:04d}.png"
                )
                save_image(crop_path, component.crop)
                row: dict[str, object] = {
                    "image_id": image_id,
                    "component_id": component.component_id,
                    "source_label": component.source_label,
                    "predicted_count": predicted_count,
                    "confidence": confidence,
                    "class_9_is_lower_bound": predicted_count == 9,
                    "component_area": component.area,
                    "bbox_x": component.bbox_x,
                    "bbox_y": component.bbox_y,
                    "bbox_width": component.bbox_width,
                    "bbox_height": component.bbox_height,
                    "rotation_degrees": component.rotation_degrees,
                    "crop_path": str(crop_path),
                }
                component_records.append(row)
                image_component_rows.append(row)

            base = relative_parent / f"{image_id}.png"
            _save_probability(
                destination / "edge_probability" / base,
                edge_probability,
            )
            save_image(destination / "edge_mask" / base, edge_mask)
            save_image(destination / "roi_mask" / base, roi_mask)
            save_image(destination / "roi_image" / base, roi_image)
            _save_probability(
                destination / "colony_probability" / base,
                colony_probability,
            )
            save_image(destination / "colony_mask" / base, colony_mask)
            save_image(
                destination / "component_overlay" / base,
                _component_overlay(
                    roi_image,
                    label_map,
                    image_component_rows,
                ),
            )

            total_count = int(sum(predicted_counts))
            has_class_9 = any(count == 9 for count in predicted_counts)
            count_records.append(
                {
                    "image_id": image_id,
                    "source_path": str(image_path),
                    "n_components": len(components),
                    "predicted_total": total_count,
                    "total_is_lower_bound": has_class_9,
                    "edge_threshold": cfg.edge_threshold,
                    "colony_threshold": cfg.colony_threshold,
                    "min_component_area": area_limit,
                    "status": "success",
                    "error": "",
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "image_id": image_id,
                    "source_path": str(image_path),
                    "error": repr(exc),
                }
            )
            count_records.append(
                {
                    "image_id": image_id,
                    "source_path": str(image_path),
                    "n_components": 0,
                    "predicted_total": 0,
                    "total_is_lower_bound": False,
                    "edge_threshold": cfg.edge_threshold,
                    "colony_threshold": cfg.colony_threshold,
                    "min_component_area": area_limit,
                    "status": "failed",
                    "error": repr(exc),
                }
            )

    destination.mkdir(parents=True, exist_ok=True)
    counts = pd.DataFrame(count_records)
    component_frame = pd.DataFrame(component_records)
    counts.to_csv(destination / "counts_per_image.csv", index=False)
    component_frame.to_csv(
        destination / "counts_per_component.csv",
        index=False,
    )
    if failures:
        pd.DataFrame(failures).to_csv(
            destination / "failed_images.csv",
            index=False,
        )
    return counts, component_frame


def run_segmentation_inference(
    cfg: PaperConfig,
    edge_weights: Path,
    colony_weights: Path,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    device_name: str = "auto",
    resize_long_side: int | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Jalankan dua U2-Net sampai colony mask, tanpa ResNet50.

    Output `colony_mask/` dapat digunakan oleh prepare-resnet sehingga tidak
    terjadi ketergantungan melingkar sebelum ResNet50 tersedia.
    """
    for name, path in {
        "edge U2-Net": edge_weights,
        "colony U2-Net": colony_weights,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"Weight {name} tidak ditemukan: {path}")

    device = select_device(device_name)
    edge_model = U2Net().to(device)
    colony_model = U2Net().to(device)
    load_checkpoint(edge_model, str(edge_weights), device)
    load_checkpoint(colony_model, str(colony_weights), device)
    edge_model.eval()
    colony_model.eval()

    source = (input_dir or cfg.preprocessed_dir).resolve()
    destination = (
        output_dir or (cfg.output_root / "segmentation")
    ).resolve()
    images = list_images(source)
    if limit is not None:
        images = images[:limit]
    records: list[dict[str, object]] = []

    for image_path in tqdm(images, desc="Two-stage U2-Net inference"):
        image_id = image_path.stem
        relative_parent = image_path.relative_to(source).parent
        base = relative_parent / f"{image_id}.png"
        try:
            image = as_three_channel(
                read_image(image_path, cv2.IMREAD_UNCHANGED)
            )
            edge_probability = predict_u2net(
                edge_model,
                image,
                device,
                resize_long_side,
            )
            edge_mask = np.where(
                edge_probability >= cfg.edge_threshold,
                255,
                0,
            ).astype(np.uint8)
            roi_mask = inner_roi_from_edge_mask(edge_mask)
            if int((roi_mask > 0).sum()) < 100:
                raise RuntimeError("Edge U2-Net tidak menghasilkan ROI valid.")
            roi_image = image.copy()
            roi_image[roi_mask == 0] = 0
            colony_probability = predict_u2net(
                colony_model,
                roi_image,
                device,
                resize_long_side,
            )
            colony_probability[roi_mask == 0] = 0.0
            colony_mask = np.where(
                colony_probability >= cfg.colony_threshold,
                255,
                0,
            ).astype(np.uint8)

            _save_probability(
                destination / "edge_probability" / base,
                edge_probability,
            )
            save_image(destination / "edge_mask" / base, edge_mask)
            save_image(destination / "roi_mask" / base, roi_mask)
            save_image(destination / "roi_image" / base, roi_image)
            _save_probability(
                destination / "colony_probability" / base,
                colony_probability,
            )
            save_image(destination / "colony_mask" / base, colony_mask)
            records.append(
                {
                    "image_id": image_id,
                    "source_path": str(image_path),
                    "colony_mask_path": str(
                        destination / "colony_mask" / base
                    ),
                    "status": "success",
                    "error": "",
                }
            )
        except Exception as exc:
            records.append(
                {
                    "image_id": image_id,
                    "source_path": str(image_path),
                    "colony_mask_path": "",
                    "status": "failed",
                    "error": repr(exc),
                }
            )

    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(destination / "segmentation_manifest.csv", index=False)
    return frame
