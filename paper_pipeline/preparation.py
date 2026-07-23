from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .components import extract_colony_components
from .config import PaperConfig
from .io_utils import binary_mask, list_images, read_image, save_image
from .preprocessing import make_edge_ring_label


def _metadata_map(metadata_csv: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_csv, dtype={"image_id": "string"})
    required = {"image_id", "split", "background"}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Metadata kehilangan kolom: {sorted(missing)}")
    return metadata.drop_duplicates("image_id").set_index("image_id")


def prepare_edge_dataset(
    cfg: PaperConfig,
    output_dir: Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Siapkan gambar penuh dan proxy label ring; tidak membuat patch."""
    destination = (output_dir or (cfg.data_root / "edge")).resolve()
    metadata = _metadata_map(cfg.metadata_csv)
    images = list_images(cfg.preprocessed_dir)
    if limit is not None:
        images = images[:limit]

    records: list[dict[str, object]] = []
    for image_path in tqdm(images, desc="Menyiapkan edge dataset"):
        image_id = image_path.stem
        if image_id not in metadata.index:
            continue
        row = metadata.loc[image_id]
        split = str(row["split"])
        background = str(row["background"])
        relative = Path(background) / f"{image_id}.png"
        plate_mask_path = cfg.counting_mask_dir / relative
        if not plate_mask_path.exists():
            raise FileNotFoundError(f"Counting mask tidak ada: {plate_mask_path}")

        image = read_image(image_path, cv2.IMREAD_GRAYSCALE)
        plate_mask = binary_mask(
            read_image(plate_mask_path, cv2.IMREAD_GRAYSCALE)
        )
        if plate_mask.shape != image.shape:
            raise ValueError(
                f"Shape tidak cocok untuk {image_id}: "
                f"{image.shape} != {plate_mask.shape}"
            )
        edge_label = make_edge_ring_label(
            plate_mask,
            ring_width=cfg.edge_ring_width,
        )

        output_image = (
            destination / split / "images" / background / f"{image_id}.png"
        )
        output_mask = (
            destination / split / "masks" / background / f"{image_id}.png"
        )
        save_image(output_image, image)
        save_image(output_mask, edge_label)
        records.append(
            {
                "image_id": image_id,
                "split": split,
                "background": background,
                "image_path": str(output_image.resolve()),
                "mask_path": str(output_mask.resolve()),
                "label_source": "proxy_ring_from_counting_mask",
                "edge_ring_width": cfg.edge_ring_width,
                "foreground_fraction": float((edge_label > 0).mean()),
            }
        )

    manifest = pd.DataFrame(records)
    destination.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination / "manifest.csv", index=False)
    return manifest


def validate_segmentation_dataset(
    root: Path,
) -> dict[str, object]:
    """Audit dataset manual edge/colony tanpa mengubah isinya."""
    report: dict[str, object] = {"root": str(root), "splits": {}}
    total_images = 0
    total_masks = 0
    missing_masks: list[str] = []
    for split in ("train", "validation"):
        image_root = root / split / "images"
        mask_root = root / split / "masks"
        images = list_images(image_root) if image_root.exists() else []
        masks = list_images(mask_root) if mask_root.exists() else []
        mask_by_relative = {
            path.relative_to(mask_root).with_suffix("").as_posix(): path
            for path in masks
        }
        for image_path in images:
            key = (
                image_path.relative_to(image_root)
                .with_suffix("")
                .as_posix()
            )
            if key not in mask_by_relative:
                missing_masks.append(f"{split}:{key}")
        report["splits"][split] = {
            "images": len(images),
            "masks": len(masks),
        }
        total_images += len(images)
        total_masks += len(masks)
    report["total_images"] = total_images
    report["total_masks"] = total_masks
    report["missing_masks"] = missing_masks
    report["ready"] = (
        total_images > 0
        and total_images == total_masks
        and not missing_masks
    )
    return report


def build_segmentation_manifest(
    root: Path,
    output_path: Path | None = None,
) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for split in ("train", "validation"):
        image_root = root / split / "images"
        mask_root = root / split / "masks"
        for image_path in list_images(image_root):
            relative = image_path.relative_to(image_root)
            candidates = [
                mask_root / relative.with_suffix(suffix)
                for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            ]
            mask_path = next((path for path in candidates if path.exists()), None)
            if mask_path is None:
                raise FileNotFoundError(
                    f"Mask pasangan tidak ditemukan: {split}/{relative}"
                )
            records.append(
                {
                    "image_id": image_path.stem,
                    "split": split,
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                }
            )
    manifest = pd.DataFrame(records)
    destination = output_path or (root / "manifest.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination, index=False)
    return manifest


def prepare_resnet_dataset(
    cfg: PaperConfig,
    colony_mask_dir: Path,
    output_dir: Path | None = None,
    min_component_area: int | None = None,
) -> pd.DataFrame:
    """Buat crop CC dan label 0–9 dari pusat bounding box AGAR.

    Paper menggunakan anotasi jumlah secara manual. Di AGAR, jumlah pada CC
    diproksikan dengan banyaknya pusat bounding box yang jatuh di komponen.
    Hanya gambar yang benar-benar memiliki anotasi yang diproses.
    """
    destination = (output_dir or (cfg.data_root / "resnet")).resolve()
    metadata = _metadata_map(cfg.metadata_csv)
    annotations = pd.read_csv(
        cfg.normalized_annotations_csv,
        dtype={"image_id": "string"},
    )
    required = {
        "image_id",
        "x_normalized",
        "y_normalized",
        "width_normalized",
        "height_normalized",
    }
    missing = required.difference(annotations.columns)
    if missing:
        raise KeyError(f"Anotasi kehilangan kolom: {sorted(missing)}")
    if "center_inside_counting_mask" in annotations:
        annotations = annotations[
            annotations["center_inside_counting_mask"].fillna(False)
        ].copy()
    annotated_ids = set(annotations["image_id"].astype(str))

    masks = list_images(colony_mask_dir)
    records: list[dict[str, object]] = []
    area_limit = (
        cfg.min_component_area
        if min_component_area is None
        else min_component_area
    )

    for mask_path in tqdm(masks, desc="Menyiapkan dataset ResNet50"):
        image_id = mask_path.stem
        if image_id not in annotated_ids or image_id not in metadata.index:
            continue
        meta = metadata.loc[image_id]
        background = str(meta["background"])
        split = str(meta["split"])
        image_path = (
            cfg.preprocessed_dir / background / f"{image_id}.png"
        )
        if not image_path.exists():
            raise FileNotFoundError(f"Input sigma040 tidak ada: {image_path}")

        image = read_image(image_path, cv2.IMREAD_GRAYSCALE)
        mask = binary_mask(read_image(mask_path, cv2.IMREAD_GRAYSCALE))
        if image.shape != mask.shape:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        components, label_map = extract_colony_components(
            image=image,
            colony_mask=mask,
            target_size=cfg.component_size,
            min_area=area_limit,
        )

        image_annotations = annotations[annotations["image_id"] == image_id]
        centers: list[tuple[int, int]] = []
        for _, annotation in image_annotations.iterrows():
            cx = float(annotation["x_normalized"]) + 0.5 * float(
                annotation["width_normalized"]
            )
            cy = float(annotation["y_normalized"]) + 0.5 * float(
                annotation["height_normalized"]
            )
            centers.append((int(round(cx)), int(round(cy))))

        for component in components:
            count = 0
            for cx, cy in centers:
                if (
                    0 <= cy < label_map.shape[0]
                    and 0 <= cx < label_map.shape[1]
                    and int(label_map[cy, cx]) == component.source_label
                ):
                    count += 1
            class_label = min(count, 9)
            crop_name = f"{image_id}_cc{component.component_id:04d}.png"
            crop_path = (
                destination
                / split
                / "images"
                / background
                / crop_name
            )
            save_image(crop_path, component.crop)
            records.append(
                {
                    "crop_id": crop_path.stem,
                    "image_id": image_id,
                    "split": split,
                    "background": background,
                    "crop_path": str(crop_path),
                    "label": class_label,
                    "label_semantics": "9_means_9_or_more",
                    "label_source": "agar_box_centers_inside_component",
                    "component_area": component.area,
                    "bbox_x": component.bbox_x,
                    "bbox_y": component.bbox_y,
                    "bbox_width": component.bbox_width,
                    "bbox_height": component.bbox_height,
                    "rotation_degrees": component.rotation_degrees,
                }
            )

    manifest = pd.DataFrame(records)
    destination.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination / "manifest.csv", index=False)
    return manifest
