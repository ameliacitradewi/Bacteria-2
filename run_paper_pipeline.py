from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from paper_pipeline.config import PaperConfig
from paper_pipeline.io_utils import list_images, read_image, save_image
from paper_pipeline.preparation import (
    build_segmentation_manifest,
    prepare_edge_dataset,
    prepare_resnet_dataset,
    validate_segmentation_dataset,
)
from paper_pipeline.preprocessing import preprocess_raw_image


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline adaptasi paper: preprocessing -> U2-Net edge -> "
            "U2-Net colony -> connected components -> ResNet50 counting."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="Audit kesiapan data, dependensi, dan checkpoint.",
    )
    doctor.set_defaults(handler=_doctor)

    prepare_edge = subparsers.add_parser(
        "prepare-edge",
        help="Siapkan full-image edge dataset tanpa patch.",
    )
    prepare_edge.add_argument("--output-dir", type=Path, default=None)
    prepare_edge.add_argument("--limit", type=int, default=None)
    prepare_edge.set_defaults(handler=_prepare_edge)

    colony_manifest = subparsers.add_parser(
        "build-colony-manifest",
        help="Validasi pasangan image-mask koloni manual dan buat manifest.",
    )
    colony_manifest.add_argument("--data-dir", type=Path, default=None)
    colony_manifest.set_defaults(handler=_build_colony_manifest)

    prepare_resnet = subparsers.add_parser(
        "prepare-resnet",
        help="Ekstrak/rotasi CC 128x128 dan proksikan label dari AGAR boxes.",
    )
    prepare_resnet.add_argument(
        "--colony-mask-dir",
        type=Path,
        required=True,
    )
    prepare_resnet.add_argument("--output-dir", type=Path, default=None)
    prepare_resnet.add_argument("--min-component-area", type=int, default=1)
    prepare_resnet.set_defaults(handler=_prepare_resnet)

    segment = subparsers.add_parser(
        "segment",
        help="Jalankan dua U2-Net sampai colony mask, sebelum ResNet50.",
    )
    segment.add_argument("--edge-weights", type=Path, default=None)
    segment.add_argument("--colony-weights", type=Path, default=None)
    segment.add_argument("--input-dir", type=Path, default=None)
    segment.add_argument("--output-dir", type=Path, default=None)
    segment.add_argument("--device", default="auto")
    segment.add_argument("--resize-long-side", type=int, default=None)
    segment.add_argument("--limit", type=int, default=None)
    segment.set_defaults(handler=_segment)

    train_u2net = subparsers.add_parser(
        "train-u2net",
        help="Latih U2-Net edge atau colony dengan parameter paper.",
    )
    train_u2net.add_argument("--task", choices=("edge", "colony"), required=True)
    train_u2net.add_argument("--manifest", type=Path, default=None)
    train_u2net.add_argument("--epochs", type=int, default=200)
    train_u2net.add_argument("--resize-long-side", type=int, default=None)
    train_u2net.add_argument("--device", default="auto")
    train_u2net.add_argument("--num-workers", type=int, default=0)
    train_u2net.set_defaults(handler=_train_u2net)

    train_resnet = subparsers.add_parser(
        "train-resnet",
        help="Latih ResNet50 kelas 0-9 dengan parameter paper.",
    )
    train_resnet.add_argument("--manifest", type=Path, default=None)
    train_resnet.add_argument("--epochs", type=int, default=200)
    train_resnet.add_argument("--batch-size", type=int, default=32)
    train_resnet.add_argument("--device", default="auto")
    train_resnet.add_argument("--num-workers", type=int, default=0)
    train_resnet.set_defaults(handler=_train_resnet)

    inference = subparsers.add_parser(
        "infer",
        help="Jalankan dua U2-Net, CC, rotasi, dan ResNet50 end-to-end.",
    )
    inference.add_argument("--edge-weights", type=Path, default=None)
    inference.add_argument("--colony-weights", type=Path, default=None)
    inference.add_argument("--resnet-weights", type=Path, default=None)
    inference.add_argument("--input-dir", type=Path, default=None)
    inference.add_argument("--output-dir", type=Path, default=None)
    inference.add_argument("--device", default="auto")
    inference.add_argument("--resize-long-side", type=int, default=None)
    inference.add_argument("--limit", type=int, default=None)
    inference.add_argument("--min-component-area", type=int, default=1)
    inference.set_defaults(handler=_infer)

    preprocess = subparsers.add_parser(
        "preprocess-reference",
        help=(
            "Jalankan aproksimasi preprocessing paper pada folder gambar. "
            "Mode riset utama proyek tetap memakai hasil sigma040."
        ),
    )
    preprocess.add_argument("--input-dir", type=Path, required=True)
    preprocess.add_argument("--output-dir", type=Path, required=True)
    preprocess.add_argument("--known-mask-dir", type=Path, default=None)
    preprocess.add_argument("--limit", type=int, default=None)
    preprocess.set_defaults(handler=_preprocess_reference)
    return parser


def _config(args: argparse.Namespace) -> PaperConfig:
    return PaperConfig.from_project_root(args.project_root)


def _doctor(args: argparse.Namespace) -> None:
    cfg = _config(args)
    try:
        import torch
        import torchvision

        torch_status = {
            "installed": True,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ),
        }
    except Exception as exc:
        torch_status = {"installed": False, "error": repr(exc)}

    edge_report = validate_segmentation_dataset(cfg.data_root / "edge")
    colony_report = validate_segmentation_dataset(cfg.data_root / "colony")
    resnet_manifest = cfg.data_root / "resnet" / "manifest.csv"
    resnet_rows = 0
    if resnet_manifest.exists():
        resnet_rows = len(pd.read_csv(resnet_manifest))
    report = {
        "branch_target": "testpipelinepaper",
        "project_root": str(cfg.project_root),
        "preprocessed_sigma040": {
            "path": str(cfg.preprocessed_dir),
            "exists": cfg.preprocessed_dir.exists(),
            "n_images": (
                len(list_images(cfg.preprocessed_dir))
                if cfg.preprocessed_dir.exists()
                else 0
            ),
        },
        "dependencies": torch_status,
        "edge_dataset": edge_report,
        "colony_dataset": colony_report,
        "resnet_dataset": {
            "manifest": str(resnet_manifest),
            "exists": resnet_manifest.exists(),
            "rows": resnet_rows,
        },
        "checkpoints": {
            "edge": (
                cfg.checkpoint_root / "u2net_edge_best.pt"
            ).exists(),
            "colony": (
                cfg.checkpoint_root / "u2net_colony_best.pt"
            ).exists(),
            "resnet50": (
                cfg.checkpoint_root / "resnet50_count_best.pt"
            ).exists(),
        },
        "scientific_limitations": [
            (
                "Edge labels generated by prepare-edge are proxy rings; "
                "paper used manually annotated rings."
            ),
            (
                "AGAR bounding boxes are not pixel-level colony masks; "
                "manual colony masks are required for faithful U2-Net training."
            ),
            (
                "ResNet labels from prepare-resnet are box-center proxies; "
                "paper manually labeled each connected component."
            ),
        ],
    }
    print(json.dumps(report, indent=2))


def _prepare_edge(args: argparse.Namespace) -> None:
    cfg = _config(args)
    frame = prepare_edge_dataset(
        cfg,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print(f"Edge images prepared: {len(frame)}")
    print(frame.groupby("split").size().to_string())


def _build_colony_manifest(args: argparse.Namespace) -> None:
    cfg = _config(args)
    root = (args.data_dir or (cfg.data_root / "colony")).resolve()
    report = validate_segmentation_dataset(root)
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise RuntimeError(
            "Dataset colony belum siap. Isi train/validation images dan "
            "mask piksel dengan nama relatif yang sama."
        )
    manifest = build_segmentation_manifest(root)
    print(f"Manifest colony: {root / 'manifest.csv'} ({len(manifest)} rows)")


def _prepare_resnet(args: argparse.Namespace) -> None:
    cfg = _config(args)
    frame = prepare_resnet_dataset(
        cfg,
        colony_mask_dir=args.colony_mask_dir.resolve(),
        output_dir=args.output_dir,
        min_component_area=args.min_component_area,
    )
    print(f"ResNet crops prepared: {len(frame)}")
    if not frame.empty:
        print(frame.groupby(["split", "label"]).size().to_string())


def _train_u2net(args: argparse.Namespace) -> None:
    from paper_pipeline.training import train_u2net

    cfg = _config(args)
    manifest = args.manifest
    if manifest is None:
        manifest = cfg.data_root / args.task / "manifest.csv"
    best = train_u2net(
        manifest_path=manifest.resolve(),
        checkpoint_dir=cfg.checkpoint_root,
        task=args.task,
        epochs=args.epochs,
        resize_long_side=args.resize_long_side,
        device_name=args.device,
        num_workers=args.num_workers,
    )
    print(f"Best checkpoint: {best}")


def _train_resnet(args: argparse.Namespace) -> None:
    from paper_pipeline.training import train_resnet50

    cfg = _config(args)
    manifest = args.manifest or (
        cfg.data_root / "resnet" / "manifest.csv"
    )
    best = train_resnet50(
        manifest_path=manifest.resolve(),
        checkpoint_dir=cfg.checkpoint_root,
        epochs=args.epochs,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Best checkpoint: {best}")


def _segment(args: argparse.Namespace) -> None:
    from paper_pipeline.inference import run_segmentation_inference

    cfg = _config(args)
    frame = run_segmentation_inference(
        cfg=cfg,
        edge_weights=(
            args.edge_weights
            or (cfg.checkpoint_root / "u2net_edge_best.pt")
        ).resolve(),
        colony_weights=(
            args.colony_weights
            or (cfg.checkpoint_root / "u2net_colony_best.pt")
        ).resolve(),
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        resize_long_side=args.resize_long_side,
        limit=args.limit,
    )
    print(frame.groupby("status").size().to_string())


def _infer(args: argparse.Namespace) -> None:
    from paper_pipeline.inference import run_full_inference

    cfg = _config(args)
    counts, components = run_full_inference(
        cfg=cfg,
        edge_weights=(
            args.edge_weights
            or (cfg.checkpoint_root / "u2net_edge_best.pt")
        ).resolve(),
        colony_weights=(
            args.colony_weights
            or (cfg.checkpoint_root / "u2net_colony_best.pt")
        ).resolve(),
        resnet_weights=(
            args.resnet_weights
            or (cfg.checkpoint_root / "resnet50_count_best.pt")
        ).resolve(),
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        resize_long_side=args.resize_long_side,
        limit=args.limit,
        min_component_area=args.min_component_area,
    )
    print(counts.to_string(index=False))
    print(f"Components: {len(components)}")


def _preprocess_reference(args: argparse.Namespace) -> None:
    source = args.input_dir.resolve()
    destination = args.output_dir.resolve()
    images = list_images(source)
    if args.limit is not None:
        images = images[: args.limit]
    records: list[dict[str, object]] = []
    for image_path in tqdm(images, desc="Reference preprocessing"):
        relative = image_path.relative_to(source).with_suffix(".png")
        image = read_image(image_path, cv2.IMREAD_COLOR)
        known_mask = None
        if args.known_mask_dir is not None:
            mask_path = args.known_mask_dir.resolve() / relative
            if not mask_path.exists():
                raise FileNotFoundError(f"Known mask tidak ada: {mask_path}")
            known_mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
            if known_mask.shape != image.shape[:2]:
                raise ValueError(
                    f"Mask/image shape berbeda untuk {relative}: "
                    f"{known_mask.shape} != {image.shape[:2]}"
                )
        result = preprocess_raw_image(image, known_plate_mask=known_mask)
        save_image(destination / "denoised" / relative, result.denoised)
        save_image(
            destination / "plate_mask" / relative,
            result.approximate_plate_mask,
        )
        save_image(
            destination / "medium_mask" / relative,
            result.approximate_medium_mask,
        )
        save_image(
            destination / "light_corrected" / relative,
            result.corrected_u8,
        )
        records.append(
            {
                "source_path": str(image_path),
                "output_path": str(
                    destination / "light_corrected" / relative
                ),
                "medium_intensity_i0": result.medium_intensity,
                "formula": "log10(I0)-log10(Ii)",
                "bilateral_filter": "range_adaptive_approximation",
            }
        )
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(
        destination / "preprocessing_manifest.csv",
        index=False,
    )
    print(f"Preprocessed images: {len(records)}")


def main() -> None:
    parser = _base_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
