from __future__ import annotations

import argparse
from pathlib import Path

from agar_pipeline.common import load_config, save_json
from agar_pipeline.data import materialize_colony_tiles
from agar_pipeline.engine import (
    compile_all_metrics,
    evaluate_colony_model,
    export_models,
    train_colony_model,
    train_plate_model,
)
from agar_pipeline.preprocess import run_preprocessing
from agar_pipeline.production import evaluate_production_pipeline, predict_single_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline end-to-end AGAR: U2-NetP plate localization + ResNet50-FPN colony counting."
    )
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "train-plate",
            "prepare-tiles",
            "train-colony",
            "evaluate",
            "evaluate-e2e",
            "predict",
            "export",
            "all",
        ],
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--overwrite-tiles", action="store_true")
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument("--force-plate", action="store_true")
    parser.add_argument("--force-intensity", action="store_true")
    parser.add_argument("--image", type=Path, help="Foto tunggal untuk command predict.")
    parser.add_argument("--prediction-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "metadata" / "resolved_config.json", config)

    if args.command in {"prepare", "all"}:
        run_preprocessing(
            config,
            force_manifest=args.force_manifest,
            force_plate=args.force_plate,
            force_intensity=args.force_intensity,
        )
    if args.command in {"train-plate", "all"}:
        train_plate_model(config)
    if args.command in {"prepare-tiles", "all"}:
        materialize_colony_tiles(config, overwrite=args.overwrite_tiles)
    if args.command in {"train-colony", "all"}:
        train_colony_model(config)
    if args.command in {"evaluate", "all"}:
        evaluate_colony_model(config, split="val")
        evaluate_colony_model(config, split="test")
        compile_all_metrics(config)
    if args.command in {"evaluate-e2e", "all"}:
        evaluate_production_pipeline(config, split="val")
        evaluate_production_pipeline(config, split="test")
        compile_all_metrics(config)
    if args.command == "predict":
        if args.image is None:
            raise SystemExit("Command predict membutuhkan --image /path/foto.jpg")
        result = predict_single_image(
            config, args.image, destination=args.prediction_output
        )
        print(f"Predicted colonies: {result['predicted_count']}")
        print(f"Output: {Path(result['overlay_path']).parent}")
    if args.command in {"export", "all"}:
        export_models(config)
        compile_all_metrics(config)


if __name__ == "__main__":
    main()
