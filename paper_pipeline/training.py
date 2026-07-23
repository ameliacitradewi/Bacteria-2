from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import ResNetCountDataset, SegmentationDataset
from .models import U2Net, build_resnet50_counter


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def segmentation_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
    truth = targets >= 0.5
    tp = float((predictions & truth).sum().item())
    fp = float((predictions & ~truth).sum().item())
    fn = float((~predictions & truth).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    conventional_f1 = (
        2.0 * precision * recall / max(precision + recall, 1e-8)
    )
    # Formula yang dicetak paper (setara F_beta dengan beta²=0.3).
    paper_f_score = (
        1.3
        * precision
        * recall
        / max(0.3 * precision + recall, 1e-8)
    )
    mae = float(torch.mean(torch.abs(probabilities - targets)).item())
    return {
        "precision": precision,
        "recall": recall,
        "f1": conventional_f1,
        "paper_f_score": paper_f_score,
        "mae": mae,
    }


def _u2net_loss(
    outputs: tuple[torch.Tensor, ...],
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    losses = [
        F.binary_cross_entropy_with_logits(
            output,
            targets,
            pos_weight=pos_weight,
        )
        for output in outputs
    ]
    return torch.stack(losses).sum()


def train_u2net(
    manifest_path: Path,
    checkpoint_dir: Path,
    task: str,
    epochs: int = 200,
    resize_long_side: int | None = None,
    device_name: str = "auto",
    seed: int = 42,
    num_workers: int = 0,
) -> Path:
    """Train salah satu dari dua U2-Net paper.

    Hyperparameter utama:
    AdamW(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4),
    cosine LR, BCEWithLogits, batch size 1, random initialization.
    """
    if task not in {"edge", "colony"}:
        raise ValueError("task harus 'edge' atau 'colony'.")
    set_seed(seed)
    device = select_device(device_name)
    train_dataset = SegmentationDataset(
        manifest_path,
        split="train",
        horizontal_flip_probability=(0.5 if task == "colony" else 0.0),
        resize_long_side=resize_long_side,
    )
    validation_dataset = SegmentationDataset(
        manifest_path,
        split="validation",
        resize_long_side=resize_long_side,
    )
    if not train_dataset or not validation_dataset:
        raise RuntimeError(
            "Dataset train/validation U2-Net belum lengkap. "
            "Periksa manifest dan mask piksel."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )
    model = U2Net(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
    )
    pos_weight = torch.tensor([8.0], device=device)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_score = -1.0
    best_path = checkpoint_dir / f"u2net_{task}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(
            train_loader,
            desc=f"U2Net {task} train {epoch}/{epochs}",
            leave=False,
        ):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = _u2net_loss(outputs, masks, pos_weight)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        validation_loss = 0.0
        score_sums = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "paper_f_score": 0.0,
            "mae": 0.0,
        }
        with torch.no_grad():
            for batch in validation_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                outputs = model(images)
                validation_loss += float(
                    _u2net_loss(outputs, masks, pos_weight).item()
                )
                scores = segmentation_scores(outputs[0], masks, threshold=0.5)
                for name, value in scores.items():
                    score_sums[name] += value

        n_train = max(len(train_loader), 1)
        n_validation = max(len(validation_loader), 1)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss / n_train,
            "validation_loss": validation_loss / n_validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        for name, value in score_sums.items():
            record[f"validation_{name}"] = value / n_validation
        history.append(record)

        current_score = float(record["validation_paper_f_score"])
        checkpoint = {
            "task": task,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": record,
            "model": "U2Net",
            "input_scaling": "RGB_0_1_no_mean_std",
        }
        torch.save(
            checkpoint,
            checkpoint_dir / f"u2net_{task}_last.pt",
        )
        if current_score > best_score:
            best_score = current_score
            torch.save(checkpoint, best_path)

        pd.DataFrame(history).to_csv(
            checkpoint_dir / f"u2net_{task}_history.csv",
            index=False,
        )
        scheduler.step()
        print(json.dumps(record, indent=2))

    return best_path


def _resnet_class_weights(
    manifest_path: Path,
    device: torch.device,
) -> torch.Tensor:
    frame = pd.read_csv(manifest_path)
    train = frame[frame["split"].astype(str) == "train"]
    counts = train["label"].value_counts().reindex(range(10), fill_value=0)
    if int(counts.max()) == 0:
        raise RuntimeError("Manifest ResNet tidak memiliki data train.")
    maximum = float(counts.max())
    weights = np.ones(10, dtype=np.float32)
    for index, count in enumerate(counts.to_numpy()):
        weights[index] = maximum / float(count) if count > 0 else 0.0
    return torch.from_numpy(weights).to(device)


def train_resnet50(
    manifest_path: Path,
    checkpoint_dir: Path,
    epochs: int = 200,
    device_name: str = "auto",
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Path:
    set_seed(seed)
    device = select_device(device_name)
    train_dataset = ResNetCountDataset(manifest_path, split="train")
    validation_dataset = ResNetCountDataset(
        manifest_path,
        split="validation",
    )
    if not train_dataset or not validation_dataset:
        raise RuntimeError(
            "Dataset train/validation ResNet50 belum lengkap."
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model = build_resnet50_counter(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    class_weights = _resnet_class_weights(manifest_path, device)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_path = checkpoint_dir / "resnet50_count_best.pt"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(
            train_loader,
            desc=f"ResNet50 train {epoch}/{epochs}",
            leave=False,
        ):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            targets = F.one_hot(labels, num_classes=10).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                weight=class_weights,
            )
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        correct = 0
        total = 0
        validation_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                targets = F.one_hot(labels, num_classes=10).float()
                logits = model(images)
                validation_loss += float(
                    F.binary_cross_entropy_with_logits(
                        logits,
                        targets,
                        weight=class_weights,
                    ).item()
                )
                predictions = torch.argmax(logits, dim=1)
                correct += int((predictions == labels).sum().item())
                total += int(labels.numel())

        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss / max(len(train_loader), 1),
            "validation_loss": validation_loss
            / max(len(validation_loader), 1),
            "validation_accuracy": correct / max(total, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": record,
            "model": "ResNet50",
            "num_classes": 10,
            "class_9_semantics": "9_or_more",
            "input_scaling": "RGB_0_1_no_mean_std",
        }
        torch.save(
            checkpoint,
            checkpoint_dir / "resnet50_count_last.pt",
        )
        accuracy = float(record["validation_accuracy"])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(checkpoint, best_path)
        pd.DataFrame(history).to_csv(
            checkpoint_dir / "resnet50_count_history.csv",
            index=False,
        )
        print(json.dumps(record, indent=2))

    return best_path

