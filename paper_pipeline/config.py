from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperConfig:
    """Konfigurasi default yang mengikuti paper Cao et al. (2024).

    Preprocessing sigma 0.04 adalah adaptasi eksperimen AGAR proyek ini.
    Parameter model, threshold, dan normalisasi komponen mengikuti paper.
    """

    project_root: Path
    preprocessed_dir: Path
    counting_mask_dir: Path
    physical_mask_dir: Path
    metadata_csv: Path
    normalized_annotations_csv: Path
    data_root: Path
    output_root: Path
    checkpoint_root: Path

    edge_threshold: float = 0.1
    colony_threshold: float = 0.9
    component_size: int = 128
    # Proxy ring dibuat mendekati rasio foreground:background 1:8 yang
    # dilaporkan paper (sekitar 11.1% piksel foreground).
    edge_ring_width: int = 80
    min_component_area: int = 1

    @classmethod
    def from_project_root(cls, project_root: Path) -> "PaperConfig":
        root = project_root.resolve()
        plate_root = root / "processed_plate_strategy_b_circle"
        return cls(
            project_root=root,
            preprocessed_dir=(
                root
                / "preprocessed_intensity_sigma040"
                / "local_flatfield"
            ),
            counting_mask_dir=plate_root / "counting_mask",
            physical_mask_dir=plate_root / "physical_plate_mask",
            metadata_csv=root / "agar_metadata" / "image_manifest.csv",
            normalized_annotations_csv=(
                plate_root / "object_annotations_normalized.csv"
            ),
            data_root=root / "paper_data",
            output_root=root / "paper_outputs",
            checkpoint_root=root / "checkpoints" / "paper_pipeline",
        )
