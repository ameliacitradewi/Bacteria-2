# run kalo lokasi output khusus: python build_agar_manifest.py ./AGAR_dataset/dataset --output ./agar_metadata
# kalo umum: python build_agar_manifest.py
# result: agar_metadata

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

MICROORGANISM_CLASSES = {
    "Staphylococcus aureus",
    "Bacillus subtilis",
    "Pseudomonas aeruginosa",
    "Escherichia coli",
    "Candida albicans",
}

ARTIFACT_CLASSES = {
    "Defect",
    "Contamination",
}


def determine_plate_condition(colonies_number: Any) -> str:
    """
    Infer kondisi plate dari colonies_number.
    """
    if colonies_number is None:
        return "unknown"

    try:
        colonies_number = int(colonies_number)
    except (TypeError, ValueError):
        return "unknown"

    if colonies_number == -1:
        return "uncountable"
    if colonies_number == 0:
        return "empty"
    if colonies_number > 0:
        return "countable"

    return "unknown"


def object_type(class_name: str) -> str:
    if class_name in MICROORGANISM_CLASSES:
        return "colony"
    if class_name in ARTIFACT_CLASSES:
        return "artifact"
    return "other"


def build_image_index(root: Path) -> dict[str, list[Path]]:
    """
    Membuat indeks gambar berdasarkan nama dasar file.
    Contoh: 1234.jpg -> key '1234'
    """
    image_index: dict[str, list[Path]] = defaultdict(list)

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_index[path.stem].append(path)

    return dict(image_index)


def select_matching_image(
    json_path: Path,
    candidates: list[Path],
) -> Path:
    """
    Memilih gambar dengan nama dasar yang sama.

    Jika terdapat lebih dari satu kandidat, prioritaskan kandidat
    yang berada pada struktur folder paling mirip dengan JSON.
    """
    if not candidates:
        raise FileNotFoundError(
            f"Tidak ditemukan gambar untuk annotation: {json_path}"
        )

    if len(candidates) == 1:
        return candidates[0]

    # Prioritaskan kandidat dengan nama parent folder yang sama.
    same_parent_name = [
        path for path in candidates
        if path.parent.name == json_path.parent.name
    ]

    if len(same_parent_name) == 1:
        return same_parent_name[0]

    candidate_text = "\n".join(str(path) for path in candidates)
    raise ValueError(
        f"Terdapat beberapa gambar dengan stem '{json_path.stem}':\n"
        f"{candidate_text}"
    )


def read_image_size(image_path: Path) -> tuple[int, int]:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
        return width, height
    except Exception as exc:
        raise RuntimeError(
            f"Gagal membaca ukuran gambar {image_path}: {exc}"
        ) from exc


def create_group_splits(
    manifest: pd.DataFrame,
    random_state: int = 42,
) -> pd.Series:
    """
    Membuat split:
    - 75% train
    - 12.5% validation
    - 12.5% test

    Split dilakukan berdasarkan group_id agar satu sample_id
    tidak tersebar di beberapa subset.
    """
    if manifest.empty:
        return pd.Series(dtype="object")

    groups = manifest["group_id"].astype(str)

    first_split = GroupShuffleSplit(
        n_splits=1,
        train_size=0.75,
        random_state=random_state,
    )

    train_index, temporary_index = next(
        first_split.split(manifest, groups=groups)
    )

    temporary = manifest.iloc[temporary_index].copy()

    second_split = GroupShuffleSplit(
        n_splits=1,
        train_size=0.50,
        random_state=random_state + 1,
    )

    validation_relative, test_relative = next(
        second_split.split(
            temporary,
            groups=temporary["group_id"].astype(str),
        )
    )

    validation_index = temporary.index[validation_relative]
    test_index = temporary.index[test_relative]

    split = pd.Series(index=manifest.index, dtype="object")
    split.loc[train_index] = "train"
    split.loc[validation_index] = "validation"
    split.loc[test_index] = "test"

    return split


def build_manifest(
    dataset_root: Path,
    output_directory: Path,
) -> None:
    dataset_root = dataset_root.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    image_index = build_image_index(dataset_root)
    json_paths = sorted(dataset_root.rglob("*.json"))

    image_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    missing_pairs: list[str] = []

    for json_path in json_paths:
        try:
            with json_path.open("r", encoding="utf-8") as file:
                annotation = json.load(file)
        except Exception as exc:
            print(f"[WARNING] Gagal membaca {json_path}: {exc}")
            continue

        candidates = image_index.get(json_path.stem, [])

        try:
            image_path = select_matching_image(json_path, candidates)
        except (FileNotFoundError, ValueError) as exc:
            missing_pairs.append(str(exc))
            continue

        width, height = read_image_size(image_path)

        sample_id = annotation.get("sample_id")
        image_id = json_path.stem
        group_id = sample_id if sample_id is not None else image_id

        background = annotation.get("background", "unknown")
        inoculated_classes = annotation.get("classes") or []
        colonies_number = annotation.get("colonies_number")
        condition = determine_plate_condition(colonies_number)
        labels = annotation.get("labels") or []

        colony_boxes = 0
        artifact_boxes = 0
        other_boxes = 0

        for annotation_index, label in enumerate(labels):
            class_name = str(label.get("class", "unknown"))
            annotation_type = object_type(class_name)

            if annotation_type == "colony":
                colony_boxes += 1
            elif annotation_type == "artifact":
                artifact_boxes += 1
            else:
                other_boxes += 1

            x = label.get("x")
            y = label.get("y")
            box_width = label.get("width")
            box_height = label.get("height")

            object_rows.append(
                {
                    "image_id": image_id,
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "annotation_index": annotation_index,
                    "annotation_id": label.get("id"),
                    "class_name": class_name,
                    "object_type": annotation_type,
                    "x": x,
                    "y": y,
                    "width": box_width,
                    "height": box_height,
                    "x_max": (
                        x + box_width
                        if isinstance(x, (int, float))
                        and isinstance(box_width, (int, float))
                        else None
                    ),
                    "y_max": (
                        y + box_height
                        if isinstance(y, (int, float))
                        and isinstance(box_height, (int, float))
                        else None
                    ),
                    "image_width": width,
                    "image_height": height,
                }
            )

        count_difference = None
        if condition == "countable":
            try:
                count_difference = int(colonies_number) - colony_boxes
            except (TypeError, ValueError):
                count_difference = None

        image_rows.append(
            {
                "image_id": image_id,
                "sample_id": sample_id,
                "group_id": group_id,
                "image_path": image_path.relative_to(dataset_root).as_posix(),
                "annotation_path": json_path.relative_to(
                    dataset_root
                ).as_posix(),
                "image_width": width,
                "image_height": height,
                "background": background,
                "plate_condition": condition,
                "colonies_number": colonies_number,
                "classes_inoculated": "|".join(
                    str(item) for item in inoculated_classes
                ),
                "n_inoculated_classes": len(inoculated_classes),
                "n_labels_total": len(labels),
                "n_colony_boxes": colony_boxes,
                "n_artifact_boxes": artifact_boxes,
                "n_other_boxes": other_boxes,
                "count_minus_colony_boxes": count_difference,
                "has_defect": int(
                    any(
                        str(label.get("class")) == "Defect"
                        for label in labels
                    )
                ),
                "has_contamination": int(
                    any(
                        str(label.get("class")) == "Contamination"
                        for label in labels
                    )
                ),
            }
        )

    manifest = pd.DataFrame(image_rows)
    objects = pd.DataFrame(object_rows)

    if manifest.empty:
        raise RuntimeError(
            "Manifest kosong. Periksa lokasi dataset dan format JSON."
        )

    # Tambahkan split dengan mencegah kebocoran sample_id.
    manifest["split"] = create_group_splits(manifest)

    # Tambahkan informasi split pada tabel objek.
    if not objects.empty:
        split_map = manifest.set_index("image_id")["split"]
        objects["split"] = objects["image_id"].map(split_map)

    manifest_path = output_directory / "image_manifest.csv"
    objects_path = output_directory / "object_annotations.csv"

    manifest.to_csv(manifest_path, index=False)
    objects.to_csv(objects_path, index=False)

    # Simpan laporan pasangan file bermasalah.
    if missing_pairs:
        missing_path = output_directory / "missing_or_ambiguous_pairs.txt"
        missing_path.write_text(
            "\n\n".join(missing_pairs),
            encoding="utf-8",
        )

    print("\nManifest berhasil dibuat")
    print(f"Image manifest : {manifest_path}")
    print(f"Object table   : {objects_path}")
    print(f"Jumlah gambar  : {len(manifest):,}")
    print(f"Jumlah objek   : {len(objects):,}")

    print("\nDistribusi split:")
    print(manifest["split"].value_counts(dropna=False))

    print("\nDistribusi background:")
    print(
        pd.crosstab(
            manifest["background"],
            manifest["split"],
            margins=True,
        )
    )

    print("\nDistribusi kondisi plate:")
    print(
        pd.crosstab(
            manifest["plate_condition"],
            manifest["split"],
            margins=True,
        )
    )

    discrepancy = manifest[
        (manifest["plate_condition"] == "countable")
        & (manifest["count_minus_colony_boxes"].fillna(0) != 0)
    ]

    print(
        "\nCountable image dengan colonies_number "
        "berbeda dari jumlah bounding box:",
        len(discrepancy),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Membuat manifest dataset AGAR."
    )
    parser.add_argument(
    "dataset_root",
    type=Path,
    nargs="?",
    default=Path("AGAR_dataset/dataset"),
    help="Folder utama dataset AGAR.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("agar_metadata"),
        help="Folder keluaran manifest.",
    )

    args = parser.parse_args()

    build_manifest(
        dataset_root=args.dataset_root,
        output_directory=args.output,
    )


if __name__ == "__main__":
    main()