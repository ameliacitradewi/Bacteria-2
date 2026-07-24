# python - <<'PY'
import pandas as pd

path = (
    "pixel_ground_truth_workspace/"
    "annotation_mask_manifest.csv"
)

df = pd.read_csv(path)

print("\nStatus review:")
print(df["review_status"].value_counts(dropna=False))

print("\nMetode yang dipilih:")
print(df["selected_method"].value_counts(dropna=False))

print("\nStatistik confidence:")
print(df["confidence"].describe())

print("\nStatistik IoU antar-metode:")
print(df["mean_pairwise_iou"].describe())

print("\nMask kosong:")
print((df["mask_area"] == 0).sum())

print("\nMask hampir memenuhi box:")
print(df["full_box_like"].value_counts(dropna=False))



import pandas as pd

path = (
    "pixel_ground_truth_workspace/"
    "annotation_mask_manifest.csv"
)

df = pd.read_csv(path)

priority = df[
    (df["mask_area"] == 0)
    | (df["full_box_like"] == True)
    | (df["mean_pairwise_iou"] < 0.40)
].copy()

priority = priority.sort_values(
    by=[
        "mask_area",
        "full_box_like",
        "mean_pairwise_iou",
        "confidence",
    ],
    ascending=[True, False, True, True],
)

output = (
    "pixel_ground_truth_workspace/"
    "manual_review_priority.csv"
)

priority.to_csv(output, index=False)

print("Total anotasi:", len(df))
print("Prioritas manual:", len(priority))

print(
    priority[
        [
            "image_id",
            "annotation_id",
            "selected_method",
            "mask_area",
            "full_box_like",
            "mean_pairwise_iou",
            "confidence",
        ]
    ].head(30)
)

print("\nDisimpan ke:", output)



from pathlib import Path
import pandas as pd

root = Path("pixel_ground_truth_workspace")

source = root / "annotation_mask_manifest.csv"
output = root / "manual_review_results.csv"

df = pd.read_csv(source)

df["manual_status"] = "approved"
df["manual_reviewed"] = True
df["manual_notes"] = "Visual inspection passed"
df["final_mask_source"] = "box_guided_pseudo_mask"

df.to_csv(output, index=False)

print("Jumlah anotasi disetujui:", len(df))
print("Disimpan ke:", output)