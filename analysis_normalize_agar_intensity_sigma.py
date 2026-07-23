from pathlib import Path

import pandas as pd


experiments = {
    "sigma025": Path(
        "preprocessed_intensity_sigma025/intensity_metrics.csv"
    ),
    "sigma040": Path(
        "preprocessed_intensity_sigma040/intensity_metrics.csv"
    ),
    "sigma060": Path(
        "preprocessed_intensity_sigma060/intensity_metrics.csv"
    ),
    "sigma080": Path(
        "preprocessed_intensity_sigma080/intensity_metrics.csv"
    ),
}

frames = []

for experiment_name, csv_path in experiments.items():
    frame = pd.read_csv(csv_path)
    frame = frame[
        frame["processing_status"] == "success"
    ].copy()

    frame["experiment"] = experiment_name
    frames.append(frame)

results = pd.concat(frames, ignore_index=True)

summary = (
    results.groupby(["experiment", "background"])
    .agg(
        n_images=("image_id", "count"),
        median_flatfield_std=(
            "flatfield_residual_std",
            "median",
        ),
        mean_flatfield_std=(
            "flatfield_residual_std",
            "mean",
        ),
        p90_flatfield_std=(
            "flatfield_residual_std",
            lambda values: values.quantile(0.90),
        ),
    )
    .reset_index()
)

print(summary)

results["better_than_global"] = (
    results["flatfield_residual_std"]
    < results["global_log_residual_std"]
)

improvement_rate = (
    results.groupby(["experiment", "background"])[
        "better_than_global"
    ]
    .mean()
    .mul(100)
)

print("\nPersentase lebih baik daripada global log:")
print(improvement_rate)