import pandas as pd

metrics = pd.read_csv(
    "preprocessed_intensity/intensity_metrics.csv"
)

successful = metrics[
    metrics["processing_status"] == "success"
]

columns = [
    "raw_medium_cv",
    "global_log_residual_std",
    "flatfield_residual_std",
]

print(successful[columns].describe())

print(
    successful.groupby("background")[columns]
    .median()
)