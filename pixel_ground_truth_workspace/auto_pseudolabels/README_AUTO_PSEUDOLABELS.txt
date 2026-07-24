AUTOMATIC PSEUDO-LABEL DATASET

Script version: 2026-07-24-auto-pseudolabel-v1
Policy: balanced

Outputs:
- binary/: automatic binary pseudo-labels (0 background, 255 colony)
- instance/: automatic instance IDs
- valid_regions/: pixels allowed to contribute to training loss
- weights/: per-pixel training weight in the range 0..255
- overlays/: automatic quality-control visualization
- annotation_auto_decisions.csv: accepted/ignored decision per bounding box
- image_auto_manifest.csv: paths and counts per image

Status meaning:
- accepted_high: strong consensus; full training weight
- accepted_medium: plausible candidate; reduced training weight
- ignored: excluded automatically; its bounding-box region has weight 0

These masks are pseudo-labels, not manually verified ground truth. Use the
valid-region and weight maps during U-Net/U2-Net training. Report the method as
weakly supervised / pseudo-label training.
