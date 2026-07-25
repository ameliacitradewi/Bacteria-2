from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from agar_pipeline.metrics import colony_detection_loss, decode_centernet
from agar_pipeline.models import ResNet50FPNCenterNet, U2NETP


def main() -> None:
    plate = U2NETP().eval()
    with torch.no_grad():
        plate_outputs = plate(torch.randn(1, 3, 64, 64))
    assert len(plate_outputs) == 7
    assert all(output.shape == (1, 1, 64, 64) for output in plate_outputs)

    colony = ResNet50FPNCenterNet(fpn_channels=32, pretrained=False).eval()
    image = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        outputs = colony(image)
    assert outputs["heatmap_logits"].shape == (1, 1, 32, 32)
    target_heatmap = torch.zeros_like(outputs["heatmap_logits"])
    target_size = torch.zeros_like(outputs["size_raw"])
    target_offset = torch.zeros_like(outputs["offset_raw"])
    reg_mask = torch.zeros_like(target_heatmap)
    losses = colony_detection_loss(
        outputs, target_heatmap, target_size, target_offset, reg_mask
    )
    assert torch.isfinite(losses["loss"])
    decoded = decode_centernet(outputs, stride=4, score_threshold=0.99, nms_iou_threshold=0.3)
    assert len(decoded) == 1
    print("Smoke test model berhasil.")


if __name__ == "__main__":
    main()
