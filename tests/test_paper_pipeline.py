from __future__ import annotations

import unittest

import cv2
import numpy as np

from paper_pipeline.components import extract_colony_components
from paper_pipeline.preprocessing import (
    inner_roi_from_edge_mask,
    make_edge_ring_label,
)


class PaperPipelineGeometryTests(unittest.TestCase):
    def test_edge_ring_recovers_inner_roi(self) -> None:
        plate = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(plate, (128, 128), 100, 255, thickness=cv2.FILLED)
        ring = make_edge_ring_label(plate, ring_width=20)
        roi = inner_roi_from_edge_mask(ring)
        self.assertGreater(int(ring.sum()), 0)
        self.assertEqual(int(roi[128, 128]), 255)

    def test_component_is_rotated_and_normalized(self) -> None:
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (128, 128),
            (25, 60),
            30,
            0,
            360,
            255,
            thickness=cv2.FILLED,
        )
        image[mask > 0] = (180, 180, 180)
        components, _ = extract_colony_components(
            image,
            mask,
            target_size=128,
            min_area=10,
        )
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].crop.shape, (128, 128, 3))


class PaperPipelineModelTests(unittest.TestCase):
    def test_model_output_shapes(self) -> None:
        import torch

        from paper_pipeline.models import U2Net, build_resnet50_counter

        u2net = U2Net().eval()
        with torch.no_grad():
            outputs = u2net(torch.rand(1, 3, 64, 64))
        self.assertEqual(len(outputs), 7)
        for output in outputs:
            self.assertEqual(tuple(output.shape), (1, 1, 64, 64))

        resnet = build_resnet50_counter().eval()
        with torch.no_grad():
            logits = resnet(torch.rand(2, 3, 128, 128))
        self.assertEqual(tuple(logits.shape), (2, 10))


if __name__ == "__main__":
    unittest.main()
