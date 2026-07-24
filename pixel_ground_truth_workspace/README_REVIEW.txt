PIXEL GROUND-TRUTH WORKSPACE

1. pseudo_masks/binary contains automatic candidate masks (0/255).
2. review_overlays shows boxes and candidate contours.
3. review_queue.csv lists annotations that require priority review.
4. annotation_crops/ contains per-box image, mask, and overlay crops.
5. Do not train U²-Net directly on unreviewed pseudo masks.
6. After manual correction, save final binary masks under:
   approved_masks/binary/<split>/<background>/<image_id>.png
7. Final masks must remain aligned to the original 2048x2048 ROI and contain
   only values 0 and 255.
