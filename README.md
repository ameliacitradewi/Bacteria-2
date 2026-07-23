# Bacteria-2

Eksperimen pemrosesan citra dan penghitungan koloni pada dataset AGAR.

## Dua pipeline yang dipisahkan

- `bacteria_segmentation_pipeline.py`: baseline klasik berbasis patch,
  Otsu/adaptive/Sauvola. Bukan pipeline utama paper.
- `run_paper_pipeline.py`: pipeline U²-Net edge → U²-Net koloni →
  connected components → ResNet50.

Petunjuk lengkap pipeline paper terdapat di
[`PAPER_PIPELINE.md`](PAPER_PIPELINE.md).

Audit kesiapan lokal:

```bash
python run_paper_pipeline.py doctor
```
