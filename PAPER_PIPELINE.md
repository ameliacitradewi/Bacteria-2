# Pipeline paper U²-Net + ResNet50

Implementasi ini mengikuti alur Cao et al. (2024), **U²-Net and
ResNet50-Based Automatic Pipeline for Bacterial Colony Counting**:

```text
preprocessing intensitas
→ U²-Net #1: edge cawan
→ threshold 0.1
→ ROI bagian dalam cawan
→ U²-Net #2: area koloni
→ threshold 0.9
→ connected-component analysis
→ rotasi sumbu utama komponen
→ resize/pad 128×128
→ ResNet50 kelas 0–9
→ jumlah koloni per gambar
```

Tidak ada patch extraction `256×256` dalam pipeline ini. Eksperimen klasik
Otsu/adaptive/Sauvola tetap berada di `bacteria_segmentation_pipeline.py` dan
bukan bagian utama pipeline paper.

Sumber metodologi:

- Paper: <https://doi.org/10.3390/microorganisms12010201>
- U²-Net: <https://arxiv.org/abs/2005.09007>

## Batas reproduksi pada dataset AGAR

Kode pipeline lengkap tersedia, tetapi model tidak dapat menghasilkan prediksi
ilmiah hanya dengan arsitektur:

1. Paper melatih U²-Net edge dengan 255 **mask piksel manual** berbentuk ring.
   AGAR tidak menyediakannya. Perintah `prepare-edge` membuat **proxy ring**
   dari counting mask yang sudah ada. Proxy ini berguna untuk eksperimen, tetapi
   tidak boleh disebut label manual yang identik dengan paper.
2. Paper melatih U²-Net koloni dengan 255 **mask piksel koloni manual**.
   AGAR yang tersedia hanya memiliki bounding box. Mask koloni manual tetap
   harus dibuat untuk reproduksi yang benar.
3. Paper memberi label jumlah `0–9` secara manual pada setiap connected
   component. `prepare-resnet` menyediakan adaptasi AGAR dengan menghitung pusat
   bounding box yang masuk ke setiap komponen. Ini adalah proxy label.
4. Kelas ResNet50 `9` berarti **9 atau lebih**, sehingga hasil penjumlahan yang
   mengandung kelas 9 adalah batas bawah.

## Instalasi

```bash
cd /Users/ameliacitra/Documents/AI-ML/Bacteria-2
source .venv/bin/activate
python -m pip install -r requirements-paper.txt
```

Periksa kesiapan:

```bash
python run_paper_pipeline.py doctor
```

## 1. Input preprocessing

Jalur utama proyek memakai hasil eksperimen yang sudah dikunci:

```text
preprocessed_intensity_sigma040/local_flatfield/
```

File tersebut sudah mencakup ROI geometri, light/flat-field correction
`sigma=0.04`, dan robust intensity scaling. Pipeline tidak melakukan koreksi
kedua.

Untuk membandingkan dengan rumus referensi `log10(I0)-log10(Ii)`, tersedia:

```bash
python run_paper_pipeline.py preprocess-reference \
  --input-dir processed_plate_strategy_b_circle/plate_crop_normalized \
  --known-mask-dir processed_plate_strategy_b_circle/counting_mask \
  --output-dir paper_data/preprocessing_reference \
  --limit 2
```

Filter bilateral adaptif paper tidak diterbitkan beserta parameter lengkapnya.
Implementasi referensi mengestimasi range sigma dari robust spread luminance dan
mencatatnya sebagai aproksimasi.

## 2. U²-Net edge cawan

Siapkan gambar penuh dan proxy edge labels:

```bash
python run_paper_pipeline.py prepare-edge
```

Output:

```text
paper_data/edge/
├── train/images/
├── train/masks/
├── validation/images/
├── validation/masks/
└── manifest.csv
```

Latih model:

```bash
python run_paper_pipeline.py train-u2net --task edge --epochs 200
```

Parameter paper yang digunakan:

- random initialization;
- AdamW, learning rate `1e-3`;
- betas `(0.9, 0.999)`, epsilon `1e-8`;
- cosine learning-rate decay;
- weight decay `1e-4`;
- BCEWithLogitsLoss;
- foreground-to-background weight `8:1`;
- batch size `1`;
- input RGB skala `0–1`, tanpa mean/std standardization.

## 3. U²-Net segmentasi koloni

Siapkan struktur berikut menggunakan **mask piksel manual**:

```text
paper_data/colony/
├── train/
│   ├── images/
│   └── masks/
└── validation/
    ├── images/
    └── masks/
```

Setiap image dan mask harus mempunyai nama relatif yang sama. Gambar input
merupakan ROI yang dihasilkan tahap U²-Net edge; mask berisi piksel koloni
sebagai foreground.

Audit dan buat manifest:

```bash
python run_paper_pipeline.py build-colony-manifest
```

Latih U²-Net kedua:

```bash
python run_paper_pipeline.py train-u2net --task colony --epochs 200
```

Training koloni menggunakan horizontal flip dengan probabilitas `0.5`, sesuai
paper. Parameter lainnya sama dengan U²-Net edge.

## 4. Hasil dua U²-Net

Setelah kedua weight tersedia:

```bash
python run_paper_pipeline.py segment
```

Output utama:

```text
paper_outputs/segmentation/
├── edge_probability/
├── edge_mask/
├── roi_mask/
├── roi_image/
├── colony_probability/
├── colony_mask/
└── segmentation_manifest.csv
```

## 5. Persiapan dan training ResNet50

Gunakan hasil `colony_mask`:

```bash
python run_paper_pipeline.py prepare-resnet \
  --colony-mask-dir paper_outputs/segmentation/colony_mask
```

Tahap ini:

- mencari connected components;
- menyelaraskan sumbu utama komponen secara vertikal;
- mempertahankan aspect ratio;
- mengecilkan sisi terpanjang bila lebih dari 128;
- melakukan zero-padding menjadi `128×128×3`;
- memberi label proxy `0–9` berdasarkan pusat bounding box AGAR.

Latih ResNet50:

```bash
python run_paper_pipeline.py train-resnet --epochs 200
```

Implementasi menggunakan random-init ResNet50 dengan output 10 kelas,
Adam `1e-4`, BCEWithLogitsLoss, serta bobot kelas
`max(class_count)/class_count`.

## 6. Inference penuh

```bash
python run_paper_pipeline.py infer --limit 2
```

Tanpa `--limit`, seluruh input sigma040 diproses. Default weight:

```text
checkpoints/paper_pipeline/u2net_edge_best.pt
checkpoints/paper_pipeline/u2net_colony_best.pt
checkpoints/paper_pipeline/resnet50_count_best.pt
```

Output akhir:

```text
paper_outputs/
├── edge_probability/
├── edge_mask/
├── roi_mask/
├── roi_image/
├── colony_probability/
├── colony_mask/
├── components/
├── component_overlay/
├── counts_per_component.csv
└── counts_per_image.csv
```

`counts_per_image.csv` menandai `total_is_lower_bound=True` bila satu atau lebih
komponen diprediksi sebagai kelas 9.

