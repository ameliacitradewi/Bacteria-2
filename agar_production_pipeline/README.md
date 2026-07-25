# AGAR Production Pipeline

Pipeline ini menambahkan training dan evaluasi production-oriented ke repository **Bacteria-2**:

1. membangun manifest AGAR;
2. menjalankan deteksi outer plate klasik yang sudah ada;
3. menjalankan normalisasi ROI dan flat-field sigma `0.04`;
4. memakai hasil outer-plate sebagai pseudo-ground-truth untuk melatih **U²-NetP**;
5. membuat tile koloni sekali saja agar preprocessing tidak dihitung ulang setiap epoch;
6. melatih detector **ResNet50-FPN CenterNet-style** satu kelas (`colony`);
7. mengevaluasi segmentasi, deteksi, dan jumlah koloni per cawan;
8. menyimpan seluruh metrik ke CSV dan lima visualisasi dari setiap tahap;
9. mengekspor TorchScript serta mencoba membuat Core ML `.mlpackage` FP16.

## Mengapa detector-nya bukan `torchvision` RetinaNet langsung?

Backbone tetap **ResNet50-FPN**, tetapi head dibuat CenterNet-style dengan output tetap:

- `heatmap`: lokasi pusat koloni;
- `size`: lebar dan tinggi bounding box;
- `offset`: koreksi posisi sub-pixel.

Model tidak memasukkan decoding atau NMS ke graph. Bentuk ini lebih stabil untuk Core ML. Tiling, decoding, penggabungan koordinat, dan NMS dijalankan oleh aplikasi macOS.


## Lisensi dataset AGAR

Situs resmi AGAR menyatakan dataset tersedia untuk riset akademik dengan lisensi **CC BY-NC 2.0**. Untuk aplikasi komersial, minta izin atau lisensi terpisah dari pemilik dataset sebelum menjadikan bobot hasil training sebagai bagian produk.

## Penempatan folder

Letakkan folder `agar_production_pipeline` di dalam root repository:

```text
Bacteria-2/
├── build_agar_manifest.py
├── detect_outer_plate_strategy_b.py
├── normalize_agar_intensity_v1.py
├── AGAR_dataset/
└── agar_production_pipeline/
```

Salin konfigurasi:

```bash
cd Bacteria-2/agar_production_pipeline
cp config.example.yaml config.yaml
```

Periksa keenam path di bagian atas `config.yaml` sebelum menjalankan training.

## Instalasi di Kaggle

Kaggle biasanya sudah memiliki PyTorch. Instal paket tambahan:

```bash
pip install -r requirements.txt
```

Untuk ekspor Core ML, instal tambahan:

```bash
pip install -r requirements-coreml.txt
```

Core ML Tools 9.0 secara resmi mendukung PyTorch sampai seri 2.7. Jika image Kaggle memakai PyTorch lebih baru dan konversi gagal, salin checkpoint/TorchScript ke environment macOS dengan PyTorch 2.7.x, lalu jalankan command `export` di sana.

Jika internet notebook dimatikan dan bobot ImageNet ResNet50 belum ada di cache, kode akan melanjutkan dengan random initialization. Untuk hasil final, lebih baik aktifkan internet sekali atau unggah bobot pretrained sebagai Kaggle Dataset.

## Jalankan bertahap

Menjalankan tahap secara terpisah adalah pilihan yang disarankan. Checkpoint dan output yang sudah ada tidak dibuat ulang secara otomatis.

```bash
python run_pipeline.py prepare --config config.yaml
python run_pipeline.py train-plate --config config.yaml
python run_pipeline.py prepare-tiles --config config.yaml
python run_pipeline.py train-colony --config config.yaml
python run_pipeline.py evaluate --config config.yaml
python run_pipeline.py evaluate-e2e --config config.yaml
python run_pipeline.py export --config config.yaml
```

Seluruh tahap sekaligus:

```bash
python run_pipeline.py all --config config.yaml
```

Prediksi satu foto setelah checkpoint tersedia:

```bash
python run_pipeline.py predict --config config.yaml --image /path/foto_iphone.jpg
```

Command `evaluate` mengukur detector pada ROI klasik yang stabil. Command `evaluate-e2e` mengukur jalur production sebenarnya dari foto mentah, sehingga error lokalisasi U²-NetP ikut masuk ke MAE dan mAP akhir.

Gunakan `--force-manifest`, `--force-plate`, atau `--force-intensity` hanya saat memang ingin mengulang preprocessing tersebut. Gunakan `--overwrite-tiles` untuk membangun ulang semua tile.

## Smoke test sebelum training penuh

Paling mudah gunakan konfigurasi yang sudah disediakan:

```bash
python run_pipeline.py all --config config.smoke.yaml
```

Atau ubah sementara:

```yaml
plate:
  max_train_images: 100
  epochs: 2

colony:
  max_train_tiles: 500
  epochs: 2
  workers: 2
```

Setelah smoke test berhasil, kembalikan `max_train_tiles: null`, tambah epoch, dan gunakan subset plate yang representatif.

## Output

Default output berada pada `production_artifacts/`:

```text
production_artifacts/
├── checkpoints/
│   ├── u2netp_plate_best.pt
│   └── resnet50_fpn_centernet_best.pt
├── export/
│   ├── PlateU2NetP.torchscript.pt
│   ├── ColonyResNet50FPN.torchscript.pt
│   ├── PlateU2NetP.mlpackage            # jika konversi berhasil
│   ├── ColonyResNet50FPN.mlpackage      # jika konversi berhasil
│   ├── production_parameters.json
│   └── export_status.csv
├── metadata/
│   ├── tile_manifest.csv
│   ├── resolved_config.json
│   └── preprocessing_summary.json
├── metrics/
│   ├── plate_training_history.csv
│   ├── plate_val_per_image.csv
│   ├── plate_val_summary.csv
│   ├── plate_test_per_image.csv
│   ├── plate_test_summary.csv
│   ├── tile_materialization_metrics.csv
│   ├── colony_training_history.csv
│   ├── colony_val_tile_detection_summary.csv
│   ├── colony_val_full_detection_summary.csv
│   ├── colony_val_count_summary.csv
│   ├── colony_val_per_image.csv
│   ├── colony_test_tile_detection_summary.csv
│   ├── colony_test_full_detection_summary.csv
│   ├── colony_test_count_summary.csv
│   ├── colony_test_per_image.csv
│   ├── production_e2e_val_per_image.csv
│   ├── production_e2e_val_count_summary.csv
│   ├── production_e2e_val_detection_summary.csv
│   ├── production_e2e_val_runtime_summary.csv
│   ├── production_e2e_test_per_image.csv
│   ├── production_e2e_test_count_summary.csv
│   ├── production_e2e_test_detection_summary.csv
│   ├── production_e2e_test_runtime_summary.csv
│   └── all_evaluation_metrics.csv
└── visual_samples/
    ├── 01_outer_plate/                 # 5 gambar
    ├── 02_normalized_roi/              # 5 gambar
    ├── 03_intensity_flatfield/          # 5 gambar
    ├── 04_training_tiles/               # 5 gambar
    ├── 05_u2netp_predictions/           # 5 gambar
    ├── 06_colony_predictions/           # 5 gambar
    └── 07_production_e2e_predictions/    # 5 gambar
```

## Metrik

### U²-NetP plate localization

- Dice;
- IoU;
- precision dan recall pixel;
- normalized center error;
- relative area error;
- relative equivalent-radius error;
- empty prediction rate.

### Detector koloni

- tile mAP, mAP@0.5, mAP@0.75;
- full-plate mAP, mAP@0.5, mAP@0.75 setelah tile merge;
- precision, recall, dan F1 pada IoU 0.5;
- TP, FP, dan FN.

### Counting end-to-end

Dihitung dua kali: pada ROI klasik untuk diagnosis detector, dan pada pipeline production mentah untuk metrik release.

- MAE;
- median absolute error;
- RMSE;
- mean bias;
- MAPE untuk plate non-empty;
- SMAPE;
- proporsi error maksimal 5 dan 10 koloni;
- proporsi error maksimal `max(2 koloni, 10%)`;
- korelasi Pearson;
- success rate lokalisasi cawan;
- mean, median, dan p90 latency untuk plate localization, normalisasi, detector, dan total pipeline.

`colonies_number` dipakai sebagai ground truth jumlah utama. `true_count_boxes` tetap disimpan agar ketidaksesuaian antara metadata count dan jumlah bounding box dapat diaudit.

## Hal yang mempercepat training

- U²-NetP memakai input 320×320 dan hanya subset pseudo-label yang representatif.
- Flat-field dihitung satu kali dan disimpan, bukan dihitung ulang setiap epoch.
- Tile koloni dimaterialisasi satu kali menjadi JPEG.
- Pada train, semua tile positif dipakai tetapi negative tile dibatasi per gambar.
- AMP aktif otomatis pada CUDA.
- Checkpoint terbaik dipilih dengan validation Dice atau validation loss.
- Early stopping mencegah epoch yang tidak lagi memberi perbaikan.

## Core ML

Perintah `export` selalu menyimpan TorchScript. Konversi Core ML dicoba dengan ML Program FP16. Status akhir ada di `export/export_status.csv`.

Model koloni mengeluarkan tensor mentah. Aplikasi harus menjalankan:

1. tile ROI berukuran 512 dengan overlap 128;
2. normalisasi ImageNet yang tercatat di `production_parameters.json`;
3. local-maximum peak extraction pada heatmap;
4. decode box menggunakan `size`, `offset`, dan stride 4;
5. buang prediksi dekat tepi tile sesuai `tile_edge_margin`;
6. ubah koordinat ke ROI global;
7. global NMS;
8. buang pusat box di luar mask cawan;
9. jumlahkan box final.

Contoh decoder dan NMS terdapat di `macos/CenterNetDecoder.swift`. Urutan integrasi dan uji parity ada di `macos/ProductionIntegration.md`.

## Validasi sebelum release

Training selesai belum berarti model siap dilepas. Sebelum release aplikasi:

1. siapkan test set iPhone yang tidak pernah masuk training;
2. jalankan preprocessing Python dan Swift pada gambar yang sama;
3. bandingkan mask, ROI, flat-field, dan tensor input;
4. kalibrasi `score_threshold` dan `global_nms_iou` pada validation iPhone;
5. ukur MAE, latency, peak memory, dan kegagalan plate localization di Mac target;
6. simpan versi model, parameter preprocessing, dan threshold sebagai satu bundle release.

Pipeline ini tidak dapat menghasilkan nilai metrik nyata tanpa file AGAR dan GPU Anda. CSV akan dibuat saat perintah training/evaluasi dijalankan. Jangan mengisi laporan dengan angka dari smoke test; gunakan `production_e2e_test_*` sebagai hasil utama release.

## Referensi

Daftar sumber teknis dan lisensi ada di `SOURCES.md`.
