# Integrasi macOS

Artefak model yang dimasukkan ke Xcode:

- `PlateU2NetP.mlpackage`
- `ColonyResNet50FPN.mlpackage`
- `production_parameters.json`
- `CenterNetDecoder.swift`

## Urutan inferensi

1. Decode foto iPhone ke orientasi pixel yang benar dan RGB 8-bit.
2. Resize salinan gambar ke `plate_input_size × plate_input_size`.
3. Normalisasi channel memakai `imagenet_mean` dan `imagenet_std`.
4. Jalankan `PlateU2NetP`; threshold dengan `plate_threshold`.
5. Ambil connected component terbesar, tutup lubang kecil, dan fit ellipse.
6. Buat square crop dari ellipse dengan `plate_crop_padding_ratio`, lalu resize ke `plate_roi_target_size`.
7. Buat counting mask menggunakan `plate_counting_scale`.
8. Jalankan normalisasi local flat-field dengan parameter di JSON. Implementasi Python pembanding ada pada `agar_pipeline/production.py` dan kode eksperimen repository `normalize_agar_intensity_v1.py`.
9. Buat tile dengan `colony_tile_size` dan `colony_tile_overlap`.
10. Normalisasi setiap tile dengan mean/std ImageNet, lalu jalankan `ColonyResNet50FPN`.
11. Decode `heatmap`, `size`, dan `offset` dengan `CenterNetDecoder.decode`.
12. Jalankan `removeTileEdgeDetections`, gabungkan koordinat global, global NMS, kemudian `keepCentersInsideMask`.
13. Jumlah box tersisa adalah jumlah koloni.

## Uji parity wajib

Sebelum release, jalankan foto yang sama melalui:

```bash
python run_pipeline.py predict --config config.yaml --image sample.jpg
```

Bandingkan output Python dan Swift untuk:

- mask cawan;
- crop ROI;
- counting mask;
- flat-field grayscale;
- jumlah dan koordinat box;
- count akhir.

Perbedaan preprocessing biasanya lebih merusak akurasi dibanding perbedaan kecil FP16. Jangan mengubah resize mode, urutan RGB, mean/std, atau threshold tanpa mengulangi evaluasi validation.

## Output model koloni

Output model sudah dalam rentang yang siap didecode:

- `heatmap`: probabilitas `[0, 1]`;
- `size`: lebar/tinggi dalam satuan feature-map;
- `offset`: offset sub-pixel `[0, 1]`;
- stride feature-map: `colony_output_stride`.

Global NMS memakai `global_nms_iou`, bukan `tile_nms_iou`.
