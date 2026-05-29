# Prediksi Potabilitas Air Sungai — Tugas Besar ABD Kelompok 1
> Random Forest Classifier · PySpark MLlib · Apache Spark · Jupyter Notebook · Docker

---

## Daftar Isi

1. [Prasyarat](#1-prasyarat)
2. [Struktur Folder](#2-struktur-folder)
3. [Membuat Semua File Konfigurasi](#3-membuat-semua-file-konfigurasi)
4. [Membangun Docker Image](#4-membangun-docker-image)
5. [Menjalankan Kontainer](#5-menjalankan-kontainer)
6. [Mengakses Jupyter Notebook](#6-mengakses-jupyter-notebook)
7. [Mengunggah Dataset ke Kontainer](#7-mengunggah-dataset-ke-kontainer)
8. [Menjalankan Analisis](#8-menjalankan-analisis)
9. [Melihat Hasil dan Output](#9-melihat-hasil-dan-output)
10. [Menghentikan Kontainer](#10-menghentikan-kontainer)
11. [Checklist Sebelum Pengumpulan](#11-checklist-sebelum-pengumpulan)

---

## 1. Prasyarat

Pastikan seluruh perangkat lunak berikut sudah terpasang sebelum memulai.

| Perangkat Lunak | Versi Minimum | Cara Cek |
|---|---|---|
| Docker Desktop | 4.25 / Engine 24.0 | `docker --version` |
| Docker Compose | 2.20 | `docker compose version` |
| Git | 2.x | `git --version` |
| WSL2 + Ubuntu 22.04 | — | `wsl --list --verbose` |
| RAM tersedia | 6 GB | Task Manager / `free -h` |

> **Windows:** Semua perintah bash dijalankan dari dalam terminal **WSL Ubuntu**,
> bukan PowerShell. Pastikan Docker Desktop sudah diintegrasikan dengan WSL2
> (Settings → Resources → WSL Integration → aktifkan Ubuntu-22.04).

**Instalasi WSL2 (jika belum ada) — jalankan dari PowerShell Administrator:**

```powershell
wsl --install -d Ubuntu-22.04
```

Setelah selesai, buka terminal Ubuntu dan arahkan ke drive C:

```bash
cd /mnt/c/Users/Muhammad\ Aqil/Downloads/ABEDEH
```

---

## 2. Struktur Folder

Berikut struktur folder lengkap yang akan digunakan:

```
ABEDEH/                                ← root proyek
├── README.md                          ← panduan ini
├── Dockerfile                         ← image PySpark + Jupyter
├── docker-compose.yml                 ← definisi service kontainer
├── requirements.txt                   ← library Python tambahan
├── water_potability.csv               ← dataset utama (3.276 sampel)
├── notebooks/
│   ├── 01_eda.ipynb                   ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb         ← Preprocessing & Feature Engineering
│   ├── 03_modeling.ipynb              ← Training & Evaluasi Random Forest
│   └── 04_visualisasi.ipynb           ← Visualisasi & Perbandingan Baseline
├── output/
│   ├── figures/                       ← grafik hasil analisis (PNG)
│   └── model/                         ← model tersimpan (opsional)
└── scripts/
    └── run_pipeline.py                ← skrip pipeline lengkap (non-interaktif)
```

Buat direktori yang belum ada dari terminal WSL Ubuntu:

```bash
cd /mnt/c/Users/Muhammad\ Aqil/Downloads/ABEDEH
mkdir -p notebooks output/figures output/model scripts
```

---

## 3. Membuat Semua File Konfigurasi

Jalankan perintah berikut satu per satu dari dalam direktori `ABEDEH/`.

### 3.1 `Dockerfile`

```bash
cat > Dockerfile << 'EOF'
FROM jupyter/pyspark-notebook:spark-3.5.0

# Gunakan root untuk instalasi
USER root

ENV DEBIAN_FRONTEND=noninteractive

# Install dependensi sistem tambahan
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Kembali ke user jovyan (default Jupyter)
USER jovyan

# Install library Python yang dibutuhkan
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Buat direktori kerja
RUN mkdir -p /home/jovyan/work/notebooks \
             /home/jovyan/work/output/figures \
             /home/jovyan/work/output/model \
             /home/jovyan/work/scripts \
             /home/jovyan/work/data

WORKDIR /home/jovyan/work

EXPOSE 8888 4040

EOF
```

### 3.2 `requirements.txt`

```bash
cat > requirements.txt << 'EOF'
pandas==2.1.4
matplotlib==3.8.2
seaborn==0.13.2
scikit-learn==1.3.2
numpy==1.26.2
plotly==5.18.0
imbalanced-learn==0.11.0
EOF
```

### 3.3 `docker-compose.yml`

```bash
cat > docker-compose.yml << 'EOF'
version: "3.8"

services:

  # ── Jupyter + PySpark ────────────────────────────────────────
  spark-jupyter:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: abedeh-spark-jupyter
    ports:
      - "8888:8888"    # Jupyter Notebook
      - "4040:4040"    # Spark UI
      - "4041:4041"    # Spark UI (job ke-2)
    environment:
      JUPYTER_ENABLE_LAB: "yes"
      JUPYTER_TOKEN: "abedeh2024"
      SPARK_OPTS: "--driver-java-options=-Xms1g --driver-java-options=-Xmx2g"
    volumes:
      # Mount folder proyek ke dalam kontainer
      - ./notebooks:/home/jovyan/work/notebooks
      - ./output:/home/jovyan/work/output
      - ./scripts:/home/jovyan/work/scripts
      - ./water_potability.csv:/home/jovyan/work/data/water_potability.csv
    restart: unless-stopped
    mem_limit: 4g
    cpus: "2.0"

volumes: {}

networks:
  default:
    name: abedeh-net
    driver: bridge
EOF
```

---

## 4. Membangun Docker Image

> **Catatan:** Proses build pertama kali membutuhkan waktu **10–20 menit** karena
> mengunduh base image Jupyter+Spark (~1.5 GB). Pastikan koneksi internet stabil.

Jalankan dari terminal WSL Ubuntu di direktori `ABEDEH/`:

```bash
# Pindah ke direktori proyek
cd /mnt/c/Users/Muhammad\ Aqil/Downloads/ABEDEH

# Build image (akan terlihat progress download dan instalasi)
docker compose build

# Verifikasi image berhasil dibuat
docker images | grep abedeh
```

Output yang diharapkan (contoh):
```
REPOSITORY                TAG       IMAGE ID       CREATED         SIZE
abedeh-abedeh-spark-jupyter   latest    a1b2c3d4e5f6   2 minutes ago   3.2GB
```

---

## 5. Menjalankan Kontainer

```bash
# Jalankan kontainer di background
docker compose up -d

# Cek status kontainer (tunggu hingga status = healthy/running)
docker compose ps

# Lihat log real-time (Ctrl+C untuk keluar dari log, kontainer tetap berjalan)
docker compose logs -f spark-jupyter
```

Output log yang diharapkan (tanda kontainer siap):
```
...
[I 2024-xx-xx xx:xx:xx ServerApp] Jupyter Server is running at:
[I 2024-xx-xx xx:xx:xx ServerApp] http://127.0.0.1:8888/lab?token=abedeh2024
```

---

## 6. Mengakses Jupyter Notebook

Setelah kontainer berjalan, buka browser dan kunjungi:

| Layanan | URL | Token/Password |
|---|---|---|
| **Jupyter Lab** | http://localhost:8888 | `abedeh2024` |
| **Spark UI** | http://localhost:4040 | — (buka setelah SparkSession aktif) |

> **Catatan:** Spark UI (port 4040) hanya aktif **selama SparkSession berjalan** di dalam notebook.
> Jika port 4040 sudah terpakai, Spark otomatis menggunakan 4041.

---

## 7. Mengunggah Dataset ke Kontainer

Dataset `water_potability.csv` sudah otomatis ter-mount ke dalam kontainer melalui konfigurasi volume di `docker-compose.yml`. Tidak perlu upload manual.

Verifikasi dari dalam Jupyter (buka terminal di Jupyter Lab → klik ikon `+` → Terminal):

```bash
ls -lh /home/jovyan/work/data/
# Output yang diharapkan:
# -rw-r--r-- 1 jovyan users 513K ... water_potability.csv
```

Atau verifikasi langsung dari WSL:

```bash
docker exec abedeh-spark-jupyter ls -lh /home/jovyan/work/data/
```

---

## 8. Menjalankan Analisis

Buka Jupyter Lab di http://localhost:8888, masuk ke folder `notebooks/`, lalu jalankan notebook secara berurutan:

### Urutan Eksekusi Notebook:

| No | File | Isi | Estimasi Waktu |
|---|---|---|---|
| 1 | `01_eda.ipynb` | EDA, statistik deskriptif, heatmap korelasi | 5–10 menit |
| 2 | `02_preprocessing.ipynb` | Imputasi NULL, normalisasi, split data | 5 menit |
| 3 | `03_modeling.ipynb` | Training Random Forest, evaluasi metrik | 10–15 menit |
| 4 | `04_visualisasi.ipynb` | Confusion matrix, feature importance, baseline comparison | 5 menit |

> **Penting:** Jalankan setiap cell dari atas ke bawah secara berurutan
> menggunakan `Shift+Enter` atau tombol ▶ Run All.

### Menjalankan Pipeline Lengkap (Alternatif):

Jika ingin menjalankan seluruh pipeline sekaligus tanpa notebook:

```bash
# Masuk ke dalam kontainer
docker exec -it abedeh-spark-jupyter bash

# Jalankan skrip pipeline
spark-submit /home/jovyan/work/scripts/run_pipeline.py
```

---

## 9. Melihat Hasil dan Output

Semua output (gambar, model) tersimpan otomatis di folder `output/` yang ter-mount ke host:

```
output/
├── figures/
│   ├── heatmap_korelasi.png       ← heatmap korelasi fitur
│   ├── distribusi_fitur.png       ← density plot per fitur
│   ├── confusion_matrix.png       ← confusion matrix hasil prediksi
│   └── feature_importance.png     ← bar chart feature importance
└── model/
    └── random_forest_model/       ← model tersimpan (opsional)
```

File-file ini dapat diakses langsung dari Windows Explorer di:
`C:\Users\Muhammad Aqil\Downloads\ABEDEH\output\`

### Monitoring Spark Jobs:

Saat notebook sedang berjalan (SparkSession aktif), buka:
- **Spark UI:** http://localhost:4040
- Lihat tab **Jobs**, **Stages**, **SQL** untuk monitoring pipeline

---

## 10. Menghentikan Kontainer

```bash
# Hentikan kontainer (data notebook tetap tersimpan di folder lokal)
docker compose down

# Jika ingin menghapus semua resource termasuk volume (HATI-HATI)
# docker compose down -v
```

---

## 11. Checklist Sebelum Pengumpulan

- [ ] Semua 4 notebook sudah dijalankan dari awal hingga akhir tanpa error
- [ ] Output gambar tersimpan di `output/figures/` (minimal 4 gambar)
- [ ] Metrik evaluasi (accuracy, F1, precision, recall) sudah dicetak di notebook
- [ ] Feature importance sudah dianalisis dan divisualisasikan
- [ ] Perbandingan dengan baseline model sudah dicantumkan
- [ ] Semua cell output notebook masih tampil (jangan clear output sebelum kumpul)
- [ ] File `water_potability.csv` tersedia di folder proyek

---

## Referensi

- [Apache Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Water Quality Dataset — Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- Alomani et al. (2022). *Prediction of Quality of Water According to a Random Forest Classifier.* IJACSA, vol. 13, no. 6.
