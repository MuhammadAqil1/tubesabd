# Panduan Setup Tugas Besar ABD — Prediksi Potabilitas Air Sungai
> Random Forest Classifier · PySpark MLlib · Jupyter Notebook · Docker
>
> **Kelompok 1** | Dataset: `water_potability.csv` (3.276 sampel, 10 fitur fisikokimia)

---

## Daftar Isi

1. [Prasyarat](#1-prasyarat)
2. [Persiapan WSL2 & Docker Desktop](#2-persiapan-wsl2--docker-desktop)
3. [Clone Repositori & Struktur Folder](#3-clone-repositori--struktur-folder)
4. [Membuat Semua File Konfigurasi dari WSL](#4-membuat-semua-file-konfigurasi-dari-wsl)
5. [Membangun Docker Image](#5-membangun-docker-image)
6. [Menjalankan Kontainer](#6-menjalankan-kontainer)
7. [Verifikasi Kontainer & Layanan](#7-verifikasi-kontainer--layanan)
8. [Mengakses Jupyter Lab & Spark UI](#8-mengakses-jupyter-lab--spark-ui)
9. [Mengunggah Dataset ke Kontainer](#9-mengunggah-dataset-ke-kontainer)
10. [Menjalankan Notebook Analisis](#10-menjalankan-notebook-analisis)
11. [Menghentikan Kontainer](#11-menghentikan-kontainer)
12. [Checklist Sebelum Pengumpulan](#12-checklist-sebelum-pengumpulan)

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

---

## 2. Persiapan WSL2 & Docker Desktop

### 2.1 Instalasi WSL2 (jika belum ada)

Buka **PowerShell sebagai Administrator** lalu jalankan:

```powershell
wsl --install -d Ubuntu-22.04
```

Tunggu hingga selesai, lalu restart komputer jika diminta. Setelah restart, Ubuntu akan meminta membuat username dan password baru — isi sesuai keinginan.

Verifikasi WSL sudah berjalan:

```powershell
wsl --list --verbose
```

Output yang diharapkan:
```
  NAME            STATE           VERSION
* Ubuntu-22.04    Running         2
```

### 2.2 Aktifkan Integrasi Docker dengan WSL2

1. Buka **Docker Desktop**
2. Klik **Settings** (ikon roda gigi)
3. Pilih **Resources** → **WSL Integration**
4. Aktifkan toggle **Ubuntu-22.04**
5. Klik **Apply & Restart**

Verifikasi Docker dapat diakses dari WSL (jalankan dari terminal Ubuntu):

```bash
docker --version
docker compose version
```

Output yang diharapkan:
```
Docker version 24.x.x, build ...
Docker Compose version v2.x.x
```

### 2.3 Update Paket Ubuntu (Opsional tapi Disarankan)

Jalankan perintah berikut dari terminal **WSL Ubuntu**:

```bash
sudo apt-get update && sudo apt-get upgrade -y
```

---

## 3. Clone Repositori & Struktur Folder

### 3.1 Clone dari GitHub

Buka terminal **WSL Ubuntu**, lalu navigasikan ke direktori kerja dan clone repositori:

```bash
# Pindah ke drive C (lokasi Windows)
cd /mnt/c/Users

# Buat folder kerja (sesuaikan nama user Windows Anda)
mkdir -p "Muhammad Aqil/tubesabd-kerja"
cd "Muhammad Aqil/tubesabd-kerja"

# Clone repositori
git clone https://github.com/MuhammadAqil1/tubesabd.git
cd tubesabd
```

> **Catatan:** Jika sudah clone, cukup masuk ke direktori:
> ```bash
> cd /mnt/c/Users/Muhammad\ Aqil/tubesabd-kerja/tubesabd
> ```

### 3.2 Buat Direktori yang Dibutuhkan

```bash
# Buat folder output (jika belum ada)
mkdir -p notebooks output/figures output/model scripts
```

### 3.3 Struktur Folder Lengkap

Setelah clone, struktur folder akan terlihat seperti ini:

```
tubesabd/                              ← root repositori
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

> **Catatan line ending:** Jika ada masalah dengan file, pastikan tidak ada CRLF:
> ```bash
> find . -name "*.py" -exec sed -i 's/\r//' {} \;
> ```

---

## 4. Membuat Semua File Konfigurasi dari WSL

> Jika sudah clone dari GitHub, **semua file ini sudah ada** — lewati ke [Langkah 5](#5-membangun-docker-image).
> Bagian ini hanya diperlukan jika Anda menyiapkan dari awal tanpa clone.

Jalankan semua perintah berikut dari dalam direktori `tubesabd/` di terminal WSL Ubuntu.

### 4.1 `Dockerfile`

```bash
cat > Dockerfile << 'EOF'
FROM jupyter/pyspark-notebook:spark-3.5.0

# Gunakan root untuk instalasi sistem
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

### 4.2 `requirements.txt`

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

### 4.3 `docker-compose.yml`

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
      - "8888:8888"    # Jupyter Lab / Notebook
      - "4040:4040"    # Spark UI (job pertama)
      - "4041:4041"    # Spark UI (job kedua, jika 4040 bentrok)
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

Verifikasi semua file sudah terbuat:

```bash
ls -lh
```

Output yang diharapkan:
```
-rw-r--r-- 1 ... Dockerfile
-rw-r--r-- 1 ... README.md
-rw-r--r-- 1 ... docker-compose.yml
-rw-r--r-- 1 ... requirements.txt
-rw-r--r-- 1 ... water_potability.csv
drwxr-xr-x 2 ... notebooks/
drwxr-xr-x 3 ... output/
drwxr-xr-x 2 ... scripts/
```

---

## 5. Membangun Docker Image

> **Catatan:** Proses build pertama kali membutuhkan waktu **10–20 menit** karena
> mengunduh base image Jupyter+Spark (~1.5 GB). Pastikan koneksi internet stabil.
> Proses ini hanya dilakukan **sekali** — build berikutnya jauh lebih cepat.

Jalankan dari terminal WSL Ubuntu di dalam direktori `tubesabd/`:

```bash
docker compose build
```

Anda akan melihat progress seperti ini (normal):
```
[+] Building 45.2s (12/12) FINISHED
 => [internal] load build definition from Dockerfile
 => [internal] load .dockerignore
 => [1/5] FROM docker.io/jupyter/pyspark-notebook:spark-3.5.0
 => [2/5] RUN apt-get update ...
 => [3/5] COPY requirements.txt /tmp/requirements.txt
 => [4/5] RUN pip install --no-cache-dir -r /tmp/requirements.txt
 => [5/5] RUN mkdir -p /home/jovyan/work/...
 => exporting to image
```

Verifikasi image berhasil dibuat:

```bash
docker images | grep abedeh
```

Output yang diharapkan:
```
REPOSITORY                        TAG       IMAGE ID       CREATED         SIZE
tubesabd-spark-jupyter            latest    a1b2c3d4e5f6   1 minute ago    3.2GB
```

---

## 6. Menjalankan Kontainer

```bash
# Jalankan kontainer di background (-d = detached mode)
docker compose up -d
```

Cek status kontainer (tunggu hingga status `running`):

```bash
docker compose ps
```

Output yang diharapkan:
```
NAME                    IMAGE                    COMMAND    SERVICE         STATUS    PORTS
abedeh-spark-jupyter    tubesabd-spark-jupyter   ...        spark-jupyter   running   0.0.0.0:8888->8888/tcp, 0.0.0.0:4040->4040/tcp
```

Lihat log untuk memastikan Jupyter sudah siap:

```bash
docker compose logs spark-jupyter
```

Tunggu hingga muncul baris seperti ini (tanda kontainer siap):
```
[I 2024-xx-xx xx:xx:xx ServerApp] Jupyter Server is running at:
[I 2024-xx-xx xx:xx:xx ServerApp] http://127.0.0.1:8888/lab?token=abedeh2024
```

Untuk melihat log secara real-time (tekan `Ctrl+C` untuk keluar dari log, **kontainer tetap berjalan**):

```bash
docker compose logs -f spark-jupyter
```

---

## 7. Verifikasi Kontainer & Layanan

### 7.1 Cek Status Kontainer

```bash
docker ps
```

Output yang diharapkan:
```
CONTAINER ID   IMAGE                     STATUS         PORTS
xxxxxxxxxxxx   tubesabd-spark-jupyter    Up 2 minutes   0.0.0.0:8888->8888/tcp
```

### 7.2 Masuk ke Dalam Kontainer (Opsional)

Jika perlu menjalankan perintah langsung di dalam kontainer:

```bash
docker exec -it abedeh-spark-jupyter bash
```

Untuk keluar dari kontainer tanpa menghentikannya:

```bash
exit
```

### 7.3 Verifikasi Dataset Tersedia di Dalam Kontainer

```bash
docker exec abedeh-spark-jupyter ls -lh /home/jovyan/work/data/
```

Output yang diharapkan:
```
-rw-r--r-- 1 jovyan users 513K ... water_potability.csv
```

### 7.4 Verifikasi Python & PySpark Berfungsi

```bash
docker exec abedeh-spark-jupyter python3 -c "
import pyspark
print('PySpark version:', pyspark.__version__)
print('PySpark OK!')
"
```

Output yang diharapkan:
```
PySpark version: 3.5.0
PySpark OK!
```

---

## 8. Mengakses Jupyter Lab & Spark UI

Buka browser dan akses layanan berikut:

| Layanan | URL | Token / Login |
|---|---|---|
| **Jupyter Lab** | http://localhost:8888 | Token: `abedeh2024` |
| **Spark UI** | http://localhost:4040 | — (aktif saat SparkSession berjalan) |

> **Catatan Spark UI:** Port 4040 hanya aktif **selama SparkSession berjalan** di dalam notebook.
> Jika port 4040 sudah terpakai proses lain, Spark otomatis menggunakan port 4041.

**Cara masuk ke Jupyter Lab:**
1. Buka http://localhost:8888 di browser
2. Ketik token: `abedeh2024` pada kolom Password / Token
3. Klik **Log in**
4. Anda akan masuk ke tampilan **Jupyter Lab**

---

## 9. Mengunggah Dataset ke Kontainer

Dataset `water_potability.csv` sudah otomatis ter-mount ke dalam kontainer melalui konfigurasi `volumes` di `docker-compose.yml`. **Tidak perlu upload manual.**

Verifikasi dari terminal Jupyter Lab (klik ikon `+` di panel kiri → **Terminal**):

```bash
# Di dalam terminal Jupyter Lab
ls -lh /home/jovyan/work/data/
```

Atau verifikasi langsung dari WSL Ubuntu:

```bash
docker exec abedeh-spark-jupyter ls -lh /home/jovyan/work/data/
```

---

## 10. Menjalankan Notebook Analisis

Buka Jupyter Lab di http://localhost:8888, lalu masuk ke folder `notebooks/` pada panel kiri. Jalankan notebook secara **berurutan dari atas ke bawah**:

### Urutan Eksekusi Notebook

| No | File Notebook | Isi | Estimasi Waktu |
|---|---|---|---|
| 1 | `01_eda.ipynb` | EDA, statistik deskriptif, heatmap korelasi, density plot | 5–10 menit |
| 2 | `02_preprocessing.ipynb` | Imputasi NULL, normalisasi, split 80/20 | 3–5 menit |
| 3 | `03_modeling.ipynb` | Training Random Forest, evaluasi, confusion matrix, feature importance | 10–15 menit |
| 4 | `04_visualisasi.ipynb` | Perbandingan baseline, tabel metrik, kesimpulan | 10–15 menit |

> **Penting:** Jalankan setiap cell dari **atas ke bawah** menggunakan `Shift+Enter`
> atau klik tombol **▶ Run All** di menu atas. Jangan lewati cell manapun.

### Cara Membuka dan Menjalankan Notebook

1. Di panel kiri Jupyter Lab, klik folder **`notebooks/`**
2. Double-klik `01_eda.ipynb` untuk membukanya
3. Klik menu **Run** → **Run All Cells**
4. Tunggu semua cell selesai (tanda: nomor di `[ ]` sudah terisi, bukan `[*]`)
5. Setelah selesai, ulangi untuk notebook berikutnya

### Memonitor Spark Jobs (Opsional)

Saat notebook sedang berjalan (setelah `SparkSession` dibuat), buka:
- **http://localhost:4040** → tab **Jobs** untuk melihat progress
- Tab **SQL** untuk query yang sedang dieksekusi
- Tab **Executors** untuk monitoring resource

### Menjalankan Pipeline Sekaligus (Alternatif)

Jika ingin menjalankan seluruh pipeline tanpa membuka notebook satu per satu:

```bash
# Masuk ke dalam kontainer
docker exec -it abedeh-spark-jupyter bash

# Jalankan skrip pipeline lengkap
python3 /home/jovyan/work/scripts/run_pipeline.py

# Keluar dari kontainer
exit
```

### Melihat Hasil Output

Semua gambar output tersimpan otomatis di folder `output/figures/` yang dapat diakses langsung dari Windows Explorer:

```
C:\Users\Muhammad Aqil\tubesabd-kerja\tubesabd\output\figures\
├── distribusi_label.png       ← distribusi kelas potabilitas
├── heatmap_korelasi.png       ← heatmap korelasi fitur
├── distribusi_fitur.png       ← density plot per fitur
├── confusion_matrix.png       ← confusion matrix hasil prediksi
├── feature_importance.png     ← bar chart feature importance
└── perbandingan_model.png     ← perbandingan semua model
```

---

## 11. Menghentikan Kontainer

Setelah selesai bekerja, hentikan kontainer dengan perintah berikut dari terminal WSL Ubuntu:

```bash
# Hentikan kontainer (semua file notebook & output tetap tersimpan)
docker compose down
```

Output yang diharapkan:
```
[+] Running 2/2
 ✔ Container abedeh-spark-jupyter  Removed
 ✔ Network abedeh-net              Removed
```

> **Catatan:** Perintah `docker compose down` **tidak** menghapus file Anda.
> Semua notebook dan output di folder `notebooks/` dan `output/` tetap aman.

Untuk menjalankan kembali di lain waktu, cukup:

```bash
# Masuk ke direktori proyek
cd /mnt/c/Users/Muhammad\ Aqil/tubesabd-kerja/tubesabd

# Jalankan kembali (tidak perlu build ulang)
docker compose up -d
```

---

## 12. Checklist Sebelum Pengumpulan

Pastikan semua item berikut terpenuhi sebelum mengumpulkan tugas:

**Notebook & Kode:**
- [ ] `01_eda.ipynb` sudah dijalankan dari awal hingga akhir tanpa error
- [ ] `02_preprocessing.ipynb` sudah dijalankan dari awal hingga akhir tanpa error
- [ ] `03_modeling.ipynb` sudah dijalankan dari awal hingga akhir tanpa error
- [ ] `04_visualisasi.ipynb` sudah dijalankan dari awal hingga akhir tanpa error
- [ ] Semua cell output masih tampil (jangan *Clear Output* sebelum dikumpulkan)

**Output & Visualisasi:**
- [ ] `output/figures/heatmap_korelasi.png` tersedia
- [ ] `output/figures/distribusi_fitur.png` tersedia
- [ ] `output/figures/confusion_matrix.png` tersedia
- [ ] `output/figures/feature_importance.png` tersedia
- [ ] `output/figures/perbandingan_model.png` tersedia

**Evaluasi Model:**
- [ ] Metrik Accuracy, F1-Score, Precision, Recall sudah tercetak di Notebook 3
- [ ] Feature importance sudah dianalisis dan diinterpretasikan
- [ ] Perbandingan dengan model baseline (Decision Tree, Logistic Regression) sudah dilakukan di Notebook 4

**Repositori GitHub:**
- [ ] Semua file sudah ter-push ke https://github.com/MuhammadAqil1/tubesabd
- [ ] `water_potability.csv` tersedia di repo
- [ ] `README.md` ini dapat dibaca dengan jelas

---

## Referensi

- [Apache Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Jupyter PySpark Docker Image](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-pyspark-notebook)
- [Water Quality Dataset — Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- Alomani et al. (2022). *Prediction of Quality of Water According to a Random Forest Classifier.* IJACSA, vol. 13, no. 6, pp. 892–899.
- Fatristya et al. (2025). *Peran Air Bersih dan Sanitasi dalam Meningkatkan Kualitas Hidup.* GeoScienceEd, vol. 6, no. 1.
