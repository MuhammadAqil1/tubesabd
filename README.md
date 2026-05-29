# Panduan Setup Tugas Besar ABD — Prediksi Potabilitas Air Sungai
> Random Forest Classifier · PySpark MLlib · Apache Spark + Hadoop · Docker
>
> **Kelompok 1** | Dataset: `water_potability.csv` (3.276 sampel, 10 fitur fisikokimia)

---

## Apakah Perlu Spark dan Hadoop?

| Komponen | Diperlukan? | Keterangan |
|---|---|---|
| **Apache Spark / PySpark** | ✅ Ya, wajib | Inti dari seluruh analisis: DataFrame API, MLlib, Pipeline |
| **Hadoop (HDFS)** | ✅ Ya, ikuti standar praktikum | Digunakan sebagai penyimpanan dataset di dalam cluster |
| **YARN** | ✅ Ya, ikuti standar praktikum | Resource manager bawaan setup dosen |

> Proyek ini menggunakan **lingkungan yang sama** dengan praktikum modul 9 —
> yaitu repositori `bigdata-spark` milik dosen. Notebook tugas besar dijalankan
> di dalam kontainer yang sama.

---

## Daftar Isi

1. [Prasyarat](#1-prasyarat)
2. [Persiapan WSL2 & Docker Desktop](#2-persiapan-wsl2--docker-desktop)
3. [Clone Repositori bigdata-spark (Lingkungan Dosen)](#3-clone-repositori-bigdata-spark-lingkungan-dosen)
4. [Clone Repositori Tugas Besar & Susun Folder](#4-clone-repositori-tugas-besar--susun-folder)
5. [Mengunduh Binary Hadoop & Spark](#5-mengunduh-binary-hadoop--spark)
6. [Membangun Docker Image](#6-membangun-docker-image)
7. [Menjalankan Kontainer](#7-menjalankan-kontainer)
8. [Verifikasi Layanan Hadoop & HDFS](#8-verifikasi-layanan-hadoop--hdfs)
9. [Mengakses Web UI](#9-mengakses-web-ui)
10. [Persiapan Dataset di HDFS](#10-persiapan-dataset-di-hdfs)
11. [Menjalankan Notebook Analisis](#11-menjalankan-notebook-analisis)
12. [Menghentikan Kontainer](#12-menghentikan-kontainer)
13. [Checklist Sebelum Pengumpulan](#13-checklist-sebelum-pengumpulan)

---

## 1. Prasyarat

Pastikan seluruh perangkat lunak berikut sudah terpasang sebelum memulai.

| Perangkat Lunak | Versi Minimum | Cara Cek |
|---|---|---|
| Docker Desktop | 4.25 / Engine 24.0 | `docker --version` |
| Docker Compose | 2.20 | `docker compose version` |
| Git | 2.x | `git --version` |
| WSL2 + Ubuntu 22.04 | — | `wsl --list --verbose` |
| RAM tersedia | 8 GB | Task Manager / `free -h` |

> **Windows:** Semua perintah bash dijalankan dari dalam terminal **WSL Ubuntu**,
> bukan PowerShell. Pastikan Docker Desktop sudah diintegrasikan dengan WSL2
> (Settings → Resources → WSL Integration → aktifkan Ubuntu-22.04).

---

## 2. Persiapan WSL2 & Docker Desktop

### 2.1 Instalasi WSL2 (jika belum ada)

Buka **PowerShell sebagai Administrator**, lalu jalankan:

```powershell
wsl --install -d Ubuntu-22.04
```

Restart komputer jika diminta. Setelah restart, buka Ubuntu dan buat username + password.

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
2. Klik **Settings** (ikon ⚙️) → **Resources** → **WSL Integration**
3. Aktifkan toggle **Ubuntu-22.04**
4. Klik **Apply & Restart**

Verifikasi dari terminal **WSL Ubuntu**:

```bash
docker --version
docker compose version
```

### 2.3 Update Paket Ubuntu

```bash
sudo apt-get update && sudo apt-get upgrade -y
```

---

## 3. Clone Repositori bigdata-spark (Lingkungan Dosen)

Semua perintah berikut dijalankan dari terminal **WSL Ubuntu**.

```bash
# Pindah ke drive C
cd /mnt/c/Users

# Buat folder kerja (ganti "Muhammad Aqil" sesuai nama user Windows Anda)
mkdir -p "Muhammad Aqil/praktikum-abd"
cd "Muhammad Aqil/praktikum-abd"

# Clone repositori lingkungan dosen
git clone https://github.com/sains-data/bigdata-spark.git
cd bigdata-spark
```

Pastikan tidak ada masalah line ending pada script shell:

```bash
find . -name "*.sh" -exec sed -i 's/\r//' {} \;
```

Buat direktori khusus untuk tugas besar:

```bash
mkdir -p tubesabd/notebooks tubesabd/data tubesabd/output/figures tubesabd/output/model tubesabd/scripts
```

---

## 4. Clone Repositori Tugas Besar & Susun Folder

Masih dari dalam direktori `bigdata-spark/`, clone repositori tugas besar kelompok 1:

```bash
# Pastikan masih di dalam bigdata-spark/
pwd
# Output: /mnt/c/Users/Muhammad Aqil/praktikum-abd/bigdata-spark

# Clone isi tugas besar ke folder tubesabd/
git clone https://github.com/MuhammadAqil1/tubesabd.git tubesabd-tmp

# Salin notebook dan dataset ke dalam struktur bigdata-spark
cp tubesabd-tmp/notebooks/*.ipynb tubesabd/notebooks/
cp tubesabd-tmp/water_potability.csv tubesabd/data/
cp tubesabd-tmp/scripts/run_pipeline.py tubesabd/scripts/

# Hapus folder temp
rm -rf tubesabd-tmp
```

Verifikasi struktur folder akhir:

```bash
ls -lh tubesabd/notebooks/
ls -lh tubesabd/data/
```

Struktur folder `bigdata-spark/` yang sudah lengkap:

```
bigdata-spark/                          ← root environment dosen
├── Dockerfile                          ← image Hadoop + Spark (Ubuntu 24.04 + Java 8)
├── bootstrap.sh                        ← inisialisasi environment saat kontainer start
├── start.sh                            ← script menjalankan kontainer
├── stop.sh                             ← script menghentikan kontainer
├── login.sh                            ← script masuk ke kontainer
├── core-site.xml                       ← konfigurasi HDFS
├── hdfs-site.xml                       ← konfigurasi replikasi HDFS
├── hadoop-env.sh                       ← environment Hadoop
├── hadoop-3.4.1.tar.gz                 ← ⬅ diunduh manual (Langkah 5)
├── spark-3.5.x-bin-hadoop3.tgz         ← ⬅ diunduh manual (Langkah 5)
│
└── tubesabd/                           ← direktori tugas besar kelompok 1
    ├── data/
    │   └── water_potability.csv        ← dataset (3.276 sampel)
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   ├── 02_preprocessing.ipynb
    │   ├── 03_modeling.ipynb
    │   └── 04_visualisasi.ipynb
    ├── output/
    │   ├── figures/                    ← output gambar
    │   └── model/                      ← model tersimpan
    └── scripts/
        └── run_pipeline.py
```

---

## 5. Mengunduh Binary Hadoop & Spark

> Hadoop dan Spark tidak disertakan di repositori karena ukurannya besar (~400 MB masing-masing).
> Unduh manual menggunakan perintah berikut dari dalam direktori `bigdata-spark/`.

### 5.1 Unduh Hadoop 3.4.1

```bash
# Pastikan masih di bigdata-spark/
cd /mnt/c/Users/Muhammad\ Aqil/praktikum-abd/bigdata-spark

wget https://downloads.apache.org/hadoop/common/hadoop-3.4.1/hadoop-3.4.1.tar.gz
```

> Jika `wget` gagal karena mirror tidak tersedia, coba link alternatif:
> ```bash
> wget https://archive.apache.org/dist/hadoop/common/hadoop-3.4.1/hadoop-3.4.1.tar.gz
> ```

### 5.2 Unduh Spark 3.5.3

```bash
wget https://downloads.apache.org/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz
```

> Link alternatif jika gagal:
> ```bash
> wget https://archive.apache.org/dist/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz
> ```

Verifikasi kedua file sudah ada:

```bash
ls -lh *.tar.gz *.tgz
```

Output yang diharapkan:
```
-rw-r--r-- 1 ... 697M ... hadoop-3.4.1.tar.gz
-rw-r--r-- 1 ... 400M ... spark-3.5.3-bin-hadoop3.tgz
```

---

## 6. Membangun Docker Image

> Proses build pertama kali membutuhkan waktu **15–30 menit** karena mengekstrak
> Hadoop dan Spark dari tarball. Pastikan koneksi internet stabil dan RAM cukup.

```bash
bash build.sh
```

Atau secara manual:

```bash
docker build -t bigdata-spark .
```

Verifikasi image berhasil dibuat:

```bash
docker images | grep bigdata-spark
```

Output yang diharapkan:
```
REPOSITORY      TAG       IMAGE ID       CREATED         SIZE
bigdata-spark   latest    xxxxxxxxxxxx   2 minutes ago   ~5GB
```

---

## 7. Menjalankan Kontainer

Gunakan script bawaan dosen untuk menjalankan kontainer:

```bash
bash start.sh
```

Script ini akan menjalankan kontainer dengan port-port berikut:

| Port | Layanan |
|---|---|
| `9870` | HDFS NameNode Web UI |
| `9866` | HDFS DataNode |
| `8088` | YARN ResourceManager Web UI |
| `9000` | HDFS RPC |
| `8000` | Spark / Jupyter (jika aktif) |

Cek apakah kontainer sudah berjalan:

```bash
docker ps
```

Output yang diharapkan:
```
CONTAINER ID   IMAGE           STATUS         NAMES
xxxxxxxxxxxx   bigdata-spark   Up 1 minute    bigdata-spark
```

---

## 8. Verifikasi Layanan Hadoop & HDFS

Masuk ke dalam kontainer:

```bash
bash login.sh
```

Atau secara manual:

```bash
docker exec -it bigdata-spark bash
```

Di dalam kontainer, jalankan perintah berikut untuk memastikan semua layanan aktif:

```bash
# Cek status HDFS
hdfs dfsadmin -report
```

Output yang diharapkan (sebagian):
```
Configured Capacity: ...
Live datanodes (1): ...
```

```bash
# Cek YARN
yarn node -list
```

Output yang diharapkan:
```
Total Nodes:1
         Node-Id             Node-State Node-Http-Address  ...
localhost:xxxx          RUNNING localhost:8042             ...
```

```bash
# Verifikasi Spark bisa berjalan
spark-shell --version
```

Keluar dari kontainer:

```bash
exit
```

---

## 9. Mengakses Web UI

Buka browser dan akses layanan berikut:

| Layanan | URL | Keterangan |
|---|---|---|
| **HDFS NameNode UI** | http://localhost:9870 | Melihat status HDFS dan file yang tersimpan |
| **YARN ResourceManager** | http://localhost:8088 | Memonitor Spark jobs yang berjalan |

---

## 10. Persiapan Dataset di HDFS

Salin dataset `water_potability.csv` ke dalam HDFS agar bisa dibaca oleh Spark:

```bash
# Masuk ke kontainer
bash login.sh

# Buat direktori di HDFS
hdfs dfs -mkdir -p /user/tubesabd/data

# Salin dataset dari host ke dalam kontainer lalu ke HDFS
# (dataset sudah ter-mount atau bisa disalin manual)
hdfs dfs -put /tubesabd/data/water_potability.csv /user/tubesabd/data/

# Verifikasi file berhasil masuk ke HDFS
hdfs dfs -ls /user/tubesabd/data/
```

Output yang diharapkan:
```
Found 1 items
-rw-r--r--   1 root supergroup     537810 ... /user/tubesabd/data/water_potability.csv
```

```bash
# Keluar dari kontainer
exit
```

> **Catatan:** Jika dataset belum ada di dalam kontainer, salin terlebih dahulu dari WSL:
> ```bash
> docker cp tubesabd/data/water_potability.csv bigdata-spark:/tubesabd/data/
> ```

---

## 11. Menjalankan Notebook Analisis

### 11.1 Jalankan Jupyter di Dalam Kontainer

```bash
# Masuk ke dalam kontainer
bash login.sh

# Install Jupyter jika belum ada
pip3 install jupyter notebook

# Jalankan Jupyter Notebook (dari dalam kontainer)
jupyter notebook --ip=0.0.0.0 --port=8000 --no-browser --allow-root \
  --NotebookApp.token='abedeh2024' \
  --notebook-dir=/tubesabd/notebooks/ &
```

Buka browser dan akses: **http://localhost:8000** dengan token `abedeh2024`

### 11.2 Urutan Eksekusi Notebook

| No | File Notebook | Isi | Estimasi Waktu |
|---|---|---|---|
| 1 | `01_eda.ipynb` | EDA, statistik deskriptif, heatmap korelasi, density plot | 5–10 menit |
| 2 | `02_preprocessing.ipynb` | Imputasi NULL, normalisasi, split 80/20 | 3–5 menit |
| 3 | `03_modeling.ipynb` | Training Random Forest, evaluasi, confusion matrix | 10–15 menit |
| 4 | `04_visualisasi.ipynb` | Perbandingan baseline, tabel metrik, kesimpulan | 10–15 menit |

> Jalankan setiap cell dari **atas ke bawah** menggunakan `Shift+Enter`.

### 11.3 Alternatif: Jalankan Pipeline Langsung (Tanpa Notebook)

```bash
# Masuk ke kontainer
bash login.sh

# Jalankan pipeline lengkap dengan spark-submit
spark-submit /tubesabd/scripts/run_pipeline.py

# Keluar dari kontainer
exit
```

### 11.4 Monitoring Spark Jobs

Saat pipeline/notebook berjalan, pantau di:
- **YARN UI:** http://localhost:8088 → klik **Applications** → lihat job yang aktif
- **Spark UI:** Klik link **ApplicationMaster** di YARN UI → tampil Spark UI dengan detail stage

---

## 12. Menghentikan Kontainer

```bash
bash stop.sh
```

Atau secara manual:

```bash
docker stop bigdata-spark
```

> Semua file di folder `tubesabd/` di host WSL tetap aman — tidak terhapus saat kontainer dihentikan.

Untuk menjalankan kembali:

```bash
bash start.sh
```

---

## 13. Checklist Sebelum Pengumpulan

**Environment & Setup:**
- [ ] `bigdata-spark` berhasil di-clone dan build
- [ ] Hadoop HDFS aktif (`hdfs dfsadmin -report` menampilkan 1 datanode)
- [ ] YARN aktif (`yarn node -list` menampilkan 1 node)
- [ ] Dataset `water_potability.csv` tersedia di HDFS: `/user/tubesabd/data/`

**Notebook & Kode:**
- [ ] `01_eda.ipynb` dijalankan tanpa error, output tampil
- [ ] `02_preprocessing.ipynb` dijalankan tanpa error, output tampil
- [ ] `03_modeling.ipynb` dijalankan tanpa error, output tampil
- [ ] `04_visualisasi.ipynb` dijalankan tanpa error, output tampil
- [ ] **Jangan** hapus output cell sebelum dikumpulkan

**Output & Visualisasi:**
- [ ] `output/figures/heatmap_korelasi.png` ada
- [ ] `output/figures/confusion_matrix.png` ada
- [ ] `output/figures/feature_importance.png` ada
- [ ] `output/figures/perbandingan_model.png` ada

**Evaluasi Model:**
- [ ] Accuracy, F1-Score, Precision, Recall tercetak di Notebook 3
- [ ] Feature importance dianalisis dan diinterpretasikan
- [ ] Perbandingan dengan baseline (Decision Tree, Logistic Regression) ada di Notebook 4

**Repositori GitHub:**
- [ ] Semua notebook terbaru ter-push ke https://github.com/MuhammadAqil1/tubesabd

---

## Referensi

- [Repositori Lingkungan Dosen (bigdata-spark)](https://github.com/sains-data/bigdata-spark)
- [Repositori Praktikum ABD](https://github.com/sains-data/praktikum-analaisis-big-data)
- [Apache Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [Water Quality Dataset — Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- Alomani et al. (2022). *Prediction of Quality of Water According to a Random Forest Classifier.* IJACSA, vol. 13, no. 6, pp. 892–899.
