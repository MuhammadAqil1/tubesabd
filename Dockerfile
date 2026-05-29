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
