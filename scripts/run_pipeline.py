#!/usr/bin/env python3
"""
run_pipeline.py — Pipeline Lengkap (Non-Interaktif)
Menjalankan seluruh tahapan analisis dari preprocessing hingga evaluasi.
Gunakan dengan: spark-submit /home/jovyan/work/scripts/run_pipeline.py
"""

import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean as spark_mean, count, when, isnan
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib
matplotlib.use('Agg')  # Backend non-interaktif untuk server
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# ─── Konfigurasi ───────────────────────────────────────────────────────────────
DATA_PATH    = '/home/jovyan/work/data/water_potability.csv'
OUTPUT_DIR   = '/home/jovyan/work/output'
FIGURES_DIR  = f'{OUTPUT_DIR}/figures'
FEATURE_COLS = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
LABEL_COL    = 'Potability'

os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Inisialisasi Spark ────────────────────────────────────────────────────────
print('[1/6] Inisialisasi SparkSession...')
spark = SparkSession.builder \
    .appName('WaterPotability_Pipeline') \
    .master('local[*]') \
    .config('spark.driver.memory', '2g') \
    .getOrCreate()
spark.sparkContext.setLogLevel('WARN')

# ─── Baca Data ─────────────────────────────────────────────────────────────────
print('[2/6] Membaca dataset...')
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
print(f'  Total baris: {df.count():,}')

# ─── Preprocessing ─────────────────────────────────────────────────────────────
print('[3/6] Preprocessing...')
# Imputasi
cols_with_null = ['ph', 'Sulfate', 'Trihalomethanes']
fill_values = {}
for c in cols_with_null:
    mean_val = df.select(spark_mean(col(c))).collect()[0][0]
    fill_values[c] = mean_val
df = df.fillna(fill_values)
df = df.dropDuplicates()

# VectorAssembler + StandardScaler
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol='features_raw', handleInvalid='keep')
scaler    = StandardScaler(inputCol='features_raw', outputCol='features', withStd=True, withMean=True)

df_assembled  = assembler.transform(df)
scaler_model  = scaler.fit(df_assembled)
df_scaled     = scaler_model.transform(df_assembled).select('features', LABEL_COL)

# Split
train_df, test_df = df_scaled.randomSplit([0.8, 0.2], seed=42)
print(f'  Train: {train_df.count():,} | Test: {test_df.count():,}')

# ─── Training ──────────────────────────────────────────────────────────────────
print('[4/6] Training Random Forest Classifier...')
t0 = time.time()
rf = RandomForestClassifier(
    labelCol=LABEL_COL, featuresCol='features',
    numTrees=100, maxDepth=10, seed=42
)
rf_model = rf.fit(train_df)
print(f'  Selesai dalam {time.time()-t0:.1f} detik.')

# ─── Evaluasi ──────────────────────────────────────────────────────────────────
print('[5/6] Evaluasi model...')
predictions = rf_model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol='prediction')

accuracy  = evaluator.evaluate(predictions, {evaluator.metricName: 'accuracy'})
f1        = evaluator.evaluate(predictions, {evaluator.metricName: 'f1'})
precision = evaluator.evaluate(predictions, {evaluator.metricName: 'weightedPrecision'})
recall    = evaluator.evaluate(predictions, {evaluator.metricName: 'weightedRecall'})

print()
print('=' * 50)
print('       HASIL EVALUASI RANDOM FOREST')
print('=' * 50)
print(f'  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)')
print(f'  F1-Score  : {f1:.4f}  ({f1*100:.2f}%)')
print(f'  Precision : {precision:.4f}  ({precision*100:.2f}%)')
print(f'  Recall    : {recall:.4f}  ({recall*100:.2f}%)')
print('=' * 50)

# ─── Visualisasi ───────────────────────────────────────────────────────────────
print('[6/6] Membuat visualisasi...')

# Confusion Matrix
y_true = [int(r.Potability) for r in predictions.select('Potability').collect()]
y_pred = [int(r.prediction) for r in predictions.select('prediction').collect()]
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(cm, display_labels=['Tidak Layak', 'Layak']).plot(cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix — Random Forest', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/confusion_matrix.png', dpi=150)
plt.close()

# Feature Importance
importances = rf_model.featureImportances.toArray()
sorted_idx = np.argsort(importances)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([FEATURE_COLS[i] for i in sorted_idx], importances[sorted_idx], color='steelblue')
ax.set_title('Feature Importance — Random Forest', fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/feature_importance.png', dpi=150)
plt.close()

print(f'  Gambar disimpan di: {FIGURES_DIR}/')

spark.stop()
print('\n✅ Pipeline selesai dijalankan!')
