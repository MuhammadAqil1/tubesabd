# -*- coding: utf-8 -*-
"""
run_all.py -- Pipeline Lengkap Water Potability Analysis
Jalankan: python run_all.py
"""

import os, sys, time, warnings
warnings.filterwarnings('ignore')

# Set stdout encoding agar aman di Windows PowerShell
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Pastikan folder output ada ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'water_potability.csv')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
MODEL_DIR   = os.path.join(OUTPUT_DIR, 'model')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

print("=" * 65)
print("  TUGAS BESAR ABD - KELOMPOK 1")
print("  Prediksi Potabilitas Air Sungai (Random Forest + PySpark)")
print("=" * 65)
print()
print("Base dir  : " + BASE_DIR)
print("Dataset   : " + DATA_PATH)
print("Output    : " + OUTPUT_DIR)
print()

# ── FASE 1: Inisialisasi SparkSession ─────────────────────────────────────────
print("-" * 65)
print("[FASE 1] Inisialisasi SparkSession ...")
print("-" * 65)

os.environ['PYSPARK_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean as spark_mean, count, when, isnan

spark = SparkSession.builder \
    .appName('WaterPotability_Pipeline') \
    .master('local[*]') \
    .config('spark.driver.memory', '2g') \
    .config('spark.sql.shuffle.partitions', '4') \
    .config('spark.ui.showConsoleProgress', 'false') \
    .getOrCreate()

spark.sparkContext.setLogLevel('ERROR')
import pyspark
print("OK - SparkSession aktif | PySpark " + pyspark.__version__ + " | local[*]")

# ── FASE 2: Baca Dataset ──────────────────────────────────────────────────────
print()
print("-" * 65)
print("[FASE 2] Membaca Dataset ...")
print("-" * 65)

df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
total_rows = df.count()

print("OK - Dataset berhasil dibaca")
print("   Jumlah baris  : " + str(total_rows))
print("   Jumlah kolom  : " + str(len(df.columns)))
print("   Kolom         : " + str(df.columns))
print()
df.show(3, truncate=False)

# ── FASE 3: EDA ───────────────────────────────────────────────────────────────
print()
print("-" * 65)
print("[FASE 3] Exploratory Data Analysis (EDA) ...")
print("-" * 65)

import pandas as pd
pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

# Statistik deskriptif
stats = df.describe().toPandas().set_index('summary').T
print()
print("=== STATISTIK DESKRIPTIF ===")
print(stats[['mean','stddev','min','max']].to_string())

# Missing values
print()
print("=== MISSING VALUES ===")
missing = df.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c)
    for c in df.columns
]).collect()[0].asDict()

for col_name, cnt in missing.items():
    pct  = cnt / total_rows * 100
    flag = ' [PERLU IMPUTASI]' if cnt > 0 else ' [LENGKAP]'
    print("  {:<25} {:>6,}  ({:.1f}%){}".format(col_name, cnt, pct, flag))

# Distribusi label
print()
print("=== DISTRIBUSI LABEL ===")
label_dist = df.groupBy('Potability').count().orderBy('Potability').toPandas()
for _, row in label_dist.iterrows():
    label = 'Layak (1)' if row['Potability'] == 1 else 'Tidak Layak (0)'
    pct   = row['count'] / total_rows * 100
    print("  {:<20}: {:,} sampel ({:.1f}%)".format(label, row['count'], pct))

# ── FASE 4: Visualisasi EDA ───────────────────────────────────────────────────
print()
print("-" * 65)
print("[FASE 4] Membuat Visualisasi EDA ...")
print("-" * 65)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df_pd = df.toPandas()

# 4a. Heatmap Korelasi
plt.figure(figsize=(12, 9))
corr = df_pd.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            square=True, linewidths=0.5, annot_kws={'size': 9})
plt.title('Heatmap Korelasi -- Water Potability Dataset', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
path_heatmap = os.path.join(FIGURES_DIR, 'heatmap_korelasi.png')
plt.savefig(path_heatmap, dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: " + path_heatmap)

# 4b. Distribusi Label
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = ['#e74c3c', '#27ae60']
axes[0].bar(['Tidak Layak (0)', 'Layak (1)'],
            label_dist['count'].tolist(), color=colors, edgecolor='black')
axes[0].set_title('Distribusi Kelas Potabilitas', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Jumlah Sampel')
for i, (v, p) in enumerate(zip(label_dist['count'], label_dist['count']/total_rows*100)):
    axes[0].text(i, v+10, '{:,}\n({:.1f}%)'.format(v, p), ha='center', fontsize=10)
axes[1].pie(label_dist['count'], labels=['Tidak Layak (0)', 'Layak (1)'],
            autopct='%1.1f%%', colors=colors, startangle=90, shadow=True)
axes[1].set_title('Proporsi Kelas', fontsize=13, fontweight='bold')
plt.tight_layout()
path_label = os.path.join(FIGURES_DIR, 'distribusi_label.png')
plt.savefig(path_label, dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: " + path_label)

# 4c. Density Plot semua fitur
FEATURE_COLS = ['ph','Hardness','Solids','Chloramines','Sulfate',
                'Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, feat in enumerate(FEATURE_COLS):
    for cls, clr, lbl in [(0,'#e74c3c','Tidak Layak (0)'), (1,'#27ae60','Layak (1)')]:
        subset = df_pd[df_pd['Potability']==cls][feat].dropna()
        axes[i].hist(subset, bins=40, alpha=0.5, density=True,
                     color=clr, label=lbl, edgecolor='none')
    axes[i].set_title('Distribusi: ' + feat, fontsize=11, fontweight='bold')
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel('Density')
    axes[i].legend(fontsize=8)
    axes[i].grid(axis='y', alpha=0.3)
fig.suptitle('Distribusi Fitur per Kelas Potabilitas', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
path_dist = os.path.join(FIGURES_DIR, 'distribusi_fitur.png')
plt.savefig(path_dist, dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: " + path_dist)

# ── FASE 5: Preprocessing ─────────────────────────────────────────────────────
print()
print("-" * 65)
print("[FASE 5] Preprocessing & Feature Engineering ...")
print("-" * 65)

from pyspark.ml.feature import VectorAssembler, StandardScaler

# Imputasi mean
cols_null = ['ph', 'Sulfate', 'Trihalomethanes']
fill_vals = {}
for c in cols_null:
    mean_val = df.select(spark_mean(col(c))).collect()[0][0]
    fill_vals[c] = mean_val
    print("  Imputasi {:<20}: mean = {:.4f}".format(c, mean_val))
df = df.fillna(fill_vals)

# Hapus duplikat
n_before = df.count()
df = df.dropDuplicates()
n_after  = df.count()
print()
print("  Duplikat dihapus  : " + str(n_before - n_after) + " baris")
print("  Dataset bersih    : {:,} baris".format(n_after))

# VectorAssembler + StandardScaler
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol='features_raw', handleInvalid='keep')
scaler    = StandardScaler(inputCol='features_raw', outputCol='features', withStd=True, withMean=True)
df_asm    = assembler.transform(df)
sc_model  = scaler.fit(df_asm)
df_scaled = sc_model.transform(df_asm).select('features', 'Potability')

# Split 80/20
train_df, test_df = df_scaled.randomSplit([0.8, 0.2], seed=42)
print()
print("  Training set : {:,} sampel (80%)".format(train_df.count()))
print("  Testing set  : {:,} sampel (20%)".format(test_df.count()))
print("  OK - Preprocessing selesai")

# ── FASE 6: Training Model ────────────────────────────────────────────────────
print()
print("-" * 65)
print("[FASE 6] Training Random Forest Classifier ...")
print("-" * 65)

from pyspark.ml.classification import (
    RandomForestClassifier, DecisionTreeClassifier,
    LogisticRegression, GBTClassifier
)

rf = RandomForestClassifier(
    labelCol='Potability', featuresCol='features',
    numTrees=100, maxDepth=10, seed=42
)
print("  numTrees=100 | maxDepth=10 | seed=42")
t0 = time.time()
rf_model = rf.fit(train_df)
elapsed  = time.time() - t0
print("  OK - Training selesai dalam {:.1f} detik".format(elapsed))

# ── FASE 7: Evaluasi ──────────────────────────────────────────────────────────
print()
print("-" * 65)
print("[FASE 7] Evaluasi Model ...")
print("-" * 65)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

predictions = rf_model.transform(test_df)
ev = MulticlassClassificationEvaluator(labelCol='Potability', predictionCol='prediction')

accuracy  = ev.evaluate(predictions, {ev.metricName: 'accuracy'})
f1        = ev.evaluate(predictions, {ev.metricName: 'f1'})
precision = ev.evaluate(predictions, {ev.metricName: 'weightedPrecision'})
recall    = ev.evaluate(predictions, {ev.metricName: 'weightedRecall'})
auc_roc   = BinaryClassificationEvaluator(
                labelCol='Potability', rawPredictionCol='rawPrediction',
                metricName='areaUnderROC').evaluate(predictions)

print()
print("  HASIL EVALUASI - RANDOM FOREST")
print("  " + "-" * 50)
targets = [('Accuracy', accuracy, 0.95), ('F1-Score', f1, 0.94),
           ('Precision', precision, 0.94), ('Recall', recall, 0.94), ('AUC-ROC', auc_roc, 0.90)]
for name, val, target in targets:
    status = 'TERCAPAI' if val >= target else 'target >= {}'.format(target)
    print("  {:<12}: {:.4f}  ({:.2f}%)  [{}]".format(name, val, val*100, status))
print("  " + "-" * 50)

# ── FASE 8: Confusion Matrix & Feature Importance ─────────────────────────────
print()
print("-" * 65)
print("[FASE 8] Confusion Matrix & Feature Importance ...")
print("-" * 65)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = [int(r.Potability) for r in predictions.select('Potability').collect()]
y_pred = [int(r.prediction) for r in predictions.select('prediction').collect()]
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(cm, display_labels=['Tidak Layak','Layak']).plot(cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix -- Random Forest Classifier', fontweight='bold')
plt.tight_layout()
path_cm = os.path.join(FIGURES_DIR, 'confusion_matrix.png')
plt.savefig(path_cm, dpi=150, bbox_inches='tight')
plt.close()
print("  TN={} | FP={} | FN={} | TP={}".format(tn, fp, fn, tp))
print("  Saved: " + path_cm)

# Feature Importance
importances = rf_model.featureImportances.toArray()
feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
print()
print("  === FEATURE IMPORTANCE (diurutkan) ===")
for rank, (feat, imp) in enumerate(feat_imp, 1):
    bar = '#' * int(imp * 200)
    print("  {}. {:<20} {:.4f}  {}".format(rank, feat, imp, bar))

colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(FEATURE_COLS))[::-1])
fig, ax = plt.subplots(figsize=(10, 6))
feats_s = [x[0] for x in feat_imp[::-1]]
vals_s  = [x[1] for x in feat_imp[::-1]]
bars = ax.barh(feats_s, vals_s, color=colors_fi, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, vals_s):
    ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
            '{:.4f}'.format(val), va='center', fontsize=9)
ax.set_xlabel('Feature Importance Score', fontsize=11)
ax.set_title('Feature Importance -- Random Forest Classifier', fontsize=13, fontweight='bold')
ax.set_xlim(0, max(vals_s)*1.2)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
path_fi = os.path.join(FIGURES_DIR, 'feature_importance.png')
plt.savefig(path_fi, dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: " + path_fi)

# ── FASE 9: Perbandingan Baseline ─────────────────────────────────────────────
print()
print("-" * 65)
print("[FASE 9] Perbandingan dengan Baseline Model ...")
print("-" * 65)

print("  Training Decision Tree ...")
dt_model  = DecisionTreeClassifier(labelCol='Potability', featuresCol='features', maxDepth=10, seed=42).fit(train_df)
print("  Training Logistic Regression ...")
lr_model  = LogisticRegression(labelCol='Potability', featuresCol='features', maxIter=100).fit(train_df)
print("  Training Gradient Boosting ...")
gbt_model = GBTClassifier(labelCol='Potability', featuresCol='features', maxIter=50, seed=42).fit(train_df)

all_models = {
    'Random Forest': rf_model,
    'Decision Tree': dt_model,
    'Logistic Regression': lr_model,
    'Gradient Boosting': gbt_model,
}

results = []
for name, model in all_models.items():
    preds = model.transform(test_df)
    acc = ev.evaluate(preds, {ev.metricName: 'accuracy'})
    f1_ = ev.evaluate(preds, {ev.metricName: 'f1'})
    pre = ev.evaluate(preds, {ev.metricName: 'weightedPrecision'})
    rec = ev.evaluate(preds, {ev.metricName: 'weightedRecall'})
    results.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1_, 'Precision': pre, 'Recall': rec})

results_df = pd.DataFrame(results)
best_name  = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']

print()
print("  === TABEL PERBANDINGAN MODEL ===")
print("  {:<22} {:>10} {:>10} {:>10} {:>10}".format('Model','Accuracy','F1-Score','Precision','Recall'))
print("  " + "-" * 65)
for _, row in results_df.iterrows():
    marker = ' <- TERBAIK' if row['Model'] == best_name else ''
    print("  {:<22} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}{}".format(
        row['Model'], row['Accuracy'], row['F1-Score'], row['Precision'], row['Recall'], marker))

# Plot perbandingan
metrics_list = ['Accuracy','F1-Score','Precision','Recall']
model_names  = results_df['Model'].tolist()
x = np.arange(len(metrics_list))
width = 0.18
colors_m = ['#2ecc71','#e74c3c','#3498db','#f39c12']

fig, ax = plt.subplots(figsize=(14, 7))
for i, (mname, clr) in enumerate(zip(model_names, colors_m)):
    vals   = [results_df[results_df['Model']==mname][m].values[0] for m in metrics_list]
    offset = (i - len(model_names)/2 + 0.5) * width
    bars_m = ax.bar(x + offset, vals, width, label=mname, color=clr, alpha=0.85,
                    edgecolor='black', linewidth=0.5)
    for b, v in zip(bars_m, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                '{:.3f}'.format(v), ha='center', va='bottom', fontsize=7.5, rotation=45)
ax.set_xlabel('Metrik Evaluasi', fontsize=12)
ax.set_ylabel('Nilai Metrik', fontsize=12)
ax.set_title('Perbandingan Performa Model -- Water Potability Prediction',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_list, fontsize=11)
ax.set_ylim(0, 1.15)
ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1, label='Target (0.95)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
path_cmp = os.path.join(FIGURES_DIR, 'perbandingan_model.png')
plt.savefig(path_cmp, dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: " + path_cmp)

# ── SELESAI ───────────────────────────────────────────────────────────────────
spark.stop()

print()
print("=" * 65)
print("  PIPELINE SELESAI!")
print("=" * 65)
print()
print("Semua output tersimpan di: " + OUTPUT_DIR)
print()
print("File yang dihasilkan:")
for f in sorted(os.listdir(FIGURES_DIR)):
    fpath = os.path.join(FIGURES_DIR, f)
    print("  [PNG] {}  ({} KB)".format(f, os.path.getsize(fpath)//1024))

print()
print("Ringkasan Hasil Evaluasi:")
print("  Accuracy  : {:.4f} ({:.2f}%)".format(accuracy, accuracy*100))
print("  F1-Score  : {:.4f}  ({:.2f}%)".format(f1, f1*100))
print("  Precision : {:.4f} ({:.2f}%)".format(precision, precision*100))
print("  Recall    : {:.4f}  ({:.2f}%)".format(recall, recall*100))
print("  AUC-ROC   : {:.4f}  ({:.2f}%)".format(auc_roc, auc_roc*100))
print()
