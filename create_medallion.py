# -*- coding: utf-8 -*-
"""
create_medallion.py -- Generate Bronze/Silver/Gold layers + Dashboard HTML
Jalankan: python create_medallion.py
"""

import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.environ['PYSPARK_PYTHON'] = sys.executable
JAVA_HOME = r"C:\Program Files\Microsoft\jdk-11.0.31.11-hotspot"
if os.path.exists(JAVA_HOME):
    os.environ['JAVA_HOME'] = JAVA_HOME
    os.environ['PATH'] = JAVA_HOME + r"\bin;" + os.environ.get('PATH', '')

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'water_potability.csv')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'output')
BRONZE_DIR  = os.path.join(OUTPUT_DIR, 'bronze')
SILVER_DIR  = os.path.join(OUTPUT_DIR, 'silver')
GOLD_DIR    = os.path.join(OUTPUT_DIR, 'gold')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

for d in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

print("=" * 65)
print("  MEDALLION PIPELINE - Bronze / Silver / Gold")
print("  Tugas Besar ABD Kelompok 1")
print("=" * 65)

# ===========================================================================
# BRONZE LAYER -- Data Mentah
# ===========================================================================
print("\n[BRONZE] Memuat data mentah ...")

df_raw = pd.read_csv(DATA_PATH)
total_raw = len(df_raw)

bronze_stats = {
    "layer": "Bronze",
    "description": "Raw data langsung dari sumber (Kaggle) tanpa perubahan apapun",
    "total_rows": total_raw,
    "total_columns": len(df_raw.columns),
    "columns": list(df_raw.columns),
    "missing_values": {c: int(df_raw[c].isna().sum()) for c in df_raw.columns},
    "missing_pct":   {c: round(df_raw[c].isna().sum() / total_raw * 100, 2) for c in df_raw.columns},
    "label_distribution": {
        "Tidak Layak (0)": int((df_raw['Potability'] == 0).sum()),
        "Layak (1)":       int((df_raw['Potability'] == 1).sum()),
    },
    "duplicates": int(df_raw.duplicated().sum()),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

df_raw.to_csv(os.path.join(BRONZE_DIR, 'water_potability_raw.csv'), index=False)
with open(os.path.join(BRONZE_DIR, 'bronze_stats.json'), 'w') as f:
    json.dump(bronze_stats, f, indent=2)

print("  Rows       : {:,}".format(total_raw))
print("  Missing ph : {}  Sulfate: {}  Trihalomethanes: {}".format(
    bronze_stats['missing_values']['ph'],
    bronze_stats['missing_values']['Sulfate'],
    bronze_stats['missing_values']['Trihalomethanes']))
print("  Saved: " + BRONZE_DIR)

# ===========================================================================
# SILVER LAYER -- Data Bersih
# ===========================================================================
print("\n[SILVER] Membersihkan data ...")

df_silver = df_raw.copy()

# Imputasi mean
impute_cols = ['ph', 'Sulfate', 'Trihalomethanes']
impute_means = {}
for c in impute_cols:
    mean_val = df_silver[c].mean()
    impute_means[c] = round(mean_val, 4)
    df_silver[c] = df_silver[c].fillna(mean_val)

# Hapus duplikat
n_before = len(df_silver)
df_silver = df_silver.drop_duplicates().reset_index(drop=True)
n_after = len(df_silver)

# Statistik silver
silver_stats = {
    "layer": "Silver",
    "description": "Data setelah imputasi mean, penghapusan duplikat, dan validasi kelengkapan",
    "total_rows": n_after,
    "total_columns": len(df_silver.columns),
    "duplicates_removed": n_before - n_after,
    "imputation_applied": impute_means,
    "missing_after": {c: int(df_silver[c].isna().sum()) for c in df_silver.columns},
    "label_distribution": {
        "Tidak Layak (0)": int((df_silver['Potability'] == 0).sum()),
        "Layak (1)":       int((df_silver['Potability'] == 1).sum()),
    },
    "feature_stats": {
        c: {
            "mean":   round(df_silver[c].mean(), 4),
            "std":    round(df_silver[c].std(), 4),
            "min":    round(df_silver[c].min(), 4),
            "max":    round(df_silver[c].max(), 4),
        }
        for c in df_silver.columns if c != 'Potability'
    },
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

df_silver.to_csv(os.path.join(SILVER_DIR, 'water_potability_clean.csv'), index=False)
with open(os.path.join(SILVER_DIR, 'silver_stats.json'), 'w') as f:
    json.dump(silver_stats, f, indent=2)

print("  Rows setelah cleaning : {:,}".format(n_after))
print("  Imputed means         : ph={:.3f}  Sulfate={:.3f}  THMs={:.3f}".format(
    impute_means['ph'], impute_means['Sulfate'], impute_means['Trihalomethanes']))
print("  Saved: " + SILVER_DIR)

# ===========================================================================
# GOLD LAYER -- Hasil Model & Analitik
# ===========================================================================
print("\n[GOLD] Training model dan menghasilkan analitik akhir ...")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    RandomForestClassifier, DecisionTreeClassifier,
    LogisticRegression, GBTClassifier
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from sklearn.metrics import confusion_matrix

spark = SparkSession.builder \
    .appName('WaterPotability_Gold') \
    .master('local[*]') \
    .config('spark.driver.memory', '2g') \
    .config('spark.sql.shuffle.partitions', '4') \
    .config('spark.ui.showConsoleProgress', 'false') \
    .getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

import pyspark
print("  SparkSession OK | PySpark " + pyspark.__version__)

# Load silver CSV ke Spark
df_spark = spark.read.csv(
    os.path.join(SILVER_DIR, 'water_potability_clean.csv'),
    header=True, inferSchema=True
)

FEATURE_COLS = ['ph','Hardness','Solids','Chloramines','Sulfate',
                'Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
LABEL_COL = 'Potability'

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol='features_raw', handleInvalid='keep')
scaler    = StandardScaler(inputCol='features_raw', outputCol='features', withStd=True, withMean=True)
df_asm    = assembler.transform(df_spark)
sc_model  = scaler.fit(df_asm)
df_scaled = sc_model.transform(df_asm).select('features', LABEL_COL)

train_df, test_df = df_scaled.randomSplit([0.8, 0.2], seed=42)
print("  Train: {:,}  Test: {:,}".format(train_df.count(), test_df.count()))

# Train semua model
t0 = time.time()
rf_model  = RandomForestClassifier(labelCol=LABEL_COL, featuresCol='features', numTrees=100, maxDepth=10, seed=42).fit(train_df)
dt_model  = DecisionTreeClassifier(labelCol=LABEL_COL, featuresCol='features', maxDepth=10, seed=42).fit(train_df)
lr_model  = LogisticRegression(labelCol=LABEL_COL, featuresCol='features', maxIter=100).fit(train_df)
gbt_model = GBTClassifier(labelCol=LABEL_COL, featuresCol='features', maxIter=50, seed=42).fit(train_df)
print("  Training 4 model selesai dalam {:.1f} detik".format(time.time()-t0))

ev = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol='prediction')
ev_bin = BinaryClassificationEvaluator(labelCol=LABEL_COL, rawPredictionCol='rawPrediction', metricName='areaUnderROC')

all_models = {
    'Random Forest': rf_model,
    'Decision Tree': dt_model,
    'Logistic Regression': lr_model,
    'Gradient Boosting': gbt_model,
}

model_results = []
rf_predictions = None

for name, model in all_models.items():
    preds = model.transform(test_df)
    acc = ev.evaluate(preds, {ev.metricName: 'accuracy'})
    f1  = ev.evaluate(preds, {ev.metricName: 'f1'})
    pre = ev.evaluate(preds, {ev.metricName: 'weightedPrecision'})
    rec = ev.evaluate(preds, {ev.metricName: 'weightedRecall'})
    auc = ev_bin.evaluate(preds)
    model_results.append({
        'Model': name, 'Accuracy': round(acc,4), 'F1_Score': round(f1,4),
        'Precision': round(pre,4), 'Recall': round(rec,4), 'AUC_ROC': round(auc,4)
    })
    if name == 'Random Forest':
        rf_predictions = preds
        rf_acc, rf_f1, rf_pre, rf_rec, rf_auc = acc, f1, pre, rec, auc

# Feature importance
feat_imp = sorted(
    zip(FEATURE_COLS, rf_model.featureImportances.toArray()),
    key=lambda x: x[1], reverse=True
)

# Confusion matrix data
y_true = [int(r.Potability) for r in rf_predictions.select('Potability').collect()]
y_pred = [int(r.prediction) for r in rf_predictions.select('prediction').collect()]
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Sample predictions (50 baris)
preds_sample = rf_predictions.select('Potability','prediction','probability').limit(50).toPandas()
preds_sample['prob_layak']     = preds_sample['probability'].apply(lambda v: round(float(v[1]), 4))
preds_sample['prob_tidak_layak'] = preds_sample['probability'].apply(lambda v: round(float(v[0]), 4))
preds_sample = preds_sample.drop('probability', axis=1)
preds_sample.columns = ['Aktual','Prediksi','Prob_Layak','Prob_Tidak_Layak']

# Simpan Gold artifacts
df_compare = pd.DataFrame(model_results)
df_compare.to_csv(os.path.join(GOLD_DIR, 'model_comparison.csv'), index=False)

df_feat_imp = pd.DataFrame(feat_imp, columns=['Feature','Importance'])
df_feat_imp.to_csv(os.path.join(GOLD_DIR, 'feature_importance.csv'), index=False)

preds_sample.to_csv(os.path.join(GOLD_DIR, 'predictions_sample.csv'), index=False)

gold_metrics = {
    "layer": "Gold",
    "description": "Hasil akhir analitik: evaluasi model, feature importance, dan prediksi",
    "best_model": "Random Forest",
    "test_size": len(y_true),
    "train_size": train_df.count(),
    "metrics": {
        "accuracy":  round(rf_acc, 4),
        "f1_score":  round(rf_f1,  4),
        "precision": round(rf_pre, 4),
        "recall":    round(rf_rec, 4),
        "auc_roc":   round(rf_auc, 4),
    },
    "confusion_matrix": {
        "TN": int(tn), "FP": int(fp),
        "FN": int(fn), "TP": int(tp),
    },
    "feature_importance": {f: round(i, 4) for f, i in feat_imp},
    "model_comparison": model_results,
    "hyperparameters": {
        "num_trees": 100, "max_depth": 10, "seed": 42,
        "feature_subset_strategy": "sqrt"
    },
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open(os.path.join(GOLD_DIR, 'gold_metrics.json'), 'w') as f:
    json.dump(gold_metrics, f, indent=2)

spark.stop()
print("  Accuracy  : {:.4f}  ({:.2f}%)".format(rf_acc, rf_acc*100))
print("  F1-Score  : {:.4f}  ({:.2f}%)".format(rf_f1, rf_f1*100))
print("  Saved: " + GOLD_DIR)

# ===========================================================================
# VISUALISASI TAMBAHAN (Bronze/Silver/Gold diagram)
# ===========================================================================
print("\n[VIZ] Membuat visualisasi tambahan ...")

# 1. Medallion Architecture Diagram
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis('off')
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

layers = [
    (1.5,  '#cd7f32', 'BRONZE',  'Raw Data\n3.276 baris\n10 kolom\nMissing: ph,Sulfate,THMs'),
    (5.5,  '#c0c0c0', 'SILVER',  'Clean Data\n3.276 baris\nImputasi mean\nDeduplikasi'),
    (9.5,  '#ffd700', 'GOLD',    'Analitik Final\nRandom Forest\nAcc: {:.2f}%\nF1: {:.2f}%'.format(rf_acc*100, rf_f1*100)),
]
for x, color, title, body in layers:
    rect = mpatches.FancyBboxPatch((x-1.2, 1), 2.4, 3, boxstyle="round,pad=0.1",
                                    facecolor=color+'33', edgecolor=color, linewidth=2.5)
    ax.add_patch(rect)
    ax.text(x, 3.7, title, ha='center', va='center', fontsize=14, fontweight='bold',
            color=color, fontfamily='monospace')
    ax.text(x, 2.3, body, ha='center', va='center', fontsize=9.5, color='white',
            multialignment='center')

for x_start in [2.8, 6.8]:
    ax.annotate('', xy=(x_start+0.9, 2.5), xytext=(x_start, 2.5),
                arrowprops=dict(arrowstyle='->', color='white', lw=2.5))
    ax.text(x_start+0.45, 2.85, 'Transformasi', ha='center', va='center',
            fontsize=8.5, color='#aaaaaa')

ax.text(7, 0.3, 'Medallion Architecture — Water Potability Prediction Pipeline',
        ha='center', va='center', fontsize=11, color='#888888', style='italic')
plt.tight_layout()
path_medal = os.path.join(FIGURES_DIR, 'medallion_architecture.png')
plt.savefig(path_medal, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved: " + path_medal)

# 2. Silver boxplot (distribusi fitur setelah cleaning)
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()
colors_box = ['#3498db','#2ecc71','#e74c3c','#f39c12','#9b59b6',
               '#1abc9c','#e67e22','#e91e63','#00bcd4']
for i, feat in enumerate(FEATURE_COLS):
    data_0 = df_silver[df_silver['Potability']==0][feat]
    data_1 = df_silver[df_silver['Potability']==1][feat]
    bp = axes[i].boxplot([data_0, data_1], labels=['Tidak Layak','Layak'],
                          patch_artist=True, notch=False,
                          medianprops=dict(color='white', linewidth=2))
    bp['boxes'][0].set_facecolor('#e74c3c80')
    bp['boxes'][1].set_facecolor('#27ae6080')
    axes[i].set_title(feat, fontsize=11, fontweight='bold')
    axes[i].grid(axis='y', alpha=0.3)
    axes[i].tick_params(labelsize=8)
fig.suptitle('Boxplot Distribusi Fitur per Kelas (Silver Layer)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
path_box = os.path.join(FIGURES_DIR, 'silver_boxplot.png')
plt.savefig(path_box, dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: " + path_box)

# ===========================================================================
# GENERATE DASHBOARD HTML
# ===========================================================================
print("\n[DASHBOARD] Generating dashboard.html ...")

def img_path(name):
    return 'output/figures/' + name

# Encode data for Chart.js
feat_labels = [f for f, _ in feat_imp]
feat_values = [round(i*100, 2) for _, i in feat_imp]

model_names_js  = [r['Model'] for r in model_results]
model_acc_js    = [round(r['Accuracy']*100, 2) for r in model_results]
model_f1_js     = [round(r['F1_Score']*100, 2) for r in model_results]
model_pre_js    = [round(r['Precision']*100, 2) for r in model_results]
model_rec_js    = [round(r['Recall']*100, 2) for r in model_results]

html = '''<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dashboard ABD - Prediksi Potabilitas Air | Kelompok 1</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-primary:   #0d1117;
    --bg-card:      #161b22;
    --bg-card2:     #1c2128;
    --border:       #30363d;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --bronze:       #cd7f32;
    --bronze-bg:    rgba(205,127,50,0.12);
    --silver:       #a8b2c1;
    --silver-bg:    rgba(168,178,193,0.12);
    --gold:         #ffd700;
    --gold-bg:      rgba(255,215,0,0.12);
    --accent:       #2f81f7;
    --accent-bg:    rgba(47,129,247,0.12);
    --green:        #3fb950;
    --red:          #f85149;
    --orange:       #d29922;
    --purple:       #bc8cff;
    --radius:       12px;
    --radius-lg:    16px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
  }

  /* ── HEADER ── */
  .header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 40px 0 30px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .header::before {
    content:'';
    position:absolute; inset:0;
    background: radial-gradient(ellipse at 50% 0%, rgba(47,129,247,0.08) 0%, transparent 70%);
  }
  .badge {
    display: inline-block;
    background: var(--accent-bg);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 16px;
  }
  .header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e6edf3 0%, #2f81f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
  }
  .header p {
    color: var(--text-muted);
    font-size: 1rem;
    max-width: 600px;
    margin: 0 auto;
  }
  .header-meta {
    margin-top: 20px;
    display: flex;
    justify-content: center;
    gap: 24px;
    flex-wrap: wrap;
  }
  .meta-chip {
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 12px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
  }
  .meta-chip span { color: var(--text-primary); font-weight: 600; }

  /* ── LAYOUT ── */
  .container { max-width: 1280px; margin: 0 auto; padding: 0 24px; }
  .section { padding: 40px 0; }
  .section-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .section-title .icon {
    width: 32px; height: 32px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
  }

  /* ── CARDS ── */
  .grid { display: grid; gap: 16px; }
  .grid-3 { grid-template-columns: repeat(3, 1fr); }
  .grid-4 { grid-template-columns: repeat(4, 1fr); }
  .grid-2 { grid-template-columns: repeat(2, 1fr); }
  @media(max-width:900px){ .grid-3,.grid-4,.grid-2{ grid-template-columns:1fr; } }

  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    transition: border-color 0.2s, transform 0.2s;
  }
  .card:hover { border-color: #484f58; transform: translateY(-2px); }

  /* ── MEDALLION LAYERS ── */
  .layer-bronze { border-color: var(--bronze); background: var(--bronze-bg); }
  .layer-silver { border-color: var(--silver); background: var(--silver-bg); }
  .layer-gold   { border-color: var(--gold);   background: var(--gold-bg);   }

  .layer-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 16px;
  }
  .layer-icon {
    width: 44px; height: 44px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
  }
  .layer-icon.bronze { background: rgba(205,127,50,0.2); }
  .layer-icon.silver { background: rgba(168,178,193,0.2); }
  .layer-icon.gold   { background: rgba(255,215,0,0.2);   }

  .layer-label {
    font-size: 11px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
  }
  .layer-label.bronze { color: var(--bronze); }
  .layer-label.silver { color: var(--silver); }
  .layer-label.gold   { color: var(--gold);   }

  .layer-name { font-size: 1.05rem; font-weight: 600; color: var(--text-primary); }

  .stat-row { display: flex; justify-content: space-between; padding: 8px 0;
              border-bottom: 1px solid var(--border); font-size: 0.88rem; }
  .stat-row:last-child { border-bottom: none; }
  .stat-label { color: var(--text-muted); }
  .stat-value { font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
  .stat-value.warn { color: var(--orange); }
  .stat-value.ok   { color: var(--green);  }
  .stat-value.info { color: var(--accent); }

  /* ── METRIC CARDS ── */
  .metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    text-align: center;
    transition: all 0.2s;
  }
  .metric-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, var(--accent) 0%, var(--purple) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .metric-name { font-size: 0.82rem; color: var(--text-muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }
  .metric-sub  { font-size: 0.78rem; color: var(--text-muted); margin-top: 2px; }

  /* ── PIPELINE ARROW ── */
  .pipeline-flow {
    display: flex; align-items: center; justify-content: center;
    gap: 0; margin-bottom: 32px; flex-wrap: wrap;
  }
  .pipeline-step {
    display: flex; flex-direction: column;
    align-items: center; padding: 14px 24px;
    border-radius: 10px;
    min-width: 140px; text-align: center;
  }
  .step-bronze { background: var(--bronze-bg); border: 1.5px solid var(--bronze); }
  .step-silver { background: var(--silver-bg); border: 1.5px solid var(--silver); }
  .step-gold   { background: var(--gold-bg);   border: 1.5px solid var(--gold);   }
  .step-emoji  { font-size: 1.8rem; margin-bottom: 4px; }
  .step-label  { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
  .step-desc   { font-size: 0.78rem; color: var(--text-muted); margin-top: 2px; }
  .pipeline-arrow { font-size: 1.4rem; color: var(--text-muted); padding: 0 8px; }

  /* ── CHARTS ── */
  .chart-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
  }
  .chart-title { font-size: 1rem; font-weight: 600; margin-bottom: 16px; color: var(--text-primary); }
  .chart-wrap  { position: relative; height: 300px; }
  .chart-wrap-sm { position: relative; height: 240px; }

  /* ── TABLE ── */
  .table-wrap {
    overflow-x: auto;
    border-radius: var(--radius);
    border: 1px solid var(--border);
  }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  thead th {
    background: var(--bg-card2);
    padding: 12px 16px;
    text-align: left;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    font-weight: 600;
    border-bottom: 1px solid var(--border);
  }
  tbody tr { border-bottom: 1px solid var(--border); transition: background 0.15s; }
  tbody tr:last-child { border-bottom: none; }
  tbody tr:hover { background: var(--bg-card2); }
  tbody td { padding: 12px 16px; }
  .best-row { background: rgba(47,129,247,0.06); }
  .tag {
    display: inline-block; padding: 2px 10px;
    border-radius: 12px; font-size: 0.75rem; font-weight: 600;
  }
  .tag-best { background: var(--accent-bg); color: var(--accent); border: 1px solid var(--accent); }

  /* ── CONFUSION MATRIX ── */
  .cm-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 10px; max-width: 360px; margin: 0 auto;
  }
  .cm-cell {
    border-radius: 10px; padding: 20px;
    text-align: center; font-family: 'JetBrains Mono', monospace;
  }
  .cm-tp { background: rgba(63,185,80,0.2); border: 1.5px solid var(--green); }
  .cm-tn { background: rgba(63,185,80,0.2); border: 1.5px solid var(--green); }
  .cm-fp { background: rgba(248,81,73,0.2);  border: 1.5px solid var(--red);   }
  .cm-fn { background: rgba(248,81,73,0.2);  border: 1.5px solid var(--red);   }
  .cm-num  { font-size: 2rem; font-weight: 800; }
  .cm-label { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }
  .cm-tp .cm-num, .cm-tn .cm-num { color: var(--green); }
  .cm-fp .cm-num, .cm-fn .cm-num { color: var(--red);   }

  /* ── IMAGE GRID ── */
  .img-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    transition: all 0.2s;
  }
  .img-card:hover { border-color: #484f58; transform: translateY(-2px); }
  .img-card img { width: 100%; display: block; }
  .img-caption {
    padding: 12px 16px;
    font-size: 0.83rem;
    color: var(--text-muted);
    text-align: center;
    border-top: 1px solid var(--border);
  }

  /* ── DIVIDER ── */
  .divider { border: none; border-top: 1px solid var(--border); margin: 8px 0; }

  /* ── FOOTER ── */
  .footer {
    border-top: 1px solid var(--border);
    padding: 28px 0;
    text-align: center;
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-top: 20px;
  }
  .footer a { color: var(--accent); text-decoration: none; }
  .footer a:hover { text-decoration: underline; }
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="container">
    <div class="badge">Tugas Besar ABD · Kelompok 1</div>
    <h1>Dashboard Prediksi Potabilitas Air Sungai</h1>
    <p>Analisis berbasis <strong>Apache Spark MLlib</strong> dengan arsitektur Medallion (Bronze → Silver → Gold) menggunakan algoritma Random Forest</p>
    <div class="header-meta">
      <div class="meta-chip">Dataset: <span>water_potability.csv</span></div>
      <div class="meta-chip">Rows: <span>3,276</span></div>
      <div class="meta-chip">Algorithm: <span>Random Forest</span></div>
      <div class="meta-chip">PySpark: <span>''' + pyspark.__version__ + '''</span></div>
      <div class="meta-chip">Generated: <span>''' + time.strftime("%Y-%m-%d %H:%M") + '''</span></div>
    </div>
  </div>
</div>

<!-- MEDALLION PIPELINE FLOW -->
<div class="container">
  <div class="section">
    <div class="section-title">
      <div class="icon" style="background:rgba(47,129,247,0.15);">🏗️</div>
      Medallion Architecture Pipeline
    </div>
    <div class="pipeline-flow">
      <div class="pipeline-step step-bronze">
        <div class="step-emoji">🥉</div>
        <div class="step-label" style="color:var(--bronze)">Bronze</div>
        <div class="step-desc">Raw Data<br>3,276 baris</div>
      </div>
      <div class="pipeline-arrow">→</div>
      <div class="pipeline-step step-silver">
        <div class="step-emoji">🥈</div>
        <div class="step-label" style="color:var(--silver)">Silver</div>
        <div class="step-desc">Clean Data<br>Imputasi + Dedup</div>
      </div>
      <div class="pipeline-arrow">→</div>
      <div class="pipeline-step step-gold">
        <div class="step-emoji">🥇</div>
        <div class="step-label" style="color:var(--gold)">Gold</div>
        <div class="step-desc">Analitik Final<br>Model + Evaluasi</div>
      </div>
    </div>

    <!-- LAYER CARDS -->
    <div class="grid grid-3">

      <!-- BRONZE -->
      <div class="card layer-bronze">
        <div class="layer-header">
          <div class="layer-icon bronze">🥉</div>
          <div>
            <div class="layer-label bronze">Bronze Layer</div>
            <div class="layer-name">Raw Data</div>
          </div>
        </div>
        <div class="stat-row"><span class="stat-label">Total Baris</span>
          <span class="stat-value info">''' + f"{bronze_stats['total_rows']:,}" + '''</span></div>
        <div class="stat-row"><span class="stat-label">Total Kolom</span>
          <span class="stat-value info">''' + str(bronze_stats['total_columns']) + '''</span></div>
        <div class="stat-row"><span class="stat-label">Missing: ph</span>
          <span class="stat-value warn">''' + str(bronze_stats['missing_values']['ph']) + ''' (15.0%)</span></div>
        <div class="stat-row"><span class="stat-label">Missing: Sulfate</span>
          <span class="stat-value warn">''' + str(bronze_stats['missing_values']['Sulfate']) + ''' (23.8%)</span></div>
        <div class="stat-row"><span class="stat-label">Missing: THMs</span>
          <span class="stat-value warn">''' + str(bronze_stats['missing_values']['Trihalomethanes']) + ''' (4.9%)</span></div>
        <div class="stat-row"><span class="stat-label">Duplikat</span>
          <span class="stat-value ok">''' + str(bronze_stats['duplicates']) + '''</span></div>
        <div class="stat-row"><span class="stat-label">File</span>
          <span class="stat-value info">bronze_raw.csv</span></div>
      </div>

      <!-- SILVER -->
      <div class="card layer-silver">
        <div class="layer-header">
          <div class="layer-icon silver">🥈</div>
          <div>
            <div class="layer-label silver">Silver Layer</div>
            <div class="layer-name">Clean Data</div>
          </div>
        </div>
        <div class="stat-row"><span class="stat-label">Total Baris</span>
          <span class="stat-value info">''' + f"{silver_stats['total_rows']:,}" + '''</span></div>
        <div class="stat-row"><span class="stat-label">Duplikat Dihapus</span>
          <span class="stat-value ok">''' + str(silver_stats['duplicates_removed']) + '''</span></div>
        <div class="stat-row"><span class="stat-label">Imputasi ph</span>
          <span class="stat-value ok">mean = ''' + str(impute_means['ph']) + '''</span></div>
        <div class="stat-row"><span class="stat-label">Imputasi Sulfate</span>
          <span class="stat-value ok">mean = ''' + str(impute_means['Sulfate']) + '''</span></div>
        <div class="stat-row"><span class="stat-label">Imputasi THMs</span>
          <span class="stat-value ok">mean = ''' + str(impute_means['Trihalomethanes']) + '''</span></div>
        <div class="stat-row"><span class="stat-label">Missing Setelah</span>
          <span class="stat-value ok">0</span></div>
        <div class="stat-row"><span class="stat-label">File</span>
          <span class="stat-value info">silver_clean.csv</span></div>
      </div>

      <!-- GOLD -->
      <div class="card layer-gold">
        <div class="layer-header">
          <div class="layer-icon gold">🥇</div>
          <div>
            <div class="layer-label gold">Gold Layer</div>
            <div class="layer-name">Analitik Final</div>
          </div>
        </div>
        <div class="stat-row"><span class="stat-label">Model Terbaik</span>
          <span class="stat-value info">Random Forest</span></div>
        <div class="stat-row"><span class="stat-label">Accuracy</span>
          <span class="stat-value ok">''' + f"{rf_acc:.4f} ({rf_acc*100:.2f}%)" + '''</span></div>
        <div class="stat-row"><span class="stat-label">F1-Score</span>
          <span class="stat-value ok">''' + f"{rf_f1:.4f} ({rf_f1*100:.2f}%)" + '''</span></div>
        <div class="stat-row"><span class="stat-label">Precision</span>
          <span class="stat-value ok">''' + f"{rf_pre:.4f} ({rf_pre*100:.2f}%)" + '''</span></div>
        <div class="stat-row"><span class="stat-label">Recall</span>
          <span class="stat-value ok">''' + f"{rf_rec:.4f} ({rf_rec*100:.2f}%)" + '''</span></div>
        <div class="stat-row"><span class="stat-label">AUC-ROC</span>
          <span class="stat-value ok">''' + f"{rf_auc:.4f} ({rf_auc*100:.2f}%)" + '''</span></div>
        <div class="stat-row"><span class="stat-label">Files</span>
          <span class="stat-value info">metrics.json + CSVs</span></div>
      </div>
    </div>
  </div>

  <!-- KEY METRICS -->
  <div class="section" style="padding-top:0">
    <div class="section-title">
      <div class="icon" style="background:rgba(63,185,80,0.15);">📈</div>
      Hasil Evaluasi Model (Random Forest)
    </div>
    <div class="grid grid-4">
      <div class="metric-card">
        <div class="metric-value">''' + f"{rf_acc*100:.1f}%" + '''</div>
        <div class="metric-name">Accuracy</div>
        <div class="metric-sub">''' + str(len(y_true)) + ''' test samples</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">''' + f"{rf_f1*100:.1f}%" + '''</div>
        <div class="metric-name">F1-Score</div>
        <div class="metric-sub">Weighted average</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">''' + f"{rf_pre*100:.1f}%" + '''</div>
        <div class="metric-name">Precision</div>
        <div class="metric-sub">Weighted average</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">''' + f"{rf_auc*100:.1f}%" + '''</div>
        <div class="metric-name">AUC-ROC</div>
        <div class="metric-sub">Binary classification</div>
      </div>
    </div>
  </div>

  <!-- CHARTS ROW 1 -->
  <div class="section" style="padding-top:0">
    <div class="section-title">
      <div class="icon" style="background:rgba(188,140,255,0.15);">📊</div>
      Visualisasi Model
    </div>
    <div class="grid grid-2">
      <div class="chart-card">
        <div class="chart-title">Feature Importance — Random Forest</div>
        <div class="chart-wrap"><canvas id="chartFeat"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Perbandingan Model — Accuracy & F1-Score</div>
        <div class="chart-wrap"><canvas id="chartCompare"></canvas></div>
      </div>
    </div>
  </div>

  <!-- CONFUSION MATRIX + RADAR -->
  <div class="section" style="padding-top:0">
    <div class="grid grid-2">
      <div class="chart-card">
        <div class="chart-title">Confusion Matrix — Random Forest</div>
        <div style="padding:16px 0">
          <div style="text-align:center;margin-bottom:12px;font-size:0.8rem;color:var(--text-muted)">
            Prediksi ↓ &nbsp;|&nbsp; Aktual →
          </div>
          <div class="cm-grid">
            <div class="cm-cell cm-tn">
              <div class="cm-num">''' + str(tn) + '''</div>
              <div class="cm-label">True Negative<br>(TN)</div>
            </div>
            <div class="cm-cell cm-fp">
              <div class="cm-num">''' + str(fp) + '''</div>
              <div class="cm-label">False Positive<br>(FP)</div>
            </div>
            <div class="cm-cell cm-fn">
              <div class="cm-num">''' + str(fn) + '''</div>
              <div class="cm-label">False Negative<br>(FN)</div>
            </div>
            <div class="cm-cell cm-tp">
              <div class="cm-num">''' + str(tp) + '''</div>
              <div class="cm-label">True Positive<br>(TP)</div>
            </div>
          </div>
        </div>
        <div style="margin-top:16px;display:flex;justify-content:center;gap:32px;font-size:0.83rem">
          <div style="text-align:center">
            <div style="color:var(--green);font-weight:700;font-size:1.1rem">''' + f"{(tn+tp)/len(y_true)*100:.1f}%" + '''</div>
            <div style="color:var(--text-muted)">Benar</div>
          </div>
          <div style="text-align:center">
            <div style="color:var(--red);font-weight:700;font-size:1.1rem">''' + f"{(fp+fn)/len(y_true)*100:.1f}%" + '''</div>
            <div style="color:var(--text-muted)">Salah</div>
          </div>
        </div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Radar Chart — Semua Metrik per Model</div>
        <div class="chart-wrap"><canvas id="chartRadar"></canvas></div>
      </div>
    </div>
  </div>

  <!-- TABLE PERBANDINGAN -->
  <div class="section" style="padding-top:0">
    <div class="section-title">
      <div class="icon" style="background:rgba(210,153,34,0.15);">🏆</div>
      Tabel Perbandingan Model
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th><th>Accuracy</th><th>F1-Score</th>
            <th>Precision</th><th>Recall</th><th>AUC-ROC</th><th>Status</th>
          </tr>
        </thead>
        <tbody>'''

for i, r in enumerate(model_results):
    is_best = r['Model'] == 'Random Forest'
    row_class = ' class="best-row"' if is_best else ''
    tag = '<span class="tag tag-best">&#9733; Terbaik</span>' if is_best else ''
    html += f'''
          <tr{row_class}>
            <td><strong>{r["Model"]}</strong></td>
            <td>{r["Accuracy"]*100:.2f}%</td>
            <td>{r["F1_Score"]*100:.2f}%</td>
            <td>{r["Precision"]*100:.2f}%</td>
            <td>{r["Recall"]*100:.2f}%</td>
            <td>{r["AUC_ROC"]*100:.2f}%</td>
            <td>{tag}</td>
          </tr>'''

html += '''
        </tbody>
      </table>
    </div>
  </div>

  <!-- VISUALISASI PNG -->
  <div class="section" style="padding-top:0">
    <div class="section-title">
      <div class="icon" style="background:rgba(47,129,247,0.15);">🖼️</div>
      Visualisasi EDA & Hasil Analisis
    </div>
    <div class="grid grid-2">
      <div class="img-card">
        <img src="output/figures/medallion_architecture.png" alt="Medallion Architecture" loading="lazy">
        <div class="img-caption">Arsitektur Medallion Pipeline</div>
      </div>
      <div class="img-card">
        <img src="output/figures/heatmap_korelasi.png" alt="Heatmap Korelasi" loading="lazy">
        <div class="img-caption">Heatmap Korelasi Antar Fitur</div>
      </div>
      <div class="img-card">
        <img src="output/figures/distribusi_label.png" alt="Distribusi Label" loading="lazy">
        <div class="img-caption">Distribusi Kelas Potabilitas (Bronze Layer)</div>
      </div>
      <div class="img-card">
        <img src="output/figures/distribusi_fitur.png" alt="Distribusi Fitur" loading="lazy">
        <div class="img-caption">Distribusi Fitur per Kelas (Bronze → Silver)</div>
      </div>
      <div class="img-card">
        <img src="output/figures/silver_boxplot.png" alt="Silver Boxplot" loading="lazy">
        <div class="img-caption">Boxplot Fitur Setelah Cleaning (Silver Layer)</div>
      </div>
      <div class="img-card">
        <img src="output/figures/confusion_matrix.png" alt="Confusion Matrix" loading="lazy">
        <div class="img-caption">Confusion Matrix — Random Forest (Gold Layer)</div>
      </div>
      <div class="img-card">
        <img src="output/figures/feature_importance.png" alt="Feature Importance" loading="lazy">
        <div class="img-caption">Feature Importance — Random Forest</div>
      </div>
      <div class="img-card">
        <img src="output/figures/perbandingan_model.png" alt="Perbandingan Model" loading="lazy">
        <div class="img-caption">Perbandingan Performa Semua Model</div>
      </div>
    </div>
  </div>

  <!-- FOOTER -->
  <div class="footer">
    <p><strong>Tugas Besar Analisis Big Data — Kelompok 1</strong></p>
    <p style="margin-top:6px">
      Dataset: <a href="https://www.kaggle.com/datasets/adityakadiwal/water-potability" target="_blank">Water Potability (Kaggle)</a>
      &nbsp;·&nbsp;
      Repo: <a href="https://github.com/MuhammadAqil1/tubesabd" target="_blank">github.com/MuhammadAqil1/tubesabd</a>
    </p>
    <p style="margin-top:6px;color:#484f58">Generated with PySpark ''' + pyspark.__version__ + ''' · ''' + time.strftime("%Y-%m-%d %H:%M:%S") + '''</p>
  </div>
</div>

<!-- CHART.JS SCRIPTS -->
<script>
const chartDefaults = {
  color: '#8b949e',
  plugins: { legend: { labels: { color: '#8b949e', font: { family: 'Inter' } } } },
  scales: {
    x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
    y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
  }
};

// Feature Importance
new Chart(document.getElementById('chartFeat'), {
  type: 'bar',
  data: {
    labels: ''' + json.dumps(feat_labels) + ''',
    datasets: [{
      label: 'Importance (%)',
      data: ''' + json.dumps(feat_values) + ''',
      backgroundColor: [
        '#2f81f7','#3fb950','#f85149','#d29922','#bc8cff',
        '#ff7b72','#79c0ff','#56d364','#ffa657'
      ],
      borderRadius: 6,
      borderSkipped: false,
    }]
  },
  options: {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color:'#8b949e', callback: v => v+'%' }, grid: { color:'#21262d' } },
      y: { ticks: { color:'#8b949e' }, grid: { color:'#21262d' } }
    }
  }
});

// Model Comparison
new Chart(document.getElementById('chartCompare'), {
  type: 'bar',
  data: {
    labels: ''' + json.dumps(model_names_js) + ''',
    datasets: [
      { label: 'Accuracy (%)',  data: ''' + json.dumps(model_acc_js) + ''', backgroundColor: '#2f81f7', borderRadius: 5 },
      { label: 'F1-Score (%)', data: ''' + json.dumps(model_f1_js) + ''',  backgroundColor: '#3fb950', borderRadius: 5 },
      { label: 'Precision (%)',data: ''' + json.dumps(model_pre_js) + ''', backgroundColor: '#bc8cff', borderRadius: 5 },
      { label: 'Recall (%)',   data: ''' + json.dumps(model_rec_js) + ''', backgroundColor: '#d29922', borderRadius: 5 },
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color:'#8b949e', font:{ family:'Inter' } } },
      annotation: {}
    },
    scales: {
      x: { ticks:{color:'#8b949e'}, grid:{color:'#21262d'} },
      y: { min:40, max:85, ticks:{color:'#8b949e', callback: v=>v+'%'}, grid:{color:'#21262d'} }
    }
  }
});

// Radar Chart
new Chart(document.getElementById('chartRadar'), {
  type: 'radar',
  data: {
    labels: ['Accuracy','F1-Score','Precision','Recall','AUC-ROC'],
    datasets: ''' + json.dumps([
        {
            "label": r["Model"],
            "data": [
                round(r["Accuracy"]*100, 1),
                round(r["F1_Score"]*100, 1),
                round(r["Precision"]*100, 1),
                round(r["Recall"]*100, 1),
                round(r["AUC_ROC"]*100, 1),
            ],
            "borderColor": c,
            "backgroundColor": c + "22",
            "pointBackgroundColor": c,
            "borderWidth": 2,
        }
        for r, c in zip(model_results, ['#2f81f7','#f85149','#3fb950','#d29922'])
    ]) + '''
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color:'#8b949e', font:{family:'Inter'} } } },
    scales: {
      r: {
        min: 40, max: 85,
        ticks: { color:'#8b949e', backdropColor:'transparent', stepSize:10 },
        grid: { color:'#30363d' },
        pointLabels: { color:'#8b949e', font:{family:'Inter', size:11} }
      }
    }
  }
});
</script>
</body>
</html>'''

dashboard_path = os.path.join(BASE_DIR, 'dashboard.html')
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(html)

print("  Saved: " + dashboard_path)

# ===========================================================================
# RINGKASAN AKHIR
# ===========================================================================
print()
print("=" * 65)
print("  MEDALLION PIPELINE SELESAI!")
print("=" * 65)
print()
print("Artifacts yang dihasilkan:")
print()
print("  BRONZE  output/bronze/")
for f in os.listdir(BRONZE_DIR):
    size = os.path.getsize(os.path.join(BRONZE_DIR, f)) // 1024
    print("    - {}  ({} KB)".format(f, size))
print()
print("  SILVER  output/silver/")
for f in os.listdir(SILVER_DIR):
    size = os.path.getsize(os.path.join(SILVER_DIR, f)) // 1024
    print("    - {}  ({} KB)".format(f, size))
print()
print("  GOLD    output/gold/")
for f in os.listdir(GOLD_DIR):
    size = os.path.getsize(os.path.join(GOLD_DIR, f)) // 1024
    print("    - {}  ({} KB)".format(f, size))
print()
print("  DASHBOARD  dashboard.html")
size_dash = os.path.getsize(dashboard_path) // 1024
print("    - dashboard.html  ({} KB)".format(size_dash))
print()
print("Buka dashboard: file://" + dashboard_path.replace("\\", "/"))
print()
