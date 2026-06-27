# ============================================================
#  BRAIN STROKE PREDICTION  |  Phase 1 — Data Loading & EDA
#  Dataset : Stroke Prediction Dataset (Kaggle – fedesoriano)
#  kaggle datasets download -d fedesoriano/stroke-prediction-dataset
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Global plot style ──────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'figure.figsize'  : (14, 6),
    'axes.titlesize'  : 14,
    'axes.labelsize'  : 12,
    'xtick.labelsize' : 10,
    'ytick.labelsize' : 10,
})

STROKE_COLORS = {0: '#4C72B0', 1: '#DD8452'}   # No-stroke blue / Stroke orange
OUTPUT_DIR    = 'outputs'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  1.  LOAD DATA
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  BRAIN STROKE PREDICTION — Phase 1: Data Loading")
print("=" * 60)

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

print(f"\n📂  Dataset loaded  →  {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\n── First 5 rows ──────────────────────────────────────")
print(df.head())


# ═══════════════════════════════════════════════════════════
#  2.  BASIC STRUCTURE
# ═══════════════════════════════════════════════════════════
print("\n\n── Data Types & Non-Null Counts ──────────────────────")
print(df.info())

print("\n── Statistical Summary (numeric) ─────────────────────")
print(df.describe().T.round(2))

print("\n── Statistical Summary (categorical) ─────────────────")
print(df.describe(include='object').T)


# ═══════════════════════════════════════════════════════════
#  3.  MISSING VALUES
# ═══════════════════════════════════════════════════════════
print("\n\n── Missing Values ────────────────────────────────────")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df  = pd.DataFrame({'Missing Count': missing,
                            'Missing %'    : missing_pct})
missing_df  = missing_df[missing_df['Missing Count'] > 0]
print(missing_df if not missing_df.empty else "✅  No missing values found.")

# Plot missing values
if not missing_df.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=missing_df.index, y='Missing %', data=missing_df,
                palette='Reds_r', ax=ax)
    ax.set_title('Missing Value Percentage by Column')
    ax.set_ylabel('Missing %')
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{bar.get_height():.2f}%",
                ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_missing_values.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  → Saved: {OUTPUT_DIR}/01_missing_values.png")


# ═══════════════════════════════════════════════════════════
#  4.  DUPLICATES
# ═══════════════════════════════════════════════════════════
dups = df.duplicated().sum()
print(f"\n── Duplicate Rows ────────────────────────────────────")
print(f"  Duplicate rows found: {dups}")


# ═══════════════════════════════════════════════════════════
#  5.  TARGET DISTRIBUTION (Class Imbalance Check)
# ═══════════════════════════════════════════════════════════
print("\n── Target Distribution (stroke) ──────────────────────")
target_counts = df['stroke'].value_counts()
target_pct    = df['stroke'].value_counts(normalize=True) * 100
print(pd.DataFrame({'Count': target_counts,
                    'Percentage': target_pct.round(2)}))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
axes[0].bar(['No Stroke (0)', 'Stroke (1)'],
            target_counts.values,
            color=[STROKE_COLORS[0], STROKE_COLORS[1]], edgecolor='white', linewidth=1.2)
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 30, f"{v:,}\n({target_pct.iloc[i]:.1f}%)",
                 ha='center', fontsize=12, fontweight='bold')
axes[0].set_title('Class Distribution', fontsize=14)
axes[0].set_ylabel('Count')

# Pie chart
axes[1].pie(target_counts.values,
            labels=['No Stroke', 'Stroke'],
            colors=[STROKE_COLORS[0], STROKE_COLORS[1]],
            autopct='%1.1f%%', startangle=90,
            explode=(0, 0.08), shadow=True, textprops={'fontsize': 12})
axes[1].set_title('Stroke vs No-Stroke Proportion', fontsize=14)

plt.suptitle('Target Variable Distribution — Severe Class Imbalance ⚠️', fontsize=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/02_target_distribution.png")


# ═══════════════════════════════════════════════════════════
#  6.  UNIQUE VALUES PER COLUMN
# ═══════════════════════════════════════════════════════════
print("\n── Unique Values Per Column ──────────────────────────")
for col in df.columns:
    n = df[col].nunique()
    if n <= 10:
        vals = df[col].unique().tolist()
        print(f"  {col:25s}  ({n} unique)  →  {vals}")
    else:
        print(f"  {col:25s}  ({n} unique)  →  [numeric / high-cardinality]")


print("\n\n✅  Phase 1 complete — proceed to phase2_data_cleaning.py")
