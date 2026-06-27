# ============================================================
#  BRAIN STROKE PREDICTION  |  Phase 3 — EDA & Analytics
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.figsize': (14, 6), 'axes.titlesize': 14})
STROKE_COLORS = {0: '#4C72B0', 1: '#DD8452'}
OUTPUT_DIR    = 'outputs'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 60)
print("  BRAIN STROKE PREDICTION — Phase 3: EDA & Analytics")
print("=" * 60)


# ── Load cleaned data ──────────────────────────────────────
df = pd.read_csv('stroke_clean.csv')
print(f"\n📂  Cleaned data loaded → {df.shape}")


# ─── Helper ───────────────────────────────────────────────
def stroke_rate(grp):
    """Return stroke rate (%) within a group."""
    return grp['stroke'].mean() * 100


# ═══════════════════════════════════════════════════════════
#  3.1  NUMERIC FEATURES — Distributions split by Stroke
# ═══════════════════════════════════════════════════════════
print("\n── 3.1  Numeric Feature Distributions ───────────────")
numeric_feats = ['age', 'avg_glucose_level', 'bmi']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, col in enumerate(numeric_feats):
    # KDE by stroke class
    for val, label, color in [(0, 'No Stroke', STROKE_COLORS[0]),
                               (1, 'Stroke',    STROKE_COLORS[1])]:
        sns.kdeplot(df[df['stroke'] == val][col], ax=axes[0, idx],
                    label=label, color=color, fill=True, alpha=0.4)
    axes[0, idx].set_title(f'{col} — KDE by Stroke')
    axes[0, idx].legend()

    # Violin plot
    sns.violinplot(data=df, x='stroke', y=col, ax=axes[1, idx],
                   palette=STROKE_COLORS, inner='quartile')
    axes[1, idx].set_title(f'{col} — Violin Plot')
    axes[1, idx].set_xticklabels(['No Stroke', 'Stroke'])

plt.suptitle('Numeric Features: Distribution by Stroke Outcome', fontsize=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_numeric_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/05_numeric_distributions.png")


# ═══════════════════════════════════════════════════════════
#  3.2  CATEGORICAL FEATURES vs STROKE RATE
# ═══════════════════════════════════════════════════════════
print("\n── 3.2  Categorical Features vs Stroke Rate ──────────")
cat_feats = ['gender', 'hypertension', 'heart_disease', 'ever_married',
             'work_type', 'Residence_type', 'smoking_status']

fig, axes = plt.subplots(3, 3, figsize=(20, 16))
axes = axes.flatten()

for idx, col in enumerate(cat_feats):
    rate = df.groupby(col).apply(stroke_rate).reset_index()
    rate.columns = [col, 'stroke_rate']
    rate = rate.sort_values('stroke_rate', ascending=False)

    bars = axes[idx].bar(rate[col].astype(str), rate['stroke_rate'],
                         color='#DD8452', edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, rate['stroke_rate']):
        axes[idx].text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.1,
                       f"{val:.1f}%", ha='center', fontsize=10, fontweight='bold')
    axes[idx].set_title(f'Stroke Rate by {col}', fontsize=12)
    axes[idx].set_ylabel('Stroke Rate (%)')
    axes[idx].tick_params(axis='x', rotation=20)

# Remove extra axes
for i in range(len(cat_feats), len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Stroke Rate Across Categorical Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_categorical_stroke_rates.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/06_categorical_stroke_rates.png")


# ═══════════════════════════════════════════════════════════
#  3.3  AGE DEEP-DIVE  (most important predictor)
# ═══════════════════════════════════════════════════════════
print("\n── 3.3  Age Deep-Dive ────────────────────────────────")
df['age_group'] = pd.cut(df['age'],
                          bins=[0, 20, 35, 50, 65, 100],
                          labels=['<20', '20-35', '35-50', '50-65', '65+'])

age_stroke = df.groupby('age_group').apply(stroke_rate).reset_index()
age_stroke.columns = ['age_group', 'stroke_rate']
age_count  = df.groupby('age_group').size().reset_index(name='count')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(age_stroke['age_group'].astype(str), age_stroke['stroke_rate'],
            color=['#4C72B0', '#64A0C8', '#DD8452', '#C44E52', '#8B0000'],
            edgecolor='white')
for i, val in enumerate(age_stroke['stroke_rate']):
    axes[0].text(i, val + 0.2, f"{val:.1f}%", ha='center', fontweight='bold')
axes[0].set_title('Stroke Rate by Age Group')
axes[0].set_ylabel('Stroke Rate (%)')
axes[0].set_xlabel('Age Group')

axes[1].bar(age_count['age_group'].astype(str), age_count['count'],
            color='steelblue', edgecolor='white')
for i, val in enumerate(age_count['count']):
    axes[1].text(i, val + 10, f"{val:,}", ha='center', fontsize=10)
axes[1].set_title('Patient Count by Age Group')
axes[1].set_ylabel('Count')
axes[1].set_xlabel('Age Group')

plt.suptitle('Age Group Analysis', fontsize=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_age_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/07_age_analysis.png")


# ═══════════════════════════════════════════════════════════
#  3.4  GLUCOSE LEVEL DEEP-DIVE
# ═══════════════════════════════════════════════════════════
print("\n── 3.4  Glucose Level Deep-Dive ─────────────────────")
df['glucose_category'] = pd.cut(df['avg_glucose_level'],
                                  bins=[0, 100, 125, 200, 300],
                                  labels=['Normal (<100)', 'Pre-diabetic (100-125)',
                                          'Diabetic (125-200)', 'High (>200)'])

gluc_stroke = df.groupby('glucose_category').apply(stroke_rate).reset_index()
gluc_stroke.columns = ['glucose_category', 'stroke_rate']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Stacked bar count
gluc_counts = df.groupby(['glucose_category', 'stroke']).size().unstack().fillna(0)
gluc_counts.plot(kind='bar', ax=axes[0],
                 color=[STROKE_COLORS[0], STROKE_COLORS[1]],
                 edgecolor='white')
axes[0].set_title('Count by Glucose Category & Stroke')
axes[0].set_xlabel('Glucose Category')
axes[0].set_ylabel('Count')
axes[0].legend(['No Stroke', 'Stroke'])
axes[0].tick_params(axis='x', rotation=15)

# Stroke rate
axes[1].bar(gluc_stroke['glucose_category'].astype(str),
            gluc_stroke['stroke_rate'],
            color=['#64A0C8', '#F0A500', '#DD8452', '#C44E52'],
            edgecolor='white')
for i, val in enumerate(gluc_stroke['stroke_rate']):
    axes[1].text(i, val + 0.1, f"{val:.1f}%", ha='center', fontweight='bold')
axes[1].set_title('Stroke Rate by Glucose Category')
axes[1].set_ylabel('Stroke Rate (%)')
axes[1].tick_params(axis='x', rotation=15)

plt.suptitle('Average Glucose Level Analysis', fontsize=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_glucose_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/08_glucose_analysis.png")


# ═══════════════════════════════════════════════════════════
#  3.5  RISK FACTOR COMBINATIONS
# ═══════════════════════════════════════════════════════════
print("\n── 3.5  Compounded Risk Factors ─────────────────────")
df['risk_score'] = (
    (df['age'] > 60).astype(int) +
    df['hypertension'] +
    df['heart_disease'] +
    (df['avg_glucose_level'] > 125).astype(int) +
    (df['bmi'] > 30).astype(int)
)

risk_stroke = df.groupby('risk_score').apply(stroke_rate).reset_index()
risk_stroke.columns = ['risk_score', 'stroke_rate']
risk_count  = df.groupby('risk_score').size().reset_index(name='count')
risk_merged = risk_stroke.merge(risk_count, on='risk_score')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(risk_merged['risk_score'].astype(str),
            risk_merged['stroke_rate'],
            color=plt.cm.Reds(np.linspace(0.3, 1.0, len(risk_merged))),
            edgecolor='white')
for i, row in risk_merged.iterrows():
    axes[0].text(i, row['stroke_rate'] + 0.3,
                 f"{row['stroke_rate']:.1f}%", ha='center', fontweight='bold')
axes[0].set_title('Stroke Rate by Compounded Risk Score')
axes[0].set_xlabel('Risk Score (0=low, 5=high)')
axes[0].set_ylabel('Stroke Rate (%)')

axes[1].bar(risk_merged['risk_score'].astype(str),
            risk_merged['count'],
            color='steelblue', edgecolor='white')
for i, row in risk_merged.iterrows():
    axes[1].text(i, row['count'] + 10, f"{row['count']:,}",
                 ha='center', fontsize=10)
axes[1].set_title('Patient Count by Risk Score')
axes[1].set_xlabel('Risk Score')
axes[1].set_ylabel('Count')

plt.suptitle('Compounded Risk Factor Analysis\n'
             '(Age>60 + Hypertension + Heart Disease + High Glucose + BMI>30)',
             fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_risk_factors.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/09_risk_factors.png")


# ═══════════════════════════════════════════════════════════
#  3.6  CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════
print("\n── 3.6  Correlation Heatmap (numeric) ───────────────")
# Encode categoricals for correlation
df_corr = df.copy()
le_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in le_cols:
    df_corr[col] = le.fit_transform(df_corr[col].astype(str))

drop_cols = ['age_group', 'glucose_category', 'risk_score']
df_corr.drop(columns=[c for c in drop_cols if c in df_corr.columns], inplace=True)

corr = df_corr.corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax,
            annot_kws={'size': 10})
ax.set_title('Feature Correlation Heatmap', fontsize=16, pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/10_correlation_heatmap.png")

# Stroke correlations specifically
print("\n── Correlation with 'stroke' target ─────────────────")
stroke_corr = corr['stroke'].drop('stroke').sort_values(key=abs, ascending=False)
print(stroke_corr.round(4))


# ═══════════════════════════════════════════════════════════
#  3.7  PAIRPLOT (top features)
# ═══════════════════════════════════════════════════════════
print("\n── 3.7  Pairplot (top numeric features) ─────────────")
pair_df = df[['age', 'avg_glucose_level', 'bmi', 'stroke']].copy()
pair_df['stroke'] = pair_df['stroke'].map({0: 'No Stroke', 1: 'Stroke'})

g = sns.pairplot(pair_df, hue='stroke',
                 palette={'No Stroke': STROKE_COLORS[0], 'Stroke': STROKE_COLORS[1]},
                 plot_kws={'alpha': 0.4, 's': 20},
                 diag_kind='kde')
g.fig.suptitle('Pairplot — Age / Glucose / BMI vs Stroke', y=1.02, fontsize=14)
plt.savefig(f'{OUTPUT_DIR}/11_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/11_pairplot.png")


print("\n\n✅  Phase 3 complete — proceed to phase4_model_training.py")
