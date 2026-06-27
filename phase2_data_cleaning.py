# ============================================================
#  BRAIN STROKE PREDICTION  |  Phase 2 — Data Cleaning
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
print("  BRAIN STROKE PREDICTION — Phase 2: Data Cleaning")
print("=" * 60)


# ═══════════════════════════════════════════════════════════
#  LOAD RAW DATA
# ═══════════════════════════════════════════════════════════
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
print(f"\n📂  Raw shape: {df.shape}")


# ═══════════════════════════════════════════════════════════
#  STEP 1 — Drop ID column (no predictive value)
# ═══════════════════════════════════════════════════════════
print("\n── Step 1: Drop 'id' column ──────────────────────────")
df.drop(columns=['id'], inplace=True)
print(f"  Remaining columns: {df.columns.tolist()}")


# ═══════════════════════════════════════════════════════════
#  STEP 2 — Handle 'gender' → Remove 'Other' (only 1 row)
# ═══════════════════════════════════════════════════════════
print("\n── Step 2: Gender value counts ───────────────────────")
print(df['gender'].value_counts())
other_mask = df['gender'] == 'Other'
print(f"  Rows with gender='Other': {other_mask.sum()}")
df = df[~other_mask]
print(f"  Shape after removing 'Other' gender: {df.shape}")


# ═══════════════════════════════════════════════════════════
#  STEP 3 — Handle missing BMI (3.93% null)
# ═══════════════════════════════════════════════════════════
print("\n── Step 3: Impute missing BMI ────────────────────────")
print(f"  Null BMI before: {df['bmi'].isnull().sum()}")

# Plot BMI distribution before imputation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['bmi'].dropna(), bins=40, color='steelblue', edgecolor='white')
axes[0].set_title('BMI Distribution BEFORE Imputation')
axes[0].set_xlabel('BMI')
axes[0].set_ylabel('Count')

# Impute with median (BMI is right-skewed; median is more robust)
bmi_median = df['bmi'].median()
df['bmi'].fillna(bmi_median, inplace=True)
print(f"  Null BMI after : {df['bmi'].isnull().sum()}")
print(f"  Imputed with median BMI = {bmi_median:.2f}")

axes[1].hist(df['bmi'], bins=40, color='seagreen', edgecolor='white')
axes[1].axvline(bmi_median, color='red', linestyle='--', label=f'Median = {bmi_median:.1f}')
axes[1].set_title('BMI Distribution AFTER Median Imputation')
axes[1].set_xlabel('BMI')
axes[1].legend()

plt.suptitle('BMI — Before vs After Imputation', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_bmi_imputation.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/03_bmi_imputation.png")


# ═══════════════════════════════════════════════════════════
#  STEP 4 — Outlier Detection & Treatment (IQR method)
# ═══════════════════════════════════════════════════════════
print("\n── Step 4: Outlier Detection (IQR) ───────────────────")
numeric_cols = ['age', 'avg_glucose_level', 'bmi']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for idx, col in enumerate(numeric_cols):
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    print(f"  {col:22s}  Q1={Q1:.2f}  Q3={Q3:.2f}  IQR={IQR:.2f}  "
          f"Bounds=[{lower:.2f}, {upper:.2f}]  Outliers={outliers}")

    # Boxplot before
    axes[0, idx].boxplot(df[col].dropna(), vert=False, patch_artist=True,
                         boxprops=dict(facecolor='#4C72B0', alpha=0.6))
    axes[0, idx].set_title(f'{col} — Before Capping')
    axes[0, idx].set_xlabel(col)

    # Cap outliers (Winsorization)
    df[col] = df[col].clip(lower=lower, upper=upper)

    # Boxplot after
    axes[1, idx].boxplot(df[col], vert=False, patch_artist=True,
                         boxprops=dict(facecolor='#DD8452', alpha=0.6))
    axes[1, idx].set_title(f'{col} — After Capping')
    axes[1, idx].set_xlabel(col)

plt.suptitle('Outlier Treatment — IQR Winsorization', fontsize=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_outlier_treatment.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/04_outlier_treatment.png")


# ═══════════════════════════════════════════════════════════
#  STEP 5 — Duplicate Check
# ═══════════════════════════════════════════════════════════
print("\n── Step 5: Duplicate Check ───────────────────────────")
dups = df.duplicated().sum()
print(f"  Duplicate rows: {dups}")
if dups > 0:
    df.drop_duplicates(inplace=True)
    print(f"  Removed duplicates. New shape: {df.shape}")


# ═══════════════════════════════════════════════════════════
#  STEP 6 — Data Type Verification & Minor Fixes
# ═══════════════════════════════════════════════════════════
print("\n── Step 6: Data Types ────────────────────────────────")
# Ensure binary columns are int
df['hypertension']  = df['hypertension'].astype(int)
df['heart_disease'] = df['heart_disease'].astype(int)
df['stroke']        = df['stroke'].astype(int)
print(df.dtypes)


# ═══════════════════════════════════════════════════════════
#  STEP 7 — Save Clean DataFrame
# ═══════════════════════════════════════════════════════════
df.to_csv('stroke_clean.csv', index=False)
print(f"\n✅  Clean dataset saved → stroke_clean.csv")
print(f"   Final shape: {df.shape}")
print(f"\n── Clean Data Preview ────────────────────────────────")
print(df.head())
print(f"\n── Null Check (should be 0) ──────────────────────────")
print(df.isnull().sum())

print("\n\n✅  Phase 2 complete — proceed to phase3_eda.py")
