# ============================================================
#  BRAIN STROKE PREDICTION  |  Phase 4 — Feature Engineering
#                                        & Model Training
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection      import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing        import LabelEncoder, StandardScaler
from sklearn.linear_model         import LogisticRegression
from sklearn.tree                 import DecisionTreeClassifier
from sklearn.ensemble             import (RandomForestClassifier,
                                           GradientBoostingClassifier,
                                           AdaBoostClassifier)
from sklearn.neighbors            import KNeighborsClassifier
from sklearn.svm                  import SVC
from sklearn.naive_bayes          import GaussianNB
from sklearn.metrics              import (accuracy_score, precision_score,
                                           recall_score, f1_score,
                                           roc_auc_score, confusion_matrix,
                                           classification_report, roc_curve,
                                           ConfusionMatrixDisplay)
from imblearn.over_sampling       import SMOTE
from xgboost                      import XGBClassifier
import joblib

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.figsize': (14, 6), 'axes.titlesize': 14})
STROKE_COLORS = {0: '#4C72B0', 1: '#DD8452'}
OUTPUT_DIR    = 'outputs'
MODELS_DIR    = 'saved_models'

import os
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


print("=" * 60)
print("  BRAIN STROKE PREDICTION — Phase 4: Model Training")
print("=" * 60)


# ═══════════════════════════════════════════════════════════
#  LOAD CLEAN DATA
# ═══════════════════════════════════════════════════════════
df = pd.read_csv('stroke_clean.csv')
print(f"\n📂  Clean data loaded → {df.shape}")


# ═══════════════════════════════════════════════════════════
#  4.1  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
print("\n── 4.1  Feature Engineering ──────────────────────────")

df_fe = df.copy()

# 1. Age groups (ordinal)
df_fe['age_group_num'] = pd.cut(df_fe['age'],
                                  bins=[0, 20, 35, 50, 65, 100],
                                  labels=[0, 1, 2, 3, 4]).astype(int)

# 2. Glucose category (ordinal)
df_fe['glucose_cat_num'] = pd.cut(df_fe['avg_glucose_level'],
                                    bins=[0, 100, 125, 200, 300],
                                    labels=[0, 1, 2, 3]).astype(int)

# 3. BMI category (ordinal)
df_fe['bmi_cat_num'] = pd.cut(df_fe['bmi'],
                                bins=[0, 18.5, 24.9, 29.9, 100],
                                labels=[0, 1, 2, 3]).astype(int)

# 4. Compounded risk score
df_fe['risk_score'] = (
    (df_fe['age'] > 60).astype(int) +
    df_fe['hypertension'] +
    df_fe['heart_disease'] +
    (df_fe['avg_glucose_level'] > 125).astype(int) +
    (df_fe['bmi'] > 30).astype(int)
)

# 5. Age × Glucose interaction
df_fe['age_glucose'] = df_fe['age'] * df_fe['avg_glucose_level']

print(f"  New features added: age_group_num, glucose_cat_num, bmi_cat_num, "
      f"risk_score, age_glucose")


# ═══════════════════════════════════════════════════════════
#  4.2  ENCODING CATEGORICAL VARIABLES
# ═══════════════════════════════════════════════════════════
print("\n── 4.2  Encoding Categoricals ────────────────────────")

le = LabelEncoder()

# Binary encode
df_fe['gender']         = le.fit_transform(df_fe['gender'])       # F=0, M=1
df_fe['ever_married']   = le.fit_transform(df_fe['ever_married']) # N=0, Y=1
df_fe['Residence_type'] = le.fit_transform(df_fe['Residence_type'])  # R=0, U=1

# One-hot encode multi-class categoricals
df_fe = pd.get_dummies(df_fe, columns=['work_type', 'smoking_status'], drop_first=False)

print(f"  Shape after encoding: {df_fe.shape}")
print(f"  All columns:\n  {list(df_fe.columns)}")


# ═══════════════════════════════════════════════════════════
#  4.3  FEATURE / TARGET SPLIT
# ═══════════════════════════════════════════════════════════
print("\n── 4.3  Feature / Target Split ───────────────────────")
X = df_fe.drop(columns=['stroke'])
y = df_fe['stroke']
print(f"  X shape: {X.shape}   |   y shape: {y.shape}")
print(f"  Target balance:\n{y.value_counts()}")


# ═══════════════════════════════════════════════════════════
#  4.4  TRAIN / TEST SPLIT (stratified)
# ═══════════════════════════════════════════════════════════
print("\n── 4.4  Train / Test Split (80/20 stratified) ────────")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train → {X_train.shape}  |  Test → {X_test.shape}")


# ═══════════════════════════════════════════════════════════
#  4.5  SMOTE — Handle Class Imbalance on TRAIN ONLY
# ═══════════════════════════════════════════════════════════
print("\n── 4.5  SMOTE Oversampling (train set) ───────────────")
print(f"  Before SMOTE: {y_train.value_counts().to_dict()}")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"  After  SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

# Visualise
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_data, title in [
    (axes[0], y_train,     'BEFORE SMOTE (Train)'),
    (axes[1], y_train_res, 'AFTER  SMOTE (Train)'),
]:
    counts = pd.Series(y_data).value_counts()
    ax.bar(['No Stroke', 'Stroke'], counts.values,
           color=[STROKE_COLORS[0], STROKE_COLORS[1]], edgecolor='white')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, f"{v:,}", ha='center', fontweight='bold')
    ax.set_title(title)
    ax.set_ylabel('Count')

plt.suptitle('Class Balance: Before vs After SMOTE', fontsize=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/12_smote_balance.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/12_smote_balance.png")


# ═══════════════════════════════════════════════════════════
#  4.6  FEATURE SCALING
# ═══════════════════════════════════════════════════════════
print("\n── 4.6  Standard Scaling ─────────────────────────────")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, f'{MODELS_DIR}/scaler.pkl')
print(f"  Scaler saved → {MODELS_DIR}/scaler.pkl")


# ═══════════════════════════════════════════════════════════
#  4.7  DEFINE MODELS
# ═══════════════════════════════════════════════════════════
print("\n── 4.7  Model Definitions ────────────────────────────")

models = {
    'Logistic Regression'      : LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'            : DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest'            : RandomForestClassifier(n_estimators=200, max_depth=8,
                                                         random_state=42, n_jobs=-1),
    'Gradient Boosting'        : GradientBoostingClassifier(n_estimators=200,
                                                             learning_rate=0.05,
                                                             max_depth=4, random_state=42),
    'XGBoost'                  : XGBClassifier(n_estimators=200, learning_rate=0.05,
                                                max_depth=5, use_label_encoder=False,
                                                eval_metric='logloss', random_state=42,
                                                n_jobs=-1),
    'AdaBoost'                 : AdaBoostClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors'      : KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    'Naive Bayes'              : GaussianNB(),
    'SVM (RBF)'                : SVC(kernel='rbf', probability=True, random_state=42),
}

print(f"  {len(models)} models to train: {list(models.keys())}")


# ═══════════════════════════════════════════════════════════
#  4.8  TRAIN & EVALUATE
# ═══════════════════════════════════════════════════════════
print("\n── 4.8  Training & Evaluation ────────────────────────")

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    # Fit
    model.fit(X_train_sc, y_train_res)

    # Predict
    y_pred      = model.predict(X_test_sc)
    y_pred_prob = model.predict_proba(X_test_sc)[:, 1]

    # CV on training set
    cv_scores = cross_val_score(model, X_train_sc, y_train_res,
                                 cv=cv, scoring='roc_auc', n_jobs=-1)

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_pred_prob)
    cv_m = cv_scores.mean()
    cv_s = cv_scores.std()

    results.append({
        'Model'         : name,
        'Accuracy'      : round(acc,  4),
        'Precision'     : round(prec, 4),
        'Recall'        : round(rec,  4),
        'F1-Score'      : round(f1,   4),
        'ROC-AUC'       : round(auc,  4),
        'CV ROC-AUC'    : round(cv_m, 4),
        'CV Std'        : round(cv_s, 4),
    })

    # Save model
    joblib.dump(model, f'{MODELS_DIR}/{name.replace(" ", "_")}.pkl')
    print(f"  ✔  {name:28s}  Acc={acc:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}  CV={cv_m:.3f}±{cv_s:.3f}")

# Save scaler feature names
joblib.dump(X.columns.tolist(), f'{MODELS_DIR}/feature_names.pkl')


# ═══════════════════════════════════════════════════════════
#  4.9  RESULTS SUMMARY TABLE
# ═══════════════════════════════════════════════════════════
print("\n── 4.9  Summary Table ────────────────────────────────")
results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
results_df.reset_index(drop=True, inplace=True)
print(results_df.to_string(index=False))
results_df.to_csv(f'{OUTPUT_DIR}/model_comparison.csv', index=False)

# Best model by ROC-AUC
best_model_name = results_df.iloc[0]['Model']
best_model      = models[best_model_name]
print(f"\n🏆  Best model (ROC-AUC): {best_model_name}")


# ═══════════════════════════════════════════════════════════
#  4.10  METRIC COMPARISON CHARTS
# ═══════════════════════════════════════════════════════════
print("\n── 4.10 Metric Comparison Charts ─────────────────────")
metrics  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 20))

for ax, metric in zip(axes, metrics):
    subset = results_df.sort_values(metric, ascending=False)
    colors = ['#C44E52' if n == best_model_name else '#4C72B0'
              for n in subset['Model']]
    bars = ax.barh(subset['Model'], subset[metric], color=colors, edgecolor='white')
    for bar, val in zip(bars, subset[metric]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center', fontsize=10)
    ax.set_xlim(0, 1.08)
    ax.set_title(f'{metric} — All Models', fontsize=13)
    ax.set_xlabel(metric)

plt.suptitle('Model Comparison — All Metrics', fontsize=16)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/13_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/13_model_comparison.png")


# ═══════════════════════════════════════════════════════════
#  4.11  ROC CURVES — All Models
# ═══════════════════════════════════════════════════════════
print("\n── 4.11 ROC Curves ───────────────────────────────────")
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

for (name, model), color in zip(models.items(), colors):
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    lw = 2.5 if name == best_model_name else 1.2
    ls = '-'  if name == best_model_name else '--'
    ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
            label=f"{name} (AUC={auc:.3f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier (AUC=0.500)')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate',  fontsize=13)
ax.set_title('ROC Curves — All Models', fontsize=15)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/14_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/14_roc_curves.png")


# ═══════════════════════════════════════════════════════════
#  4.12  CONFUSION MATRICES (Top 4 Models)
# ═══════════════════════════════════════════════════════════
print("\n── 4.12 Confusion Matrices (Top 4) ───────────────────")
top4  = results_df.head(4)['Model'].tolist()
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes  = axes.flatten()

for idx, name in enumerate(top4):
    model  = models[name]
    y_pred = model.predict(X_test_sc)
    cm     = confusion_matrix(y_test, y_pred)
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=['No Stroke', 'Stroke'])
    disp.plot(ax=axes[idx], colorbar=False, cmap='Blues')
    axes[idx].set_title(f'{name}', fontsize=12)
    axes[idx].grid(False)

plt.suptitle('Confusion Matrices — Top 4 Models', fontsize=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/15_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/15_confusion_matrices.png")


# ═══════════════════════════════════════════════════════════
#  4.13  FEATURE IMPORTANCE (Best Tree-Based Model)
# ═══════════════════════════════════════════════════════════
print("\n── 4.13 Feature Importance ───────────────────────────")
# Use Random Forest for stable importances
rf_model = models['Random Forest']
feat_imp  = pd.Series(rf_model.feature_importances_,
                       index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(feat_imp)))
feat_imp.plot(kind='barh', ax=ax, color=colors[::-1], edgecolor='white')
ax.set_title('Feature Importance — Random Forest', fontsize=15)
ax.set_xlabel('Importance Score')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/16_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  → Saved: {OUTPUT_DIR}/16_feature_importance.png")

print("\nTop 10 Features:")
print(feat_imp.head(10).round(4))


# ═══════════════════════════════════════════════════════════
#  4.14  DETAILED REPORT — Best Model
# ═══════════════════════════════════════════════════════════
print(f"\n── 4.14 Classification Report ({best_model_name}) ───")
best_preds = best_model.predict(X_test_sc)
print(classification_report(y_test, best_preds,
                             target_names=['No Stroke', 'Stroke']))


print("\n\n✅  Phase 4 complete — all models saved to saved_models/")
print(f"   Best model: {best_model_name}")
print("   Proceed to phase5_prediction.py for inference pipeline.")
