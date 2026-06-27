# ============================================================
#  BRAIN STROKE PREDICTION  |  Phase 5 — Inference Pipeline
#  Use trained models to predict stroke risk for new patients
# ============================================================

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = 'saved_models'


# ═══════════════════════════════════════════════════════════
#  LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════
scaler        = joblib.load(f'{MODELS_DIR}/scaler.pkl')
feature_names = joblib.load(f'{MODELS_DIR}/feature_names.pkl')

# Load best models for ensemble / individual predictions
models = {
    'Random Forest'     : joblib.load(f'{MODELS_DIR}/Random_Forest.pkl'),
    'XGBoost'           : joblib.load(f'{MODELS_DIR}/XGBoost.pkl'),
    'Gradient Boosting' : joblib.load(f'{MODELS_DIR}/Gradient_Boosting.pkl'),
}


# ═══════════════════════════════════════════════════════════
#  PREPROCESSING FUNCTION  (mirrors Phase 4 exactly)
# ═══════════════════════════════════════════════════════════
def preprocess_patient(patient_dict: dict) -> np.ndarray:
    """
    Takes a raw patient dict and returns a scaled feature vector
    ready for model inference.

    Parameters
    ----------
    patient_dict : dict  with keys:
        gender          : 'Male' | 'Female'
        age             : float
        hypertension    : 0 | 1
        heart_disease   : 0 | 1
        ever_married    : 'Yes' | 'No'
        work_type       : 'Private' | 'Self-employed' | 'Govt_job'
                          | 'children' | 'Never_worked'
        Residence_type  : 'Urban' | 'Rural'
        avg_glucose_level : float
        bmi             : float
        smoking_status  : 'formerly smoked' | 'never smoked'
                          | 'smokes' | 'Unknown'

    Returns
    -------
    np.ndarray : scaled (1, n_features) array
    """
    p = patient_dict.copy()

    # ── Binary encodings ──────────────────────────────────
    p['gender']         = 1 if p['gender'] == 'Male'  else 0
    p['ever_married']   = 1 if p['ever_married'] == 'Yes' else 0
    p['Residence_type'] = 1 if p['Residence_type'] == 'Urban' else 0

    # ── Derived / interaction features ───────────────────
    p['age_group_num']    = (0 if p['age'] <= 20 else
                             1 if p['age'] <= 35 else
                             2 if p['age'] <= 50 else
                             3 if p['age'] <= 65 else 4)
    p['glucose_cat_num']  = (0 if p['avg_glucose_level'] <= 100 else
                             1 if p['avg_glucose_level'] <= 125 else
                             2 if p['avg_glucose_level'] <= 200 else 3)
    p['bmi_cat_num']      = (0 if p['bmi'] < 18.5 else
                             1 if p['bmi'] < 25.0 else
                             2 if p['bmi'] < 30.0 else 3)
    p['risk_score']       = (int(p['age'] > 60) + p['hypertension'] +
                             p['heart_disease'] +
                             int(p['avg_glucose_level'] > 125) +
                             int(p['bmi'] > 30))
    p['age_glucose']      = p['age'] * p['avg_glucose_level']

    # ── One-hot: work_type ────────────────────────────────
    work_types = ['Govt_job', 'Never_worked', 'Private',
                  'Self-employed', 'children']
    for wt in work_types:
        p[f'work_type_{wt}'] = 1 if p.get('work_type') == wt else 0

    # ── One-hot: smoking_status ───────────────────────────
    smoke_cats = ['Unknown', 'formerly smoked', 'never smoked', 'smokes']
    for sc in smoke_cats:
        p[f'smoking_status_{sc}'] = 1 if p.get('smoking_status') == sc else 0

    # ── Drop raw categorical fields ───────────────────────
    for key in ['work_type', 'smoking_status']:
        p.pop(key, None)

    # ── Build ordered feature vector ─────────────────────
    row = pd.DataFrame([p], columns=feature_names).fillna(0)
    return scaler.transform(row)


# ═══════════════════════════════════════════════════════════
#  PREDICT FUNCTION
# ═══════════════════════════════════════════════════════════
def predict_stroke_risk(patient_dict: dict,
                         threshold: float = 0.35) -> dict:
    """
    Returns stroke probability and risk level for a patient.

    Parameters
    ----------
    patient_dict : dict  (see preprocess_patient for keys)
    threshold    : float — decision boundary (default 0.35 for higher recall)

    Returns
    -------
    dict with keys: probabilities, ensemble_prob, prediction, risk_level
    """
    X_scaled = preprocess_patient(patient_dict)
    probs    = {}
    for name, model in models.items():
        probs[name] = round(model.predict_proba(X_scaled)[0][1], 4)

    ensemble_prob = round(np.mean(list(probs.values())), 4)
    prediction    = int(ensemble_prob >= threshold)

    if ensemble_prob < 0.20:
        risk_level = '🟢 LOW RISK'
    elif ensemble_prob < 0.40:
        risk_level = '🟡 MODERATE RISK'
    elif ensemble_prob < 0.60:
        risk_level = '🟠 HIGH RISK'
    else:
        risk_level = '🔴 VERY HIGH RISK'

    return {
        'individual_probs' : probs,
        'ensemble_prob'    : ensemble_prob,
        'prediction'       : prediction,
        'risk_level'       : risk_level,
    }


# ═══════════════════════════════════════════════════════════
#  DEMO PREDICTIONS
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  BRAIN STROKE PREDICTION — Phase 5: Inference Demo")
print("=" * 60)

test_cases = [
    # Case 1 – Low risk: young, healthy
    {
        'label'             : 'Case 1 — Young Healthy Male',
        'gender'            : 'Male',
        'age'               : 27.0,
        'hypertension'      : 0,
        'heart_disease'     : 0,
        'ever_married'      : 'No',
        'work_type'         : 'Private',
        'Residence_type'    : 'Urban',
        'avg_glucose_level' : 88.0,
        'bmi'               : 22.5,
        'smoking_status'    : 'never smoked',
    },
    # Case 2 – Moderate risk: middle-aged, hypertensive
    {
        'label'             : 'Case 2 — Middle-aged Hypertensive Female',
        'gender'            : 'Female',
        'age'               : 52.0,
        'hypertension'      : 1,
        'heart_disease'     : 0,
        'ever_married'      : 'Yes',
        'work_type'         : 'Self-employed',
        'Residence_type'    : 'Rural',
        'avg_glucose_level' : 118.0,
        'bmi'               : 28.4,
        'smoking_status'    : 'formerly smoked',
    },
    # Case 3 – High risk: elderly, multiple comorbidities
    {
        'label'             : 'Case 3 — Elderly High-Risk Male',
        'gender'            : 'Male',
        'age'               : 72.0,
        'hypertension'      : 1,
        'heart_disease'     : 1,
        'ever_married'      : 'Yes',
        'work_type'         : 'Private',
        'Residence_type'    : 'Urban',
        'avg_glucose_level' : 195.0,
        'bmi'               : 34.2,
        'smoking_status'    : 'smokes',
    },
]

print()
for case in test_cases:
    label  = case.pop('label')
    result = predict_stroke_risk(case, threshold=0.35)
    print(f"─── {label} ───")
    for model_name, prob in result['individual_probs'].items():
        print(f"  {model_name:22s}: {prob:.2%}")
    print(f"  {'Ensemble Probability':22s}: {result['ensemble_prob']:.2%}")
    print(f"  {'Prediction':22s}: {'STROKE' if result['prediction'] == 1 else 'NO STROKE'}")
    print(f"  {'Risk Level':22s}: {result['risk_level']}")
    print()


# ═══════════════════════════════════════════════════════════
#  BATCH PREDICTION FROM CSV
# ═══════════════════════════════════════════════════════════
def batch_predict(input_csv: str, output_csv: str,
                   threshold: float = 0.35) -> pd.DataFrame:
    """
    Run predictions on a batch CSV (same schema as training data, no 'stroke' col).
    Appends columns: stroke_probability, stroke_prediction, risk_level.
    """
    df      = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        patient = row.to_dict()
        patient.pop('id', None)
        patient.pop('stroke', None)
        res = predict_stroke_risk(patient, threshold)
        results.append({
            'ensemble_prob' : res['ensemble_prob'],
            'prediction'    : res['prediction'],
            'risk_level'    : res['risk_level'],
        })
    out_df = pd.concat([df, pd.DataFrame(results)], axis=1)
    out_df.to_csv(output_csv, index=False)
    print(f"\n✅  Batch predictions saved → {output_csv}")
    return out_df


print("\n✅  Phase 5 complete.")
print("   Call predict_stroke_risk(patient_dict) for single inference.")
print("   Call batch_predict(input_csv, output_csv) for bulk inference.")
