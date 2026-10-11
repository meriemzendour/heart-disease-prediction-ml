"""
=============================================================
 APPLICATION WEB FLASK : PRÉDICTION MALADIES CARDIAQUES
 CRISP-DM — Phase Déploiement (VERSION FINALE — 16 FEATURES)
=============================================================
Cette application utilise STRICTEMENT :
  - Le même modèle entraîné (best_model.pkl)
  - Les 13 features cliniques de base
  - + 3 features engineered : chol_thalach_ratio, bp_age_product, age_group
  - Le même SimpleImputer (médiane) ajusté sur X_train
  - Le même StandardScaler ajusté sur X_train imputé
  - La même logique de prédiction et niveaux de risque
=============================================================
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ── CHARGEMENT DES ARTEFACTS ──────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, 'best_model.pkl')
SCALER_PATH   = os.path.join(BASE_DIR, 'scaler.pkl')
IMPUTER_PATH  = os.path.join(BASE_DIR, 'imputer.pkl')
FEAT_PATH     = os.path.join(BASE_DIR, 'feature_names.pkl')

try:
    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    imputer       = joblib.load(IMPUTER_PATH)
    feature_names = joblib.load(FEAT_PATH)   # liste de 16 noms

    MODEL_LOADED = True
    MODEL_NAME   = type(model).__name__

    print(f"✅ Modèle chargé        : {MODEL_NAME}")
    print(f"   Features attendues   : {len(feature_names)} → {feature_names}")
    print(f"   Scaler n_features_in : {scaler.n_features_in_}")

except FileNotFoundError as e:
    MODEL_LOADED = False
    MODEL_NAME   = "Non chargé"
    print(f"⚠️  Fichier introuvable : {e}")


# ── FEATURES DE BASE (13 variables cliniques saisies par l'utilisateur) ───────
BASE_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]


# ── FEATURE ENGINEERING (identique au script d'entraînement) ─────────────────
def appliquer_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les 3 variables dérivées dans le même ordre que l'entraînement."""
    df = df.copy()
    df['chol_thalach_ratio'] = df['chol'] / df['thalach']
    df['bp_age_product']     = df['trestbps'] * df['age']
    df['age_group']          = pd.cut(
        df['age'],
        bins=[0, 40, 50, 60, 100],
        labels=[0, 1, 2, 3]
    ).astype(float)
    return df


# ── PRÉDICTION ────────────────────────────────────────────────────────────────
def predire(age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak, slope, ca, thal):
    """
    Pipeline complet :
      1. Créer le DataFrame avec les 13 features de base
      2. Appliquer le feature engineering → 16 features
      3. Imputation médiane (imputer ajusté sur X_train)
      4. Normalisation StandardScaler (ajusté sur X_train imputé)
      5. Prédiction + probabilité
    """
    # 1 — DataFrame base
    donnees = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs,
          restecg, thalach, exang, oldpeak, slope, ca, thal]],
        columns=BASE_FEATURES
    )

    # 2 — Feature engineering
    donnees = appliquer_feature_engineering(donnees)

    # S'assurer que l'ordre des colonnes correspond à feature_names
    donnees = donnees[feature_names]

    print("\n===== DONNÉES ENVOYÉES AU MODÈLE =====")
    print(donnees.T)
    print(f"Colonnes ({donnees.shape[1]}) : {list(donnees.columns)}")
    print("======================================\n")

    # 3 — Imputation
    donnees_imp = imputer.transform(donnees)

    # 4 — Normalisation
    donnees_sc = scaler.transform(donnees_imp)

    # 5 — Prédiction
    prediction  = model.predict(donnees_sc)[0]
    probabilite = float(model.predict_proba(donnees_sc)[0][1])

    # Niveau de risque (identique au script)
    if probabilite < 0.3:
        niveau       = "FAIBLE"
        niveau_class = "low"
    elif probabilite < 0.6:
        niveau       = "MODÉRÉ"
        niveau_class = "medium"
    else:
        niveau       = "ÉLEVÉ"
        niveau_class = "high"

    # Variables dérivées (pour affichage dans result.html)
    chol_thalach_ratio = round(chol / thalach, 3)
    bp_age_product     = int(trestbps * age)
    age_group_val      = 0 if age <= 40 else (1 if age <= 50 else (2 if age <= 60 else 3))

    return {
        'prediction'         : int(prediction),
        'probabilite'        : round(probabilite * 100, 1),
        'niveau'             : niveau,
        'niveau_class'       : niveau_class,
        'malade'             : int(prediction) == 1,
        'chol_thalach_ratio' : chol_thalach_ratio,
        'bp_age_product'     : bp_age_product,
        'age_group'          : age_group_val,
    }


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           model_name=MODEL_NAME,
                           model_loaded=MODEL_LOADED)


@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': "Modèle non chargé. Exécutez heart_disease_final.py en premier."}), 503

    try:
        data   = request.form
        result = predire(
            age      = float(data['age']),
            sex      = int(data['sex']),
            cp       = int(data['cp']),
            trestbps = float(data['trestbps']),
            chol     = float(data['chol']),
            fbs      = int(data['fbs']),
            restecg  = int(data['restecg']),
            thalach  = float(data['thalach']),
            exang    = int(data['exang']),
            oldpeak  = float(data['oldpeak']),
            slope    = int(data['slope']),
            ca       = float(data['ca']),
            thal     = float(data['thal']),
        )
        return render_template('result.html',
                               result=result,
                               form_data=data,
                               model_name=MODEL_NAME)
    except Exception as e:
        return render_template('index.html',
                               error=str(e),
                               model_name=MODEL_NAME,
                               model_loaded=MODEL_LOADED)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
