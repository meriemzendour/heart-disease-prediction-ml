# Application Flask : Prédiction Maladies Cardiaques
## Structure du projet

```
flask_app/
├── app.py                  ← Application Flask principale
├── templates/
│   ├── index.html          ← Formulaire de saisie des données
│   └── result.html         ← Page de résultat
├── model/                  ← Dossier créé par le script d'analyse
│   ├── best_model.pkl      ← Meilleur modèle entraîné (joblib)
│   ├── scaler.pkl          ← StandardScaler ajusté
│   └── feature_names.pkl   ← Noms des 13 features
└── README.md
```

## Installation et lancement

### Étape 1 — Installer les dépendances
```bash
pip install flask joblib scikit-learn pandas numpy imbalanced-learn
```

### Étape 2 — Générer le modèle
Placez `heart_cleveland_upload.csv` dans le dossier parent, puis :
```bash
cd ..
python heart_disease_project_v2_corrected.py
```
Cela génère `outputs/best_model.pkl`, `outputs/scaler.pkl`, `outputs/feature_names.pkl`.

### Étape 3 — Copier les fichiers de modèle
```bash
mkdir -p flask_app/model
cp outputs/best_model.pkl flask_app/model/
cp outputs/scaler.pkl     flask_app/model/
cp outputs/feature_names.pkl flask_app/model/
```

### Étape 4 — Lancer l'application
```bash
cd flask_app
python app.py
```

### Étape 5 — Accéder à l'interface
Ouvrez votre navigateur à l'adresse : http://localhost:5000

## Variables d'entrée (identiques au code d'analyse)

| Variable  | Type    | Description                          | Valeurs |
|-----------|---------|--------------------------------------|---------|
| age       | int     | Âge du patient                       | 20–80   |
| sex       | int     | Sexe                                 | 0=F, 1=H|
| cp        | int     | Type de douleur thoracique           | 0–3     |
| trestbps  | float   | Tension au repos (mm Hg)             | 80–210  |
| chol      | float   | Cholestérol (mg/dl)                  | 100–600 |
| fbs       | int     | Glycémie > 120 mg/dl                 | 0/1     |
| restecg   | int     | ECG au repos                         | 0–2     |
| thalach   | float   | Fréquence cardiaque max (bpm)        | 60–210  |
| exang     | int     | Angine d'effort                      | 0/1     |
| oldpeak   | float   | Dépression segment ST                | 0–7     |
| slope     | int     | Pente segment ST                     | 0–2     |
| ca        | float   | Vaisseaux colorés (fluoroscopie)     | 0–3     |
| thal      | float   | Thalassémie                          | 3/6/7   |

## Niveaux de risque (identiques au code d'analyse)

| Probabilité | Niveau  |
|-------------|---------|
| < 30%       | FAIBLE  |
| 30% – 60%   | MODÉRÉ  |
| > 60%       | ÉLEVÉ   |

## Avertissement
Cette application est un outil académique d'aide à la décision.
Elle ne remplace pas un diagnostic médical professionnel.
