# Heart Disease Prediction Using Machine Learning

## Overview

This project uses Machine Learning algorithms to predict the presence of heart disease based on clinical data.

It includes data analysis, data preprocessing, model training, model evaluation, and a web application developed with Flask.

## Technologies Used

- **Python**
- **Pandas and NumPy**
- **Scikit-learn**
- **Imbalanced-learn**
- **Matplotlib and Seaborn**
- **Flask**
- **HTML and CSS**
- **Jupyter Notebook**

## Project Objectives

- Analyze clinical data related to heart disease.
- Clean and preprocess the dataset.
- Explore data using statistical analysis and visualizations.
- Train and compare different Machine Learning classification models.
- Evaluate model performance using appropriate metrics.
- Develop a web application to make predictions using the trained model.

## Machine Learning Models

The project explores the following classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gradient Boosting

## Project Structure

```text
heart-disease-prediction-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── heart_cleveland.arff
│   └── heart_cleveland_upload.csv
├── notebooks/
│   └── heart_disease_analysis.ipynb
├── src/
│   └── heart_disease_final.py
├── app/
│   ├── app.py
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   └── model and preprocessing files
└── outputs/
    ├── visualizations
    └── trained model files
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/meriemzendour/heart-disease-prediction-ml.git
cd heart-disease-prediction-ml
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Web Application

From the root directory of the project, run:

```bash
python app/app.py
```

If the application starts successfully, open the following address in your browser:

```text
http://127.0.0.1:5000
```

Make sure that the trained model and preprocessing files are present at the locations expected by the Flask application.

## Results and Visualizations

The project includes visualizations to support data analysis and model evaluation, including:

- Descriptive data analysis.
- Confusion matrices.
- ROC curves.
- Model performance comparison.
- Feature importance and Brier score analysis.
- Learning curves.

Quantitative performance metrics should be reported after verifying the actual evaluation results.

## Disclaimer

This project is intended for educational and research purposes only. Its predictions are not medical diagnoses and must not replace advice, diagnosis, or treatment provided by qualified healthcare professionals.

## Author

**Meriem Zendour**

Master's Student in Data Science and Intelligent Systems.

Interests: Machine Learning, Artificial Intelligence, and Data Science.
