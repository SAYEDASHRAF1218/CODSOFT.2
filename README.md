# CODSOFT.2
📌 Internship Task: Customer Churn Prediction

🎯 Objective

Develop a machine learning model to predict whether a customer will churn (exit) or stay with a bank using historical customer data. This project simulates a real-world business problem faced by subscription-based companies.


---

📂 Dataset

Source: Bank Customer Churn Prediction - Kaggle

File: Churn_Modelling.csv

Target Column: Exited (1 = Churned, 0 = Retained)

Features: Customer demographics (Age, Gender, Geography), account information (Balance, CreditScore, Tenure), and product usage (NumOfProducts, IsActiveMember, etc.)



✅ Requirements / Deliverables

1. Data Loading & Preprocessing

Load the dataset.

Remove unnecessary identifier columns (RowNumber, CustomerId, Surname).

Encode categorical variables (Geography, Gender).

Scale numerical features.



2. Exploratory Data Analysis (EDA)

Explore feature distributions.

Analyze churn rate.

Identify correlations between features and target.



3. Handle Class Imbalance

Apply SMOTE to balance churn vs. non-churn customers.



4. Model Training

Train at least three models:

Logistic Regression

Random Forest

Gradient Boosting




5. Model Evaluation

Classification Report (Precision, Recall, F1-score)

Confusion Matrix

ROC-AUC Score

Precision-Recall Curve



6. Model Persistence

Save the best performing model using joblib.

Save the scaler for future inference.



7. Documentation

A well-structured README.md summarizing:

Problem statement

Dataset description

Approach taken

Results and metrics

Future improvements

Expected Output

Clean and modular Python code (script or Jupyter Notebook).

Visualizations for EDA and evaluation metrics.

A saved best model (best_churn_model.joblib) and scaler (scaler_churn.joblib).

A README file describing the project and results.


Example Code Skeleton

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv('data/Churn_Modelling.csv')

# Drop ID columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Features and Target
X = df.drop(columns=['Exited'])
y = df['Exited']

# One-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Handle imbalance
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train models
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, random_state=42)

models = {'Logistic Regression': log_reg, 'Random Forest': rf, 'Gradient Boosting': gb}

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    preds = model.predict(X_test)
    print(f"\n{name} Results:")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# Save best model (example: Gradient Boosting)
joblib.dump(gb, 'best_churn_model.joblib')
joblib.dump(scaler, 'scaler_churn.joblib')


🔮 Future Improvements

Try advanced models like XGBoost, LightGBM, CatBoost.

Apply hyperparameter tuning (GridSearch/RandomizedSearch).

Add feature importance analysis.

Build a dashboard (Streamlit/Flask) to serve predictions interactively.

