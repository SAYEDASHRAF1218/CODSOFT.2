💳 Fraud Detection using Machine Learning

This project predicts fraudulent transactions using machine learning models on financial transaction data. The dataset is highly imbalanced, so techniques like SMOTE oversampling and feature scaling are applied to improve performance.

📌 Features

Handles class imbalance using SMOTE

Feature scaling with StandardScaler

Implements three machine learning models:

Logistic Regression

Decision Tree

Random Forest

Evaluates models with:

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

ROC-AUC Score

📂 Dataset

Training shape: 129,668 rows × 23 columns

Test shape: 55,572 rows × 23 columns

Target variable: is_fraud (binary classification)

🛠 Dependencies

Install the required Python libraries:

pandas
numpy
scikit-learn
imbalanced-learn


Install all at once:

pip install pandas numpy scikit-learn imbalanced-learn

▶️ How to Run
Option 1: Google Colab

Upload fraudd_final.ipynb and the dataset (fraudTrain.csv, fraudTest.csv)

Run all cells sequentially

Option 2: Local Jupyter Notebook
git clone <your-repo-link>
cd <repo-folder>
jupyter notebook fraudd_final.ipynb

📊 Model Performance
Model	Accuracy	ROC-AUC
Logistic Regression	95%	0.874
Decision Tree	97%	0.672
Random Forest	99%	0.707
Key Observations:

Random Forest achieved the best overall accuracy (99%) but struggled with recall on the fraud class.

Logistic Regression had the best balance with ROC-AUC = 0.874.

Severe class imbalance impacts minority class performance even after SMOTE.
