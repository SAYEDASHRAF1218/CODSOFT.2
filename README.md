# Credit Card Fraud Detection

This project implements machine learning models to detect fraudulent credit card transactions using the **Fraud Detection Dataset**.  
Due to the high class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to balance the dataset before training.

---

## 📌 Dataset
- **Train set shape:** (6226, 23)  
- **Test set shape:** (4667, 23)  
- Target column: `is_fraud` (0 = Legitimate, 1 = Fraud)  

---

## ⚙️ Workflow
1. Import required libraries  
2. Load training and testing datasets  
3. Remove non-numerical columns  
4. Handle class imbalance using **SMOTE**  
5. Scale features with **StandardScaler**  
6. Train multiple models:  
   - Logistic Regression  
   - Decision Tree  
   - Random Forest  
7. Evaluate results with:  
   - Confusion Matrix  
   - Classification Report  
   - ROC-AUC Score  

---

## 🧪 Results

### Logistic Regression
```
Confusion Matrix:
[[   0 4646]
 [   0   21]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.00      0.00      0.00      4646
         1.0       0.00      1.00      0.01        21

    accuracy                           0.00      4667
   macro avg       0.00      0.50      0.00      4667
weighted avg       0.00      0.00      0.00      4667

ROC-AUC Score: 0.5
```

### Decision Tree
```
Confusion Matrix:
[[4642    4]
 [  21    0]]

Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      4646
         1.0       0.00      0.00      0.00        21

    accuracy                           0.99      4667
   macro avg       0.50      0.50      0.50      4667
weighted avg       0.99      0.99      0.99      4667

ROC-AUC Score: 0.4995
```

### Random Forest
```
Confusion Matrix:
[[4646    0]
 [  21    0]]

Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      4646
         1.0       0.00      0.00      0.00        21

    accuracy                           1.00      4667
   macro avg       0.50      0.50      0.50      4667
weighted avg       0.99      1.00      0.99      4667

ROC-AUC Score: 0.5
```

---

## 📊 Observations
- Logistic Regression failed to classify legitimate transactions (all predicted as fraud).  
- Decision Tree and Random Forest classified legitimate transactions well but **completely failed on fraud detection** due to extreme class imbalance.  
- ROC-AUC scores are ~0.5, indicating **poor discrimination ability**.  

---

## 🚀 Future Improvements
- Use **ensemble methods** like XGBoost or LightGBM.  
- Apply **advanced resampling techniques** (e.g., SMOTEENN, ADASYN).  
- Engineer additional fraud-related features.  
- Try **deep learning models** for anomaly detection.

---
