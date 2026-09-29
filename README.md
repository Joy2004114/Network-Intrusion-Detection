# Network Intrusion Detection using Machine Learning and Deep Learning

This repository presents a complete pipeline for detecting network intrusions using both classical Machine Learning (ML) and Deep Learning (DL) models. The system includes data preprocessing, SMOTE-based class balancing, model training and evaluation, performance comparison, and SHAP-based Explainable AI (XAI) analysis.

---

## Dataset

- **Name**: Friday-WorkingHours-Morning.pcap_ISCX.csv  
- **Source**: [Kaggle - Network Intrusion Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)  
- **Target Column**: `Label` (binary: Benign vs Attack)  
- **Problem Type**: Binary Classification  

---
## Pipeline Overview

1. **Data Preprocessing**

   * Label encoding and column renaming
   * Missing value imputation
   * Infinite value handling
   * SMOTE oversampling for class imbalance
   * Feature scaling using `StandardScaler`

2. **Model Training**

   * **Machine Learning Models**: Random Forest, Logistic Regression, SVM, Naive Bayes, XGBoost
   * **Deep Learning Models**: ANN, Transformer-style FFNN, GRU, Autoencoder, Deep Neural Network

3. **Model Evaluation**

   * Metrics: Accuracy, R² Score, Mean Absolute Error (MAE), Mean Squared Error (MSE), F1 Score
   * Additional classification metrics: Precision, Recall, Specificity, Balanced Accuracy, ROC-AUC, PR-AUC, and MCC
   * Confusion matrix analysis
   * ROC and Precision-Recall curve comparison

4. **Visualization**

   * Label distribution before and after SMOTE
   * Bar plots for R² Score and MSE comparisons
   * Confusion matrices for all models
   * ROC and Precision-Recall curves
   * Deep Learning training and validation loss curves

5. **Explainable AI (XAI)**

   * SHAP-based model explainability
   * Global feature importance analysis
   * Local explanation of individual predictions
   * SHAP analysis for Random Forest and XGBoost
   * Identification of important network-flow features contributing to model predictions



## Results

### Machine Learning Models

| Model               | Accuracy | R² Score | MAE    | MSE    | F1 Score |
| ------------------- | -------- | -------- | ------ | ------ | -------- |
| Random Forest       | 0.9991   | 0.9819   | 0.0009 | 0.0009 | 0.9579   |
| Logistic Regression | 0.9561   | 0.9736   | 0.0439 | 0.0439 | 0.3176   |
| SVM                 | 0.9660   | 0.9820   | 0.0340 | 0.0340 | 0.3770   |
| Naive Bayes         | 0.7946   | 0.8962   | 0.2054 | 0.2054 | 0.0911   |
| XGBoost             | 0.9993   | 0.9963   | 0.0007 | 0.0007 | 0.9678   |

### Deep Learning Models

| Model       | Accuracy | R² Score | MAE    | MSE    | F1 Score |
| ----------- | -------- | -------- | ------ | ------ | -------- |
| ANN         | 0.9761   | 0.9837   | 0.0239 | 0.0239 | 0.4603   |
| Transformer | 0.9757   | 0.9843   | 0.0243 | 0.0243 | 0.4566   |
| GRU         | 0.9755   | 0.9868   | 0.0245 | 0.0245 | 0.4561   |
| Autoencoder | 0.9735   | 0.9833   | 0.0265 | 0.0265 | 0.4358   |
| DNN         | 0.9770   | 0.9825   | 0.0230 | 0.0230 | 0.4690   |

### Additional Evaluation

* **Precision, Recall, Specificity, Balanced Accuracy**
* **ROC-AUC, PR-AUC, and MCC**
* **Confusion Matrix Analysis**
* **ROC and Precision-Recall Curves**

### Explainable AI (XAI)

* SHAP analysis was performed for **Random Forest** and **XGBoost**.
* Important features included **Destination Port**, **Init_Win_bytes_forward**, **Init_Win_bytes_backward**, **Flow IAT Min**, and **Bwd Packets/s**.
* SHAP summary and local waterfall plots were used to explain model predictions.


## Visualizations

* **Label Distribution**: Before and after applying SMOTE
* **Model Comparison**:

  * Bar plots for R² Score and MSE
  * ML vs DL model performance visualized using Matplotlib
* **Model Evaluation**:

  * Confusion matrices for all models
  * ROC curves and Precision-Recall curves
  * Deep Learning training and validation loss curves
* **Explainable AI (XAI)**:

  * SHAP summary plots
  * SHAP feature importance analysis
  * SHAP waterfall plots for individual predictions


## Notes

* All models are trained on the same split (70% train, 30% test).
* SMOTE is only applied to the training set to prevent data leakage.
* Feature scaling is performed using `StandardScaler`.
* DL models use `EarlyStopping` to avoid overfitting.
* GRU input may require reshaping to 3D format for compatibility.
* Additional evaluation includes Precision, Recall, Specificity, Balanced Accuracy, ROC-AUC, PR-AUC, and MCC.
* SHAP is used for explainability and feature importance analysis of Random Forest and XGBoost.

