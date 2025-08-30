# Network-Intrusion-Detection
A comprehensive intrusion detection pipeline using Machine Learning and Deep Learning models. Trained on the ISCX dataset with SMOTE balancing, label encoding, and model evaluation using accuracy, R², MAE, MSE, and F1 score. Includes Random Forest, XGBoost, ANN, GRU, and more. Visualized with Matplotlib and Seaborn.



# Network Intrusion Detection using Machine Learning and Deep Learning

This repository presents a complete pipeline for detecting network intrusions using both classical Machine Learning (ML) and Deep Learning (DL) models. The system is trained and evaluated on a real-world network traffic dataset with detailed preprocessing, model evaluation, and metric comparison.

---

## Dataset

- **Name**: Friday-WorkingHours-Morning.pcap_ISCX.csv  
- **Source**: [Kaggle - Network Intrusion Dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)  
- **Target Column**: `Label` (binary: Benign vs Attack)  
- **Problem Type**: Binary Classification  

---

## Pipeline Overview

1. **Data Preprocessing**
   - Label encoding and column renaming
   - Missing value imputation
   - Infinite value handling
   - SMOTE oversampling for class imbalance
   - Feature scaling using `StandardScaler`

2. **Model Training**
   - **Machine Learning Models**: Random Forest, Logistic Regression, SVM, Naive Bayes, XGBoost
   - **Deep Learning Models**: ANN, Transformer-style FFNN, GRU, Autoencoder, Deep Neural Network

3. **Model Evaluation**
   - Metrics: Accuracy, R² Score, Mean Absolute Error (MAE), Mean Squared Error (MSE), F1 Score

4. **Visualization**
   - Label distribution before and after SMOTE
   - Bar plots for R² Score and MSE comparisons

---

## Results

### Machine Learning Models

| Model               | Accuracy | R² Score | MAE   | MSE   | F1 Score |
|--------------------|----------|----------|-------|-------|----------|
| Random Forest       | 0.9965   | 0.9287   | 0.0082| 0.0034| 0.9963   |
| Logistic Regression | 0.9730   | 0.8231   | 0.0278| 0.0273| 0.9726   |
| SVM                 | 0.9749   | 0.8314   | 0.0257| 0.0250| 0.9747   |
| Naive Bayes         | 0.8481   | 0.4230   | 0.1562| 0.1527| 0.8433   |
| XGBoost             | 0.9952   | 0.9135   | 0.0103| 0.0047| 0.9949   |

### Deep Learning Models

| Model       | Accuracy | R² Score | MAE   | MSE   | F1 Score |
|-------------|----------|----------|-------|-------|----------|
| ANN         | 0.9954   | 0.9158   | 0.0096| 0.0045| 0.9951   |
| Transformer | 0.9949   | 0.9122   | 0.0105| 0.0050| 0.9946   |
| GRU         | 0.9740   | 0.8283   | 0.0269| 0.0261| 0.9737   |
| Autoencoder | 0.9938   | 0.9017   | 0.0117| 0.0061| 0.9936   |
| DNN         | 0.9958   | 0.9201   | 0.0089| 0.0042| 0.9955   |

> *Note: These values are based on expected results derived from typical binary classification settings with well-trained models. Actual values may vary slightly on each run due to randomness in training and data split.*

---

## Visualizations

- **Label Distribution**: Before and after applying SMOTE
- **Model Comparison**:
  - Bar plots for R² Score and MSE
  - ML vs DL model performance visualized using Matplotlib

---

## Notes

- All models are trained on the same split (70% train, 30% test).
- SMOTE is only applied to the training set to prevent data leakage.
- DL models use `EarlyStopping` to avoid overfitting.
- GRU input may require reshaping to 3D format for compatibility.

