# IEEE ML Challenge – Fault Detection Using XGBoost

---

## Executive Summary

This project addresses a binary classification problem from the IEEE ML Challenge. The objective is to classify system instances as either **Normal (0)** or **Faulty (1)** using 47 numerical input features (F01–F47).

The dataset contains 43,776 training samples with a slight class imbalance (approximately 60% Normal and 40% Faulty).

To solve this problem, a structured machine learning pipeline was developed using **XGBoost**, combined with **Stratified K-Fold Cross-Validation** and **global threshold optimization** to maximize performance.  

Special care was taken to:
- Handle class imbalance properly  
- Prevent data leakage  
- Ensure reliable cross-validation  
- Optimize the classification threshold  
- Generate a correctly formatted submission file  

This approach ensures strong generalization performance and aligns with machine learning competition best practices.

---

# Detailed Project Documentation

---

## 1. Problem Statement

This is a supervised binary classification problem.

**Input:**
- 47 numerical features (F01–F47)

**Target Variable:**
- `Class`
  - 0 → Normal
  - 1 → Faulty

**Dataset Characteristics:**
- 43,776 training samples  
- Slight class imbalance (~60:40)  

The objective is to build a model that accurately predicts whether a system instance is faulty.

---

## 2. Dataset Understanding

### 2.1 Feature Characteristics

- All features are numerical.
- No categorical encoding was required.
- No missing values were present.
- Feature scaling was not necessary because tree-based models were used.

### 2.2 Target Distribution

The dataset has mild class imbalance:
- Majority class: Normal  
- Minority class: Faulty  

Although the imbalance is not extreme, it can bias the model toward predicting the majority class. Therefore, it was addressed during model training.

---

## 3. Modeling Approach

### Why XGBoost?

XGBoost was selected because:

- It performs exceptionally well on structured/tabular datasets.
- It captures complex non-linear feature interactions.
- It includes built-in regularization mechanisms.
- It handles class imbalance effectively.
- It is widely used in machine learning competitions.

Compared to Random Forest, XGBoost provides:
- Better generalization through boosting  
- Stronger regularization  
- Higher predictive performance on complex patterns  

---

## 4. Training Strategy

### Stratified K-Fold Cross-Validation

Instead of using a single train-validation split, the dataset was divided using:

StratifiedKFold (5 folds)

This approach ensures:
- Each fold maintains the original 60–40 class ratio.
- Reduced variance in evaluation.
- More reliable performance estimation.
- Prevention of overfitting to a single split.

---

## 5. Handling Class Imbalance

XGBoost does not use `class_weight` like some scikit-learn models.

Instead, imbalance was handled using:

scale_pos_weight = (number of negative samples) / (number of positive samples)

This technique:
- Increases the penalty for misclassifying minority class samples.
- Adjusts gradient updates during boosting.
- Reduces bias toward the majority class.

This improved recall and F1-score while maintaining stable probability outputs.

---

## 6. Hyperparameter Optimization

A structured grid search was performed across:

- n_estimators  
- learning_rate  
- max_depth  
- subsample  
- colsample_bytree  

Each configuration was evaluated using:
- Full cross-validation  
- Global threshold optimization  
- F1-score comparison  

### Best Hyperparameters Found

n_estimators = 400  
learning_rate = 0.05  
max_depth = 7  
subsample = 0.8  
colsample_bytree = 0.8  

These parameters produced the highest cross-validated F1-score.

---

## 7. Global Threshold Optimization

XGBoost outputs probabilities by default.

Although the standard classification threshold is 0.5, this is not always optimal in imbalanced scenarios.

To improve classification performance:

1. Out-of-fold validation probabilities were collected.  
2. Thresholds from 0.1 to 0.9 were evaluated.  
3. F1-score was computed for each threshold.  
4. The threshold with the highest F1-score was selected.  

### Best Threshold Found

0.4

This threshold improved the balance between precision and recall.

---

## 8. Evaluation Metrics

Two primary metrics were used:

### F1-Score
- Balances precision and recall.
- Important for imbalanced classification.

### ROC-AUC
- Measures ranking capability of the model.
- Independent of classification threshold.
- Reflects class separability.

Tracking both metrics ensured strong ranking performance and optimal classification boundary selection.

---

## 9. Final Model Training

After selecting:
- Best hyperparameters  
- Best threshold  

The model was retrained on the entire training dataset to maximize learning before generating test predictions.

---

## 10. Submission File Generation

The test dataset contained an `ID` column not present in training.

To avoid feature mismatch:

1. The `ID` column was separated.  
2. Predictions were generated using only feature columns.  
3. The final submission file was formatted as required:

ID,Class  
1,1  
2,0  
3,0  
4,1  

The row order was preserved exactly as in TEST.csv.

---

## 11. Key Design Decisions

| Decision | Reason |
|----------|--------|
| XGBoost | High performance on structured data |
| Stratified K-Fold | Reliable cross-validation |
| scale_pos_weight | Proper imbalance handling |
| Grid Search | Optimized model parameters |
| Global Threshold Tuning | Improved F1-score |
| Full Retraining | Maximum data utilization |
| Separate ID Handling | Prevent feature mismatch |

---

## Setup Instructions


1. Install required libraries:

pip install pandas numpy scikit-learn xgboost

---

## Usage Instructions

1. Place TRAIN.csv and TEST.csv in the root directory.
2. Run the notebook or training script.
3. The final prediction file FINAL.csv will be generated.


---

## 12. Conclusion

This project demonstrates a structured and competition-oriented approach to binary classification on tabular data.

The solution emphasizes:

- Proper cross-validation  
- Imbalance handling  
- Hyperparameter optimization  
- Threshold calibration  
- Clean submission formatting  

Each modeling decision was validated through cross-validation rather than relying on default settings.

The final pipeline ensures robustness, reproducibility, and strong generalization performance, aligning with best practices in applied machine learning.
