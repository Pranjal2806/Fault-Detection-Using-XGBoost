IEEE ML Challenge – Fault Detection Using XGBoost
Executive Summary

This project addresses a binary classification problem from the IEEE ML Challenge, where the objective is to classify system instances as either Normal (0) or Faulty (1) based on 47 numerical features (F01–F47). The dataset contains 43,776 training samples with a slight class imbalance (approximately 60% normal and 40% faulty).

To solve this problem, a robust machine learning pipeline was developed using XGBoost, combined with Stratified K-Fold Cross-Validation and global threshold optimization to maximize F1-score performance. Special attention was given to handling class imbalance, preventing data leakage, and ensuring reliable validation through out-of-fold predictions.

The final solution includes:

Cross-validated model training

Imbalance handling using scale_pos_weight

Global threshold tuning

Final training on the full dataset

Properly formatted prediction file generation for submission

This approach ensures high generalization performance and aligns with competition evaluation standards.

Detailed Project Documentation
1. Problem Statement

The task is a supervised binary classification problem:

Input: 47 numerical features (F01–F47)

Target variable: Class

0 → Normal

1 → Faulty

Training samples: 43,776

Slight class imbalance (~60:40)

The goal is to build a model that accurately predicts whether a given system instance is faulty.

2. Understanding the Dataset
2.1 Feature Characteristics

All features are numerical.

No categorical encoding was required.

No missing values were observed.

Feature scaling was not necessary due to tree-based modeling.

2.2 Target Distribution

The dataset has mild imbalance:

Majority class: Normal

Minority class: Faulty

Although not extreme, this imbalance can bias the model toward predicting the majority class. Therefore, it was handled carefully.

3. Approach and Methodology
3.1 Why XGBoost?

XGBoost was selected because:

It performs exceptionally well on structured/tabular data.

It captures non-linear feature interactions.

It is robust to feature scaling.

It includes built-in mechanisms for handling imbalance.

It is widely used in machine learning competitions.

Compared to Random Forest, XGBoost offers:

Better regularization

Boosting-based error correction

Higher predictive power for complex datasets

4. Model Training Strategy
4.1 Stratified K-Fold Cross Validation

Instead of using a single train-validation split, the dataset was divided using:

StratifiedKFold (5 folds)

Why?

Maintains the original 60–40 class ratio in each fold.

Reduces variance in performance estimates.

Prevents overfitting to a single validation split.

Produces more reliable performance metrics.

This ensures the evaluation reflects real-world performance.

5. Handling Class Imbalance

XGBoost does not use class_weight like scikit-learn models.

Instead, imbalance was handled using:

scale_pos_weight = (number of negative samples) / (number of positive samples)

Why this works:

Increases penalty for misclassifying minority class.

Adjusts gradient updates during boosting.

Prevents bias toward majority class.

This improves recall and F1-score without distorting probability estimates.

6. Hyperparameter Optimization

A structured grid search was performed across:

n_estimators

learning_rate

max_depth

subsample

colsample_bytree

Each combination was evaluated using:

Full cross-validation

Global threshold optimization

F1-score comparison

Best parameters found:

n_estimators = 400
learning_rate = 0.05
max_depth = 7
subsample = 0.8
colsample_bytree = 0.8

These parameters provided the highest cross-validated F1-score.

7. Global Threshold Optimization

XGBoost outputs probabilities.

By default, classification threshold = 0.5.

However, 0.5 is not always optimal for imbalanced classification.

Therefore:

All out-of-fold validation probabilities were collected.

A range of thresholds (0.1 to 0.9) was tested.

F1-score was computed for each threshold.

The threshold with maximum F1-score was selected.

Best threshold found:

0.4

This improved balance between precision and recall.

8. Evaluation Metrics

Two key metrics were tracked:

F1-Score

Measures balance between precision and recall.

Especially important in imbalance scenarios.

ROC-AUC

Measures ranking ability of model.

Independent of classification threshold.

Reflects overall separability between classes.

Using both metrics ensured:

Strong ranking performance

Optimal decision boundary

9. Final Model Training

After selecting:

Best hyperparameters

Best threshold

The model was retrained on the entire training dataset.

This allows the model to:

Learn from all available data

Maximize predictive power before test submission

10. Submission File Generation

The test dataset included an ID column not present in training.

To prevent feature mismatch:

ID column was separated.

Predictions were generated using feature-only data.

Final submission file was created in required format:

ID,Class
1,1
2,0
3,0
4,1

The order of rows was preserved exactly as in TEST.csv.

11. Key Design Decisions
Decision	Reason
XGBoost	Best for tabular structured data
Stratified K-Fold	Reliable validation
scale_pos_weight	Proper imbalance handling
Global threshold tuning	Optimized F1-score
Full retraining	Maximum data utilization
Separate ID handling	Prevent feature mismatch
12. Conclusion

This project demonstrates a structured and competition-grade approach to binary classification on tabular data.

The solution emphasizes:

Proper cross-validation

Imbalance management

Hyperparameter optimization

Threshold calibration

Clean submission formatting

Rather than relying on default parameters, each modeling decision was intentional and validated through cross-validation results.

The final pipeline ensures both robustness and reproducibility, aligning with best practices in applied machine learning.
