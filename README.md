Financial Fraud Detection Using Machine Learning
Overview

This project builds a machine learning model to detect fraudulent financial transactions using a synthetic dataset. It demonstrates a complete end-to-end ML workflow, including:

Exploratory Data Analysis (EDA)

Feature engineering

Data preprocessing using ColumnTransformer

Model training with Pipeline

Hyperparameter tuning using RandomizedSearchCV

Evaluation with advanced metrics (precision, recall, F1, ROC–AUC)

Model explainability through feature importance

The final model achieves a ROC–AUC score of 1.0 on the synthetic dataset.

Dataset

The dataset contains 10,000 synthetic financial transactions with features such as:

amount

transaction_type

merchant_category

country

device_risk_score

ip_risk_score

hour

is_fraud (target)

The dataset is fully synthetic and contains no sensitive information.

Feature Engineering

Several domain-inspired features were added to improve model performance:

risk_total — combined device and IP risk

is_night — indicator for late-night transactions

log_amount — log-transformed transaction amount

merchant_risk — fraud rate per merchant category

country_risk — fraud rate per country

amount_risk_interaction — interaction between amount and risk score

These features better reflect real-world fraud analysis patterns.

Modeling Approach

A Random Forest Classifier is trained using a Scikit-learn Pipeline, which includes:

One-hot encoding for categorical features

Passthrough for numeric and engineered features

class_weight="balanced" to handle class imbalance

Hyperparameter tuning is performed using:

RandomizedSearchCV
scoring = "roc_auc"
cv = 3
n_iter = 20
n_jobs = -1

Evaluation

Metrics used:

Accuracy

Precision

Recall

F1-score

ROC–AUC

Confusion matrix

Classification report

Model performance on the test set (synthetic data):

Metric	Score
Accuracy	1.0
Precision	1.0
Recall	1.0
F1-score	1.0
ROC–AUC	1.0

These perfect results are expected because the dataset is artificially separable.

Model Explainability

Feature importance scores were extracted using the trained Random Forest model and the transformed feature names from the preprocessing pipeline.

Key insights:

Device and IP risk scores are the strongest predictors

Transaction amount and hour contribute meaningfully

Merchant and country risk features add additional signal

Key Takeaways

Risk-based features are the strongest fraud indicators

Feature engineering significantly improves interpretability

Pipelines help keep preprocessing and modeling consistent

RandomizedSearchCV efficiently identifies strong hyperparameters

Synthetic data explains the perfect model performance

Future Improvements

Add rolling user-level features (transaction velocity, history)

Experiment with boosting models (XGBoost, LightGBM, CatBoost)

Add SHAP-based model explainability

Deploy the model using FastAPI or Streamlit

Implement adjustable fraud alert thresholds