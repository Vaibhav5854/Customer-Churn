## Customer Churn Prediction & Risk Analytics

An end-to-end machine learning system for predicting customer churn, comparing traditional machine learning models with a deep learning ANN, and converting churn probabilities into actionable customer risk segments.

## Project Overview

Customer churn is a major business problem in the banking sector. The objective of this project is to identify customers who are likely to leave the bank and provide a data-driven risk assessment that can support customer retention strategies.

The project covers the complete ML lifecycle:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Model Training
- Hyperparameter Tuning
- Cross-Validation
- ML vs Deep Learning Model Comparison
- Classification Threshold Optimization
- Risk Segmentation
- Model Interpretability
- Interactive Prediction
- Cloud Deployment

## Dataset

The project uses the Churn Modelling dataset containing customer demographic, financial and banking-related attributes.

Important features include:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Membership Status
- Estimated Salary

Target variable:

- `Exited` — 1 indicates customer churn and 0 indicates customer retention.

## Feature Engineering

Six additional features were created to capture meaningful relationships between customer attributes:

| Feature | Description |
|---|---|
| `BalanceSalaryRatio` | Customer balance relative to estimated salary |
| `TenureByAge` | Tenure relative to customer age |
| `HasZeroBalance` | Indicates whether the customer has zero balance |
| `BalancePerProduct` | Balance relative to number of products |
| `SalaryPerProduct` | Salary relative to number of products |
| `IsSenior` | Indicates customers aged 50 or above |

## Models

The project compares four traditional machine learning algorithms with a deep learning model.

### Traditional Machine Learning

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

### Deep Learning

- Keras Artificial Neural Network (ANN)

The ANN architecture consists of:

Input Layer
↓
Dense Layer (64 neurons, ReLU)
↓
Dropout (30%)
↓
Dense Layer (32 neurons, ReLU)
↓
Dropout (20%)
↓
Output Layer (1 neuron, Sigmoid)

Early stopping was used during ANN training to reduce overfitting.

## Model Optimization

Traditional ML models were optimized using:

- RandomizedSearchCV
- 5-fold Stratified Cross-Validation
- ROC-AUC as the hyperparameter optimization metric

The classification threshold was further optimized using out-of-fold predictions and F1-score rather than relying only on the default 0.5 threshold.

## Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The project also includes:

- ROC-AUC comparison
- F1-score comparison
- Precision and Recall comparison
- Confusion matrices
- Feature importance analysis

## Risk Segmentation

Predicted churn probabilities are converted into three customer risk categories:

| Churn Probability | Risk Segment | Business Action |
|---|---|---|
| < 30% | Low | Routine monitoring |
| 30–60% | Medium | Monitor and engage |
| ≥ 60% | High | Immediate retention action |

## Model Interpretability

Permutation feature importance is used to identify features that have the greatest impact on model performance.

The project currently provides feature importance analysis for the XGBoost model.

## Streamlit Application

The trained models are integrated into an interactive Streamlit application.

### Predict Churn

- Customer information input
- Churn probability
- Model-wise predictions
- Risk segment
- Recommended business action

### EDA & Insights

- Overall churn distribution
- Churn by geography
- Churn by age group
- Churn by active membership
- Churn by number of products
- Churn by gender
- Balance distribution
- Correlation matrix

### Model Analysis

- Model performance comparison
- ROC-AUC comparison
- F1-score comparison
- Precision comparison
- Recall comparison
- XGBoost feature importance
- Risk segmentation analysis

## Project Structure

Customer-Churn/
├── app.py
├── Churn_Modelling.csv
├── experiments.ipynb
├── prediction1.ipynb
├── ml_models.joblib
├── keras_ann.h5
├── ann_preprocessor.joblib
├── model_thresholds.joblib
├── optimized_model_comparison.csv
├── xgboost_feature_importance.csv
├── risk_segmentation.csv
├── requirements.txt
└── README.md

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## Deployment

The application is deployed using Streamlit Community Cloud.

The deployed application provides an end-to-end interface for customer churn prediction, exploratory analysis, model comparison and customer risk analytics.

## Key Outcome

The project demonstrates an end-to-end approach to customer churn analytics by combining predictive modeling, model evaluation, threshold optimization, customer risk segmentation, interpretability and deployment into a single application.
