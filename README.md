# Credit Risk Modeling – Probability of Default (PD)

This project demonstrates the end-to-end process of modeling **Probability of Default (PD)** — a fundamental component of credit risk assessment used by financial institutions to evaluate loan applications and optimize portfolio risk.

We explore both interpretable and advanced methods:
- **Logistic Regression** (baseline, transparent model),
- **Gradient Boosted Trees** (XGBoost, for nonlinear patterns and better recall).

## 📊 Problem Statement

The goal is to train and compare models that can predict whether a loan applicant will default. Special focus is given to:
- handling imbalanced datasets,
- evaluating model performance beyond accuracy,
- optimizing decision thresholds to balance business risk and profit.

## 📁 Dataset

Synthetic dataset simulating loan applications. Each record contains:
- **Demographic info** (e.g., age, income, credit history),
- **Loan attributes** (amount, intent, grade, interest rate),
- **Target**: whether the loan defaulted (binary).

### Selected Features

| Feature | Description |
|--------|-------------|
| `person_age` | Age of the applicant |
| `person_income` | Annual income |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Credit grade assigned to the loan |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Interest rate on the loan |
| `loan_status` | Target: 1 if defaulted, 0 otherwise |
| `cb_person_default_on_file` | Has the person defaulted before |
| `cb_person_cred_hist_length` | Length of credit history |

## 🔧 Methods Used

### Data Cleaning
- Removal of duplicates and outliers using IQR.
- Label encoding for ordinal features.
- One-hot encoding for nominal features.
- Handled missing values via outlier filtering.

### Baseline Model: Logistic Regression
- Verified statistical assumptions (multicollinearity, linear log-odds, large sample).
- Assessed performance via metrics such as:
  - Accuracy
  - Precision, Recall, F1-Score
  - AUC-ROC
- Experimented with **decision threshold tuning** for cost-sensitive classification.

### Advanced Model: XGBoost
- Captured non-linear relationships.
- Significantly improved **recall for defaulted loans**.
- Achieved higher **AUC** and lower **Expected Loss**.

### Business Strategy Simulation
- Simulated threshold-based loan acceptance policies.
- Calculated financial impact of various decision thresholds.
- Identified the **optimal threshold** (e.g., 0.88) maximizing expected profit.

## 📈 Results Summary

| Metric                | Logistic Regression | XGBoost |
|----------------------|---------------------|---------|
| Accuracy             | 80%                 | 93%     |
| Default Recall       | 16%                 | 72%     |
| AUC                  | 0.75                | 0.94    |
| Expected Loss (EL)   | 8.41M               | 8.15M   |
| Savings vs Baseline  | —                   | **+258k** |

## 📌 Key Insights

- Logistic Regression is interpretable but struggles with defaults in imbalanced data.
- XGBoost dramatically improves performance with similar interpretability via SHAP/feature importance.
- Threshold selection is critical: optimizing for recall or profit leads to different strategies.

## 🧠 Technologies

- Python
- `pandas`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`
- `xgboost`
