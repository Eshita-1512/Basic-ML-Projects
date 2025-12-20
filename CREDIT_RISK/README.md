# Credit Risk Prediction using Machine Learning

## 📌 Project Overview

This project focuses on **credit risk prediction**, where the goal is to predict whether a loan applicant is likely to **default or not** (`loan_status`).

I implemented and compared **three machine learning models**:

* **Logistic Regression** (baseline, interpretable model)
* **Random Forest Classifier** (bagging-based ensemble)
* **XGBoost Classifier** (gradient boosting, final best model)

The project emphasizes **proper evaluation, generalization, and metric-driven decision making**, rather than blindly increasing model complexity.

---

## 🎯 Problem Statement

Given historical loan applicant data (income, credit score, debt ratio, loan details, etc.), predict the probability of loan default.

This is a **binary classification problem** commonly encountered in:

* Banking
* Lending platforms
* Risk analytics

---

## 🗂️ Dataset

The dataset contains LendingClub-style loan information, including:
https://www.kaggle.com/datasets/wordsforthewise/lending-club

### Numerical Features

* `loan_amnt`
* `int_rate`
* `installment`
* `fico_range_low`
* `annual_inc`
* `dti`
* `revol_util`
* `open_acc`, `total_acc`
* `delinq_2yrs`, `inq_last_6mths`

### Categorical Features

* `term`
* `grade`
* `sub_grade`
* `emp_length`
* `purpose`

### Target Variable

* `loan_status` (0 = Non-default, 1 = Default)

The dataset is **well-balanced (~52% / 48%)**, making ROC-AUC a suitable evaluation metric.

---

## 🛠️ Preprocessing Pipeline

A robust preprocessing pipeline was built using **scikit-learn**:

* **Missing value handling** using `SimpleImputer`
* **Categorical encoding** using `OneHotEncoder`
* **Column-wise transformations** using `ColumnTransformer`

> Note: Feature scaling was intentionally avoided for tree-based models (Random Forest & XGBoost), as it does not impact split-based learning.

---

## 🤖 Models Implemented

### 1️⃣ Logistic Regression

* Used as a **baseline model**
* Provides interpretability and serves as a benchmark
* Helps identify whether non-linear models truly add value

### 2️⃣ Random Forest Classifier

* Ensemble of decision trees trained using **bootstrap sampling**
* Reduces variance compared to a single decision tree
* Captures non-linear relationships

### 3️⃣ XGBoost Classifier (Final Model)

* Gradient Boosted Decision Trees
* Strong generalization performance
* Handles non-linearities and feature interactions efficiently
* Regularization used to avoid overfitting

---

## 📊 Evaluation Strategy

The models were evaluated using **unseen test data**.

### Metrics Used

* **ROC-AUC (Primary Metric)** → measures ranking quality
* Accuracy (secondary)
* Classification Report (Precision, Recall, F1-score)

### Key Checks

* Train vs Test performance comparison
* Baseline accuracy comparison
* Overfitting / underfitting analysis

---

## 📈 Results Summary

| Model               | ROC-AUC  | Notes                                        |
| ------------------- | -------- | -------------------------------------------- |
| Logistic Regression | Baseline | Interpretable, lower performance             |
| Random Forest       | Improved | Captures non-linearity                       |
| XGBoost             | **Best** | Strong generalization, stable train-test gap |

> Additional manual feature engineering was tested but **reverted after observing a drop in ROC-AUC**, prioritizing empirical validation over assumptions.

---

## 🧠 Key Learnings

* Strong models like **XGBoost often outperform heavy manual feature engineering** on well-designed tabular data
* **ROC-AUC is more reliable than accuracy** for credit risk problems
* Feature removal and hyperparameter tuning can be more effective than feature addition
* Metric-driven decisions are critical in real-world ML

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook
```

Each model is implemented in a separate notebook for clarity.

---

## 📦 Tech Stack

* Python
* NumPy, Pandas
* scikit-learn
* XGBoost
* Jupyter Notebook

---

## 📌 Future Improvements

* Hyperparameter optimization using GridSearch / Optuna
* SHAP-based feature importance analysis
* Threshold tuning based on business cost
* Model explainability for regulatory use cases

---

## 👤 Author

**Eshita**
Machine Learning Enthusiast | Credit Risk Modeling

---

⭐ If you find this project useful, feel free to star the repository!

