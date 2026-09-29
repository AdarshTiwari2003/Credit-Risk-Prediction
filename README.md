# 💳 Credit Risk Prediction — Industry-Level Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-green)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple)
![Status](https://img.shields.io/badge/Project-Completed-success)

## 📌 Project Overview

**Credit Risk Prediction** is an end-to-end Machine Learning project designed to predict whether a customer is likely to **default on a loan**.

The project simulates a real-world banking/financial-services use case where historical customer, credit, financial, and loan information is analyzed to identify potential credit risk.

The solution includes the complete ML workflow:

**Data Collection → Data Cleaning → EDA → Feature Engineering → Preprocessing → Model Training → Hyperparameter Tuning → Evaluation → Explainability → Model Deployment Preparation**

> ⚠️ **Important:** The dataset used in this project is synthetic and created for educational and portfolio purposes. It should not be used for real-world lending or credit decisions.

---

# 🎯 Business Problem

Financial institutions need to evaluate the risk associated with lending money to customers.

Incorrectly approving a high-risk borrower can result in financial losses, while incorrectly rejecting a low-risk borrower can result in lost business opportunities.

### Business Objective

Build a machine learning classification system that estimates the probability of **loan default** based on customer and loan characteristics.

The model can help organizations:

* Identify potentially high-risk applications
* Prioritize applications for manual review
* Analyze customer credit risk
* Support risk-management workflows
* Understand important factors associated with default
* Optimize decision thresholds according to business requirements

---

# 🧠 Machine Learning Problem

### Problem Type

**Binary Classification**

### Target Variable

`loan_default`

| Value | Meaning                   |
| ----- | ------------------------- |
| `0`   | Customer does not default |
| `1`   | Customer defaults         |

---

# 📊 Dataset

The project contains approximately **12,000 customer loan records**.

### Main Features

| Feature              | Description                             |
| -------------------- | --------------------------------------- |
| `customer_id`        | Unique customer identifier              |
| `age`                | Customer age                            |
| `annual_income`      | Annual customer income                  |
| `employment_years`   | Years of employment                     |
| `employment_type`    | Type of employment                      |
| `home_ownership`     | Home ownership status                   |
| `dependents`         | Number of dependents                    |
| `credit_score`       | Customer credit score                   |
| `num_credit_lines`   | Number of credit lines                  |
| `credit_utilization` | Credit utilization ratio                |
| `existing_debt`      | Existing customer debt                  |
| `loan_amount`        | Requested loan amount                   |
| `loan_term_months`   | Loan duration                           |
| `interest_rate`      | Loan interest rate                      |
| `loan_purpose`       | Purpose of the loan                     |
| `delinquencies`      | Number of previous delinquencies        |
| `late_payment_12m`   | Late payments during previous 12 months |
| `bankruptcy_history` | Previous bankruptcy indicator           |
| `loan_default`       | Target variable                         |

---

# 🔍 Exploratory Data Analysis

The project performs multiple EDA activities, including:

* Dataset shape and structure
* Data types
* Missing-value analysis
* Duplicate-value analysis
* Statistical summary
* Target distribution
* Credit-score analysis
* Loan amount analysis
* Default rate by loan purpose
* Correlation analysis

### Example Questions

The analysis investigates questions such as:

* Does credit score relate to default risk?
* How does loan amount vary across default groups?
* Which loan purposes have different default rates?
* Does credit utilization provide a useful risk signal?
* Which financial variables are strongly related to one another?

---

# 🛠️ Feature Engineering

Additional financial features are created to improve the model.

### Debt-to-Income Ratio

```text
Debt-to-Income = Existing Debt / Annual Income
```

### Loan-to-Income Ratio

```text
Loan-to-Income = Loan Amount / Annual Income
```

### Payment Pressure

```text
Payment Pressure = Loan Amount / Loan Term
```

These ratios provide additional context beyond individual raw variables.

---

# 🧹 Data Preprocessing

The ML pipeline handles preprocessing automatically.

### Numerical Features

* Missing-value imputation using median
* Standardization using `StandardScaler`

### Categorical Features

* Missing-value imputation using most frequent value
* One-hot encoding
* Unknown-category handling

The project uses Scikit-learn's `Pipeline` and `ColumnTransformer` to prevent inconsistent preprocessing between training and testing data.

---

# 🤖 Machine Learning Models

Three major classification approaches are implemented.

### 1. Logistic Regression

Used as the baseline classification model.

Advantages:

* Simple
* Interpretable
* Fast
* Useful as a benchmark

### 2. Random Forest

An ensemble of decision trees designed to capture nonlinear relationships.

Advantages:

* Handles nonlinear patterns
* Works with mixed feature relationships
* Provides useful feature-importance analysis
* Robust against individual-tree overfitting

### 3. Gradient Boosting

A boosting-based ensemble model that builds trees sequentially to improve prediction performance.

---

# ⚙️ Hyperparameter Tuning

Random Forest is optimized using:

**RandomizedSearchCV + Stratified 5-Fold Cross-Validation**

Parameters explored include:

* Number of estimators
* Maximum tree depth
* Minimum samples required for splitting
* Minimum samples per leaf
* Maximum features

The optimization objective is:

```text
ROC-AUC
```

---

# 📈 Model Evaluation

Multiple metrics are used instead of relying only on accuracy.

### Accuracy

Measures the overall proportion of correct predictions.

### Precision

Measures how many predicted defaults were actually defaults.

### Recall

Measures how many actual defaults were detected.

### F1-Score

Balances precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between default and non-default customers across classification thresholds.

---

# 📊 Evaluation Visualizations

The notebook includes:

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Threshold Analysis
* Feature Importance

---

# 🎚️ Decision Threshold Analysis

The project does not assume that `0.50` is always the best classification threshold.

Different thresholds are evaluated to understand the trade-off between:

* Precision
* Recall
* F1-score

For example, a financial institution may choose a threshold based on the relative business cost of:

* Missing a potential default
* Flagging a customer who would not default

This demonstrates a more realistic approach than simply maximizing accuracy.

---

# 🔎 Model Explainability

Permutation Importance is used to identify features that have the greatest effect on model performance.

This helps answer:

> **Which customer and financial characteristics provide the most useful predictive information?**

Potentially important variables include:

* Credit score
* Credit utilization
* Debt-to-income ratio
* Loan-to-income ratio
* Delinquencies
* Previous late payments
* Loan amount

The exact importance should be interpreted from the trained model rather than assumed in advance.

---

# 💾 Model Persistence

The final trained pipeline can be saved using Joblib:

```python
joblib.dump(best_model, "credit_risk_model.joblib")
```

This allows the complete preprocessing + prediction pipeline to be reused later.

---

# 📁 Project Structure

```text
Credit-Risk-Prediction/
│
├── credit_risk_prediction_industry_level.ipynb
│
├── credit_risk_industry_dataset.csv
│
├── credit_risk_model.joblib
│
└── README.md
```

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

## 2. Navigate to the Project

```bash
cd Credit-Risk-Prediction
```

## 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

## 5. Open the Notebook

Open:

```text
credit_risk_prediction_industry_level.ipynb
```

Run the cells from top to bottom.

---

# 💼 Business Impact

A production-grade credit-risk solution could help a financial organization:

* Prioritize applications for additional review
* Identify potentially risky applications
* Improve portfolio monitoring
* Support credit-risk analytics
* Analyze drivers associated with loan default
* Create risk segments
* Adjust decision thresholds according to business costs

However, a real deployment would require additional validation, governance, fairness testing, regulatory review, monitoring, and out-of-time testing.

---

# 🔐 Responsible AI Considerations

Credit decisions can have significant effects on individuals.

Before using a similar model in production, organizations should consider:

* Fairness and bias testing
* Explainability requirements
* Data privacy
* Regulatory requirements
* Model monitoring
* Data drift
* Out-of-time validation
* Human review
* Appropriate use of sensitive/protected attributes

This portfolio project does **not** represent a production lending decision system.

---

# 📚 Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Logistic Regression
* Random Forest
* Gradient Boosting
* RandomizedSearchCV

### Model Management

* Joblib

### Development Environment

* Jupyter Notebook
* VS Code
* GitHub

---

# 📌 Resume Project Description

### Credit Risk Prediction | Machine Learning & Financial Risk Analytics

* Built an end-to-end machine learning pipeline to predict loan default risk using customer credit, financial, employment, and loan attributes.
* Performed data quality analysis, EDA, missing-value treatment, categorical encoding, scaling, and financial-ratio feature engineering.
* Compared Logistic Regression, Random Forest, and Gradient Boosting models using Precision, Recall, F1-score, and ROC-AUC.
* Performed RandomizedSearchCV with stratified cross-validation and analyzed classification-threshold trade-offs for business-oriented risk decisions.
* Implemented permutation-based feature importance and serialized the final preprocessing-model pipeline using Joblib.

---

# ⭐ Key Skills Demonstrated

```text
Python
Pandas
NumPy
EDA
Data Cleaning
Feature Engineering
Classification
Logistic Regression
Random Forest
Gradient Boosting
Hyperparameter Tuning
Cross Validation
ROC-AUC
Precision
Recall
F1-Score
Confusion Matrix
Model Explainability
Scikit-learn Pipelines
Joblib
Business Analytics
Credit Risk Analytics
```

---

# 👨‍💻 Author

**Adarsh Tiwari**

Aspiring AI/ML Engineer | Data Analyst

Skills: Python • SQL • Machine Learning • Data Analysis • Power BI • Tableau

---

⭐ **If you find this project useful, consider giving the repository a star!**
