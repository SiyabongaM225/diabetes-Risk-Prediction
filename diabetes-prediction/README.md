# 🩺 Diabetes Risk Prediction

An end-to-end machine learning project that uses Logistic Regression to estimate diabetes risk from patient health measurements.

The project covers the complete machine learning workflow, including exploratory data analysis, data quality assessment, preprocessing, model development, cross-validation, hyperparameter tuning, classification threshold analysis, model evaluation and deployment through a Streamlit web application.

> ⚠️ **Disclaimer:** This project is intended for educational and portfolio purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice.

---

## 📌 Project Overview

The objective of this project was to build a binary classification model capable of estimating whether a patient is likely to have diabetes based on several health measurements.

The project was designed to develop practical skills in:

- Exploratory Data Analysis (EDA)
- Data cleaning and quality assessment
- Handling missing and invalid values
- Feature analysis
- Logistic Regression
- Feature scaling
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- Classification threshold optimization
- Model serialization
- Streamlit deployment

---

## 🎯 Objectives

The main objectives of the project were to:

1. Explore the dataset and understand its structure.
2. Identify potential data quality issues.
3. Investigate missing and invalid values.
4. Analyze relationships between the features and diabetes outcome.
5. Build a Logistic Regression classification model.
6. Evaluate the model using:
   - Accuracy
   - Precision
   - Recall
   - ROC-AUC
7. Use cross-validation to assess model stability.
8. Tune the Logistic Regression hyperparameters.
9. Evaluate different classification thresholds.
10. Save the trained model.
11. Deploy the model through a simple Streamlit web application.

---

# 📊 Dataset

The project uses the **Pima Indians Diabetes Dataset**.

The dataset contains:

- **768 observations**
- **8 predictor variables**
- **1 binary target variable**

### Features

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | Two-hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Patient age |
| Outcome | Diabetes outcome |

### Target Variable

`Outcome` is the target variable:

- `0` — No diabetes
- `1` — Diabetes

The dataset contains:

- 500 observations with outcome `0`
- 268 observations with outcome `1`

This represents a moderately imbalanced target distribution.

---

# 🔎 Exploratory Data Analysis

The exploratory analysis examined:

- Dataset dimensions
- Data types
- Descriptive statistics
- Target distribution
- Missing values
- Duplicate records
- Zero values
- Feature correlations

## Data Quality Findings

### Missing values

No explicit missing values were identified:

```text
Total missing values: 0