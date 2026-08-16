# 🩺 Diabetes Risk Prediction

An end-to-end machine learning project that uses **Logistic Regression** to estimate diabetes risk based on patient health measurements.

The project covers the complete machine learning workflow, from **exploratory data analysis and data quality assessment** through **model development, cross-validation, hyperparameter tuning, threshold optimization, evaluation, model serialization, and deployment using Streamlit**.

> ⚠️ **Disclaimer:** This project is intended for educational and portfolio purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## 🚀 Live Demo

🔗 [Try the Diabetes Risk Prediction App](https://diabetes-risk-predictiongit-n5mxbsja6dpvx7jngdzjaa.streamlit.app/)

## 📌 Project Overview

Diabetes is a major health condition that can be associated with several measurable health and demographic factors.

The objective of this project was to develop a binary classification model that estimates the likelihood of diabetes based on patient measurements from the **Pima Indians Diabetes Dataset**.

The project was also designed as a practical exercise in developing an end-to-end machine learning solution, rather than focusing only on model training.

The workflow includes:

1. Data loading
2. Exploratory Data Analysis
3. Data quality assessment
4. Missing-value analysis
5. Invalid-value analysis
6. Correlation analysis
7. Data preprocessing
8. Logistic Regression
9. Model evaluation
10. Cross-validation
11. Hyperparameter tuning
12. Classification threshold analysis
13. Final model evaluation
14. Model serialization
15. Streamlit deployment

---

# 🎯 Project Objectives

The main objectives of this project were to:

- Perform exploratory data analysis (EDA)
- Understand the structure and distribution of the dataset
- Identify missing and potentially invalid values
- Investigate relationships between features and diabetes outcome
- Prepare the data for machine learning
- Build a Logistic Regression classification model
- Evaluate the model using:
  - Accuracy
  - Precision
  - Recall
  - ROC-AUC
- Use cross-validation to evaluate model stability
- Tune the Logistic Regression hyperparameters
- Investigate different classification thresholds
- Select a threshold that improves recall
- Generate a confusion matrix
- Generate an ROC curve
- Save the trained model
- Build a simple web interface using Streamlit

---

# 📊 Dataset

This project uses the **Pima Indians Diabetes Dataset**.

The dataset contains **768 observations** and **9 columns**.

There are:

- 8 predictor variables
- 1 binary target variable

## Features

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times the patient has been pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skin fold thickness |
| `Insulin` | Two-hour serum insulin |
| `BMI` | Body Mass Index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Patient age |
| `Outcome` | Diabetes outcome |

## Target Variable

The `Outcome` column is the target variable:

| Value | Meaning |
|---:|---|
| `0` | No diabetes |
| `1` | Diabetes |

The dataset contains:

- **500 observations** with `Outcome = 0`
- **268 observations** with `Outcome = 1`

This means the target variable is moderately imbalanced, making metrics such as precision, recall and ROC-AUC important in addition to accuracy.

---

# 🛠️ Technologies Used

The project was developed using:

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Joblib**
- **Streamlit**
- **Jupyter Notebook**
- **KaggleHub**

---

# 🔍 Exploratory Data Analysis

The first stage of the project focused on understanding the dataset before building a machine learning model.

The analysis included:

- Dataset shape
- Column names
- Data types
- Descriptive statistics
- Target distribution
- Missing-value analysis
- Duplicate detection
- Zero-value analysis
- Correlation analysis

---

## Dataset Shape

The dataset contains:

```text
768 rows
9 columns
```

---

## Data Types

The dataset contains:

- 4 floating-point columns
- 5 integer columns

There were no unexpected data types identified.

---

# 📊 Visual Analysis

## Target Distribution

<img width="493" height="369" alt="image" src="https://github.com/user-attachments/assets/57e6d76a-2cef-43ce-a929-8138c2d8f2e6" />


The dataset contains 500 observations with no diabetes (`Outcome = 0`) and
268 observations with diabetes (`Outcome = 1`).

---

## 🔥 Feature Correlation

<img width="807" height="643" alt="image" src="https://github.com/user-attachments/assets/8db15ff1-f843-4447-97f6-e0e022111a93" />

The correlation analysis shows that `Glucose` has the strongest correlation
with the diabetes outcome, followed by `Insulin` and `BMI`.

---

# 📈 Model Evaluation

## Confusion Matrix

<img width="474" height="364" alt="image" src="https://github.com/user-attachments/assets/76f128eb-6232-4f2e-b1a0-fc3e19b1a5ed" />

The final model produced the following confusion matrix:

```text
[[73, 27],
 [13, 41]]
```

---
# 🧹 Data Quality Analysis

Before training the model, several data quality checks were performed.

## Missing Values

The dataset was checked for explicit missing values using Pandas.

Result:

```text
Total missing values: 0
```

Therefore, no explicit missing-value imputation was required.

---

## Duplicate Records

Duplicate rows were also checked.

Result:

```text
Duplicate rows: 0
```

No duplicate records were identified.

---

## Zero-Value Analysis

Zero values were examined because some medical measurements may use zero as a placeholder for an unavailable measurement.

The analysis showed:

```text
Pregnancies                 111
Glucose                       0
BloodPressure                 0
SkinThickness                 0
Insulin                       0
BMI                           0
DiabetesPedigreeFunction      0
Age                           0
Outcome                     500
```

Zero is a valid value for `Pregnancies` and `Outcome`.

The medical measurement features did not contain zero values requiring replacement in this dataset.

---

# 📈 Correlation Analysis

A correlation matrix was created to investigate relationships between the predictor variables and the diabetes outcome.

The correlations with `Outcome` were:

| Feature | Correlation with Outcome |
|---|---:|
| `Glucose` | 0.496 |
| `Insulin` | 0.377 |
| `BMI` | 0.316 |
| `SkinThickness` | 0.295 |
| `Age` | 0.238 |
| `Pregnancies` | 0.222 |
| `BloodPressure` | 0.174 |
| `DiabetesPedigreeFunction` | 0.174 |

`Glucose` had the strongest positive correlation with the diabetes outcome.

However, correlation does not imply causation. Therefore, correlation values were used as part of the exploratory analysis rather than as the sole basis for feature selection.

---

# 🤖 Machine Learning Approach

## Feature and Target Separation

The `Outcome` column was separated from the predictor variables.

```python
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
```

The resulting feature set contains:

```text
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
```

---

# 🔀 Train/Test Split

The dataset was divided into training and testing sets.

A **stratified train-test split** was used to preserve the distribution of the two target classes.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The test set was kept separate for final model evaluation.

---

# ⚙️ Data Preprocessing

The predictor variables were standardized using `StandardScaler`.

A Scikit-learn Pipeline was used to combine preprocessing and model training.

This approach ensures that the same preprocessing procedure is applied consistently during:

- Model training
- Cross-validation
- Testing
- Future predictions in the Streamlit application

---

# 🧠 Logistic Regression

Logistic Regression was selected because the project is a binary classification problem.

The model estimates the probability that a patient belongs to the positive diabetes class.

The general workflow was:

```text
Raw Features
     ↓
StandardScaler
     ↓
Logistic Regression
     ↓
Probability
     ↓
Classification Threshold
     ↓
Prediction
```

---

# 📏 Baseline Model Evaluation

The initial Logistic Regression model was evaluated using the test dataset.

The baseline results were:

| Metric | Score |
|---|---:|
| Accuracy | **70.78%** |
| Precision | **58.82%** |
| Recall | **55.56%** |
| ROC-AUC | **82.63%** |

Although the baseline accuracy was below the desired 75% target, the ROC-AUC of 82.63% indicated that the model had useful discriminatory ability.

This motivated further validation and tuning.

---

# 🔄 Five-Fold Cross-Validation

Five-fold cross-validation was performed to evaluate the stability of model performance across different subsets of the training data.

## Results

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy | **78.99%** | ±1.60% |
| Precision | **74.87%** | ±5.63% |
| Recall | **60.76%** | ±5.47% |
| ROC-AUC | **87.15%** | ±1.67% |

The cross-validation results were stronger than the initial single train-test evaluation.

The average accuracy was approximately **79%**, while the average ROC-AUC was approximately **87%**.

This provided stronger evidence that the model was learning useful patterns rather than relying on a single data split.

---

# ⚙️ Hyperparameter Tuning

Grid Search was used to tune the Logistic Regression regularization parameter `C`.

The best parameter was:

```text
C = 0.1
```

The best cross-validation ROC-AUC was:

```text
0.8731
```

The tuned Logistic Regression model was then evaluated on the previously unseen test set.

---

# 🎚️ Classification Threshold Analysis

Logistic Regression produces a probability between 0 and 1.

By default, a probability of 0.50 is normally used as the classification threshold:

```text
Probability >= 0.50 → Positive
Probability < 0.50 → Negative
```

However, changing the threshold changes the balance between:

- Accuracy
- Precision
- Recall

Because identifying positive cases was an important consideration in this project, multiple thresholds were evaluated.

## Threshold Results

<img width="271" height="173" alt="image" src="https://github.com/user-attachments/assets/edbc6d52-2817-4957-a439-ff8665691bd3" />

| Threshold | Accuracy | Precision | Recall |
|---:|---:|---:|---:|
| 0.20 | 70.52% | 54.47% | 93.93% |
| 0.25 | 75.08% | 59.38% | 90.19% |
| 0.30 | 78.18% | 64.39% | 83.64% |
| **0.35** | **79.97%** | **68.88%** | **77.57%** |
| 0.40 | 79.80% | 71.43% | 70.09% |
| 0.45 | 79.32% | 73.02% | 64.49% |
| 0.50 | 79.15% | 75.00% | 60.28% |
| 0.55 | 79.15% | 77.56% | 56.54% |
| 0.60 | 79.64% | 82.01% | 53.27% |
| 0.65 | 77.85% | 81.97% | 46.73% |
| 0.70 | 75.90% | 83.67% | 38.32% |

A threshold of **0.35** was selected because it provided a strong balance between accuracy and recall.

At this threshold:

- Accuracy = **79.97%**
- Precision = **68.88%**
- Recall = **77.57%**

The threshold selection was performed before the final evaluation on the held-out test set.

---

# 🏆 Final Model Evaluation

The final model was evaluated on the previously unseen test dataset using the selected threshold of **0.35**.

## Final Results

| Metric | Final Test Result |
|---|---:|
| **Accuracy** | **74.03%** |
| **Precision** | **60.29%** |
| **Recall** | **75.93%** |
| **ROC-AUC** | **82.67%** |

---

## Performance Interpretation

### Accuracy — 74.03%

The model correctly classified approximately 74% of the observations in the final test set.

This was slightly below the initial project target of 75%.

### Precision — 60.29%

When the model predicted a positive diabetes outcome, approximately 60% of those predictions were correct.

### Recall — 75.93%

The model identified approximately 76% of the actual positive diabetes cases.

Recall was an important consideration when selecting the 0.35 threshold.

### ROC-AUC — 82.67%

The ROC-AUC score of 0.827 indicates that the model has good ability to distinguish between the two outcome classes.

---

# 📊 Confusion Matrix

<img width="512" height="367" alt="image" src="https://github.com/user-attachments/assets/e5425440-db36-4e63-b0ce-9437a649369a" />

The final confusion matrix was:

```text
[[73, 27],
 [13, 41]]
```

This can be represented as:

| | Predicted No Diabetes | Predicted Diabetes |
|---|---:|---:|
| **Actual No Diabetes** | 73 | 27 |
| **Actual Diabetes** | 13 | 41 |

### Interpretation

The model correctly classified:

- **73** non-diabetic patients
- **41** diabetic patients

The model incorrectly classified:

- **27** non-diabetic patients as diabetic
- **13** diabetic patients as non-diabetic

The 13 false negatives are particularly relevant when considering recall.

---

# 📈 ROC Curve

The final model achieved:

```text
ROC-AUC = 0.8267
```

The ROC curve was generated using the predicted probabilities from the Logistic Regression model.

<img width="573" height="375" alt="image" src="https://github.com/user-attachments/assets/5f154756-8ef0-4564-ab12-0bf3f4e06f12" />

The ROC-AUC score indicates that the model performs substantially better than random classification.

---

# 💾 Model Serialization

The final trained model was saved using Joblib.

The saved model is located at:

```text
models/diabetes_logistic_regression.pkl
```

The selected classification threshold is stored separately in:

```text
models/threshold.txt
```

The saved model contains the preprocessing and Logistic Regression components required to make predictions on new data.

This allows the Streamlit application to load the trained model without retraining it.

---

# 🌐 Streamlit Application

A Streamlit web application was developed to provide a simple user interface for interacting with the trained model.

The application allows users to enter:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The application then:

1. Collects the user inputs.
2. Creates a Pandas DataFrame.
3. Passes the data through the saved machine learning pipeline.
4. Calculates the predicted probability.
5. Applies the 0.35 classification threshold.
6. Displays the estimated diabetes risk.
7. Displays the model's predicted risk category.

Lower-value Profile

<img width="1657" height="1252" alt="image" src="https://github.com/user-attachments/assets/34d63ed9-742b-495e-b1a2-951395a61e7c" />

Higher-value Profile

<img width="1669" height="1267" alt="image" src="https://github.com/user-attachments/assets/a75a6a13-a43a-4648-98d8-fe7e4150398e" />

---

# 🖥️ Application Workflow

```text
User Input
    │
    ▼
Patient Measurements
    │
    ▼
Pandas DataFrame
    │
    ▼
Saved Machine Learning Pipeline
    │
    ├── StandardScaler
    │
    └── Logistic Regression
            │
            ▼
      Prediction Probability
            │
            ▼
     Threshold = 0.35
            │
            ▼
     Risk Classification
            │
            ▼
       Streamlit UI
```

---

# 📁 Project Structure

```text
diabetes-prediction/
│
├── models/
│   ├── diabetes_logistic_regression.pkl
│   └── threshold.txt
│
├── notebooks/
│   └── diabetes_prediction.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation and Setup

## Prerequisites

Python 3.10+ is recommended.

---

## 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Navigate into the project directory:

```bash
cd diabetes-prediction
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Streamlit Application

From the project directory, run:

```bash
streamlit run app.py
```

Alternatively:

```bash
python -m streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 📓 Running the Jupyter Notebook

The complete model development process is contained in:

```text
notebooks/diabetes_prediction.ipynb
```

Start Jupyter:

```bash
jupyter notebook
```

Then open the notebook and run the cells from top to bottom.

The notebook contains:

- Data loading
- Dataset inspection
- EDA
- Data quality checks
- Missing-value analysis
- Duplicate analysis
- Correlation analysis
- Train/test splitting
- Logistic Regression
- Cross-validation
- Hyperparameter tuning
- Threshold analysis
- Final model evaluation
- Confusion matrix
- ROC curve
- Model saving

---

# 🧪 Example Prediction Inputs

The Streamlit application accepts the following inputs:

| Input | Example |
|---|---:|
| Pregnancies | 6 |
| Glucose | 150 |
| Blood Pressure | 80 |
| Skin Thickness | 35 |
| Insulin | 200 |
| BMI | 35 |
| Diabetes Pedigree Function | 0.60 |
| Age | 50 |

The model then returns an estimated probability and classification based on the selected threshold.

---

# ⚠️ Limitations

This project has several limitations that should be considered.

## Dataset Size

The dataset contains only 768 observations.

A larger dataset could provide more information for training and validation.

## Dataset Population

The model was trained on the Pima Indians Diabetes Dataset and may not generalize to different populations.

## External Validation

The model has not been evaluated against an independent external dataset.

## Model Selection

Only Logistic Regression was used as the primary classification algorithm.

Other algorithms may capture nonlinear relationships more effectively.

## Threshold Selection

The selected threshold of 0.35 improves recall but also changes the balance between precision and false positives.

## Medical Application

The model is not clinically validated and should not be used for medical diagnosis or treatment decisions.

---

# 🔮 Future Improvements

Potential future improvements include:

- Compare Logistic Regression against Random Forest
- Compare Logistic Regression against Gradient Boosting
- Experiment with Support Vector Machines
- Perform more extensive hyperparameter tuning
- Investigate feature engineering
- Perform probability calibration
- Evaluate the model using an external dataset
- Add automated unit tests
- Add input validation
- Add interactive visualizations
- Improve the Streamlit user experience
- Deploy the Streamlit application online
- Implement model monitoring
- Track model versions
- Add CI/CD for automated testing

---

# 🎓 Key Skills Demonstrated

This project demonstrates practical experience with:

### Data Analysis

- Exploratory Data Analysis
- Descriptive statistics
- Data quality assessment
- Correlation analysis
- Data visualization

### Data Preparation

- Missing-value analysis
- Duplicate detection
- Feature/target separation
- Train/test splitting
- Feature standardization

### Machine Learning

- Logistic Regression
- Binary classification
- Probability-based prediction
- Hyperparameter tuning
- Cross-validation

### Model Evaluation

- Accuracy
- Precision
- Recall
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Classification threshold analysis

### Deployment

- Model serialization with Joblib
- Streamlit
- Loading trained models
- Building an interactive prediction interface

### Software/Project Skills

- Project organization
- Documentation
- Reproducibility
- Environment management
- Version control

---

# 💡 Key Learning Outcomes

The project provided practical experience in taking a machine learning problem from raw data through to a working application.

The main learning outcomes were:

1. Understanding the importance of exploratory data analysis before modelling.
2. Identifying potential data quality issues rather than assuming that a dataset is clean.
3. Understanding why missing-value analysis is important even when no explicit null values are present.
4. Understanding the importance of feature scaling for Logistic Regression.
5. Using cross-validation to obtain a more reliable estimate of model performance.
6. Using hyperparameter tuning to improve model selection.
7. Understanding the trade-off between precision and recall.
8. Understanding how classification thresholds affect predictions.
9. Evaluating models using multiple performance metrics.
10. Saving and loading trained machine learning models.
11. Integrating a machine learning model into a web application.
12. Structuring a machine learning project for reproducibility and deployment.

---

# 📌 Conclusion

This project demonstrates an end-to-end machine learning workflow for diabetes risk prediction.

The final Logistic Regression model achieved:

- **74.03% test accuracy**
- **60.29% test precision**
- **75.93% test recall**
- **82.67% test ROC-AUC**

Five-fold cross-validation produced an average accuracy of approximately **78.99%** and an average ROC-AUC of approximately **87.15%**.

A classification threshold of **0.35** was selected after evaluating the trade-offs between accuracy, precision and recall.

The trained model was serialized using Joblib and integrated into an interactive Streamlit application.

Although the final test accuracy was slightly below the initial 75% target, the project successfully demonstrates the complete process of developing, evaluating and deploying a machine learning classification solution.

---

# ⚠️ Medical Disclaimer

This application is a **machine learning portfolio project** and is intended for educational purposes only.

The predictions generated by this application should **not** be interpreted as a medical diagnosis.

The model has not been clinically validated and should not be used to make medical decisions.

Always consult a qualified healthcare professional for medical advice, diagnosis and treatment.

---

## 👤 Author

**Siyabonga M Tshabalala**

Machine Learning / Data Analytics Portfolio Project

---

⭐ If you found this project useful, feel free to explore the notebook and Streamlit application to see the complete machine learning workflow.
