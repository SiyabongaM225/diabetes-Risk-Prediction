import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# LOAD MODEL

@st.cache_resource
def load_model():

    project_root = Path(__file__).resolve().parent

    model_path = (
        project_root
        / "models"
        / "diabetes_logistic_regression.pkl"
    )

    threshold_path = (
        project_root
        / "models"
        / "threshold.txt"
    )

    model = joblib.load(model_path)

    with open(threshold_path, "r") as file:
        threshold = float(file.read())

    return model, threshold


model, threshold = load_model()


# HERO SECTION

st.title("🩺 Diabetes Risk Prediction")

st.subheader(
    "An interactive machine-learning application for estimating diabetes risk."
)

st.write(
    """
    Enter the patient's health measurements below. The trained
    Logistic Regression model will calculate an estimated diabetes
    risk and classify the result using the selected probability threshold.
    """
)

st.info(
    "💡 **How it works:** Enter all eight measurements and select "
    "**Predict Diabetes Risk** to generate a prediction."
)

# PATIENT INFORMATION

st.markdown(
    '<div class="section-title">Patient Information</div>',
    unsafe_allow_html=True
)

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    # COLUMN 1

    with col1:

        st.subheader("Basic Measurements")

        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
            help="Number of times the patient has been pregnant."
        )

        glucose = st.number_input(
            "Glucose Level",
            min_value=1.0,
            max_value=300.0,
            value=120.0,
            step=1.0,
            help="Plasma glucose concentration."
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=1.0,
            max_value=200.0,
            value=70.0,
            step=1.0,
            help="Diastolic blood pressure measurement."
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=1.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            help="Triceps skin fold thickness."
        )

    # COLUMN 2

    with col2:

        st.subheader("Additional Measurements")

        insulin = st.number_input(
            "Insulin Level",
            min_value=1.0,
            max_value=1000.0,
            value=100.0,
            step=1.0,
            help="Two-hour serum insulin level."
        )

        bmi = st.number_input(
            "BMI",
            min_value=1.0,
            max_value=100.0,
            value=30.0,
            step=0.1,
            help="Body Mass Index."
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.01,
            max_value=3.0,
            value=0.5,
            step=0.01,
            help="Diabetes pedigree function score."
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30,
            step=1,
            help="Patient age in years."
        )

    st.write("")

    submitted = st.form_submit_button(
        "🔍 Predict Diabetes Risk",
        use_container_width=True
    )

# PREDICTION

if submitted:

    # Create DataFrame using the exact feature names
    # used during model training.

    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Generate probability
    probability = model.predict_proba(
        input_data
    )[0, 1]

    # Apply selected threshold
    prediction = int(
        probability >= threshold
    )

    risk_percentage = probability * 100


    # RESULTS

    st.divider()

    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns([1, 1])

    # RESULT MESSAGE

    with result_col1:

        if prediction == 1:

            st.error(
                "### Higher Predicted Risk"
            )

            st.write(
                """
                Based on the information provided, the model
                predicts a higher likelihood of diabetes.
                """
            )

        else:

            st.success(
                "### Lower Predicted Risk"
            )

            st.write(
                """
                Based on the information provided, the model
                predicts a lower likelihood of diabetes.
                """
            )

    # PROBABILITY

    with result_col2:

        st.metric(
            label="Estimated Diabetes Risk",
            value=f"{risk_percentage:.1f}%"
        )

        st.progress(
            min(probability, 1.0)
        )

        st.caption(
            f"Decision threshold: {threshold:.2f}"
        )


    # INPUT SUMMARY

    with st.expander("View submitted patient information"):

        display_data = pd.DataFrame({
            "Measurement": [
                "Pregnancies",
                "Glucose",
                "Blood Pressure",
                "Skin Thickness",
                "Insulin",
                "BMI",
                "Diabetes Pedigree Function",
                "Age"
            ],
            "Value": [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]
        })

        st.dataframe(
            display_data,
            hide_index=True,
            use_container_width=True
        )


# MODEL INFORMATION

st.divider()

with st.expander("📊 About the Machine Learning Model"):

    st.write(
        """
        This application uses a Logistic Regression classification
        model trained on the Pima Indians Diabetes Dataset.
        """
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    metric_col1.metric(
        "Test Accuracy",
        "74.03%"
    )

    metric_col2.metric(
        "Precision",
        "60.29%"
    )

    metric_col3.metric(
        "Recall",
        "75.93%"
    )

    metric_col4.metric(
        "ROC-AUC",
        "82.67%"
    )

    st.write("")

    st.write(
        """
        **Model configuration**

        - Algorithm: Logistic Regression
        - Feature preprocessing: StandardScaler
        - Regularization parameter: C = 0.1
        - Classification threshold: 0.35
        - Cross-validation: 5-fold
        """
    )

# METHODOLOGY

with st.expander("🔬 Project Methodology"):

    st.write(
        """
        The project followed an end-to-end machine-learning workflow:
        """
    )

    st.markdown(
        """
        1. **Exploratory Data Analysis**  
           Examined distributions, correlations and target balance.

        2. **Data Quality Assessment**  
           Checked for missing values, duplicates, invalid values
           and data types.

        3. **Preprocessing**  
           Features were standardized using `StandardScaler`.

        4. **Model Development**  
           A Logistic Regression classifier was trained.

        5. **Model Validation**  
           Five-fold cross-validation and hyperparameter tuning
           were used to evaluate model stability.

        6. **Threshold Analysis**  
           Different probability thresholds were evaluated.
           A threshold of 0.35 was selected to improve recall.

        7. **Deployment**  
           The trained pipeline was saved using Joblib and
           integrated into this Streamlit application.
        """
    )


# DISCLAIMER

st.divider()

st.warning(
    """
    **Important:** This application is for educational and
    portfolio purposes only. It is not a medical diagnostic tool.

    The prediction should not be used as a substitute for
    professional medical advice, diagnosis or treatment.
    """
)


# FOOTER
st.markdown(
    """
    <div class="footer">

        Built as an end-to-end machine-learning portfolio project
        using Python, Scikit-learn and Streamlit.

    </div>
    """,
    unsafe_allow_html=True
)