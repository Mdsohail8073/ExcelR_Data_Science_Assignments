import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")
st.title("Diabetes Prediction App")
st.write("Enter the health measurements below to predict the model's diabetes outcome.")

MODEL_PATH = Path(__file__).with_name("logistic_model.pkl")

@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)

if not MODEL_PATH.exists():
    st.error("Model file not found. Run the notebook's Streamlit Deployment save-model cell first.")
    st.stop()

model = load_model()

with st.form("prediction_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
    insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age,
    }])

    try:
        prediction = model.predict(input_data)[0]
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
        else:
            probability = None

        if int(prediction) == 1:
            st.warning("Prediction: Diabetes outcome (class 1)")
        else:
            st.success("Prediction: No diabetes outcome (class 0)")

        if probability is not None:
            st.write(f"Model-estimated probability for class 1: {probability:.1%}")

        st.caption("This is an educational machine-learning demo, not a medical diagnosis.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
