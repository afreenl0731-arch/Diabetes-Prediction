import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")   # ✅ ADD THIS

st.title("🩺 Diabetes Prediction System")

st.write("Enter patient details below:")

# Input fields with default values
pregnancies = st.number_input("Pregnancies", value=1)
glucose = st.number_input("Glucose Level", value=100)
blood_pressure = st.number_input("Blood Pressure", value=70)
skin_thickness = st.number_input("Skin Thickness", value=20)
insulin = st.number_input("Insulin Level", value=80)
bmi = st.number_input("BMI", value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", value=0.5)
age = st.number_input("Age", value=30)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    # ✅ APPLY SCALING
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.write("Prediction value:", prediction)  # Debug

    if prediction[0] == 1:
        st.error("⚠️ The person is Diabetic")
    else:
        st.success("✅ The person is Not Diabetic")