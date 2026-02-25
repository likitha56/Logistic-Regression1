# app.py

import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open(r"C:\Users\likitha\Desktop\LogisticRegression2\diabetes_prediction_model.pkl", "rb"))
print(model.feature_names_in_)

st.title("Diabetes Prediction App")

# Inputs
Pregnancies = st.number_input("Pregnancies", min_value=0, value=1)
Glucose = st.number_input("Glucose", min_value=0, value=120)
BloodPressure = st.number_input("BloodPressure", min_value=0, value=70)
SkinThickness = st.number_input("SkinThickness", min_value=0, value=20)
Insulin = st.number_input("Insulin", min_value=0, value=79)
BMI = st.number_input("BMI", min_value=0.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, value=0.5)
Age = st.number_input("Age", min_value=1, value=30)

if st.button("Predict"):

    input_data = pd.DataFrame(
        [[Glucose, BMI, Age]],
        columns=model.feature_names_in_
    )

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Person is Diabetic")
    else:
        st.success("✅ Person is Not Diabetic")