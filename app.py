import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open("sleep_disorder_model.pkl", "rb") as f:
    model = pickle.load(f)

# Set up the title and description for the app
st.title("🛏️ Sleep Disorder Prediction")
st.write("Enter the details below to predict the type of sleep disorder.")

# Input fields for user data
age = st.slider("Age (years)", 18, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
occupation = st.text_input("Occupation")
sleep_duration = st.slider("Sleep Duration (hours)", 4, 12, 7)
quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
physical_activity = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])
stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obesity"])
blood_pressure = st.slider("Blood Pressure (mmHg)", 80, 180, 120)
heart_rate = st.slider("Heart Rate (bpm)", 40, 120, 70)
daily_steps = st.slider("Daily Steps", 0, 20000, 5000)

# Prepare data for prediction when the user clicks the button
if st.button("Predict"):
    # Convert inputs to a DataFrame to match the model's expected format
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Occupation': [occupation],
        'Sleep_duration': [sleep_duration],
        'Quality_of_sleep': [quality_of_sleep],
        'Physical_activity': [physical_activity],
        'Stress Level': [stress_level],
        'BMI_category': [bmi_category],
        'Blood_pressure': [blood_pressure],
        'Heart_rate': [heart_rate],
        'Daily Steps': [daily_steps]
    })
    
    # One-hot encoding of categorical columns (same as in your model)
    input_data_encoded = pd.get_dummies(input_data, columns=['Gender', 'Occupation', 'Physical_activity', 'BMI_category'])
    
    # Ensure all columns match the training data's columns (add missing columns as 0)
    missing_cols = set(model.feature_names_in_) - set(input_data_encoded.columns)
    for col in missing_cols:
        input_data_encoded[col] = 0
    input_data_encoded = input_data_encoded[model.feature_names_in_]  # Reorder columns to match the model

    # Perform prediction
    prediction = model.predict(input_data_encoded)
    prediction_label = ['None', 'Insomnia', 'Sleep Apnea'][prediction[0]]
    
    # Display the prediction result
    st.write(f"🛌 **Predicted Sleep Disorder:** {prediction_label}")



