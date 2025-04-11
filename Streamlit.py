import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, and feature columns
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('columns.pkl', 'rb') as f:
    required_columns = pickle.load(f)

st.title("🎓 Admission Prediction App")

def one_hot_encode_user_input(input_df, required_columns):
    # One-hot encode University_Rating and Research
    input_encoded = pd.get_dummies(input_df, columns=['University_Rating', 'Research'], dtype=int)

    # Add any missing columns and fill with 0
    for col in required_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Ensure order matches training
    input_encoded = input_encoded[required_columns]
    return input_encoded

# Collect user input
st.sidebar.header("Enter Applicant Information")
gre = st.sidebar.slider("GRE Score", 260, 340, 300)
toefl = st.sidebar.slider("TOEFL Score", 0, 120, 100)
univ_rating = st.sidebar.selectbox("University Rating", [1, 2, 3, 4, 5])
sop = st.sidebar.slider("SOP Strength (1-5)", 1.0, 5.0, 3.0, 0.5)
lor = st.sidebar.slider("LOR Strength (1-5)", 1.0, 5.0, 3.0, 0.5)
cgpa = st.sidebar.slider("CGPA (out of 10)", 0.0, 10.0, 8.0, 0.1)
research = st.sidebar.selectbox("Research Experience", ['No', 'Yes'])

# Convert categorical input
research_binary = 1 if research == 'Yes' else 0

# Create input DataFrame
input_data = pd.DataFrame({
    'GRE_Score': [gre],
    'TOEFL_Score': [toefl],
    'University_Rating': [str(univ_rating)],
    'SOP': [sop],
    'LOR': [lor],
    'CGPA': [cgpa],
    'Research': [str(research_binary)]
})

# Predict on button click
if st.button("Predict Admission"):
    input_encoded = one_hot_encode_user_input(input_data, required_columns)
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    result = "✅ Likely to be Admitted!" if prediction > 0.80 else "❌ Unlikely to be Admitted"
    st.subheader(result)
    
st.write("Loss Curve of the model:")
st.image("loss_curve.png", caption="Loss Curve", use_container_width=True)

