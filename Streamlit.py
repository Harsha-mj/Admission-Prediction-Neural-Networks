import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Set the page title
st.title("🎓 Admission Prediction App")
st.write("""
This app predicts the likelihood of admission to a university based on various features.
""")

# Load the pre-trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Function for one-hot encoding user input
def one_hot_encode_user_input(data):
    # Create a DataFrame from the user input
    df = pd.DataFrame(data, index=[0])

    # One-hot encode the University_Rating and Research columns
    df_encoded = pd.get_dummies(df, columns=['University_Rating', 'Research'], dtype=int)

    # Ensure all columns match the model's input features (add missing columns if needed)
    required_columns = model.feature_names_in_  # Get the columns used during model training
    missing_cols = set(required_columns) - set(df_encoded.columns)
    for col in missing_cols:
        df_encoded[col] = 0
    df_encoded = df_encoded[required_columns]  # Reorder columns to match model's expected input

    return df_encoded

# Form for user input
with st.form("admission_details"):
    st.subheader("Enter Student Details")

    # User inputs
    gre_score = st.number_input("GRE Score", min_value=260, max_value=340, value=300)
    toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
    university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
    sop = st.selectbox("Statement of Purpose (SOP)", [1, 2, 3, 4, 5])
    lor = st.selectbox("Letter of Recommendation (LOR)", [1, 2, 3, 4, 5])
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0)
    research = st.selectbox("Research Experience", [0, 1])

    # Submit button
    submitted = st.form_submit_button("Predict Admission Chance")

if submitted:
    # Prepare input data
    input_data = {
        'GRE Score': gre_score,
        'TOEFL Score': toefl_score,
        'University_Rating': university_rating,
        'SOP': sop,
        'LOR': lor,
        'CGPA': cgpa,
        'Research': research
    }

    # One-hot encode the input data
    input_data_encoded = one_hot_encode_user_input(input_data)

    # Scale input data
    input_data_scaled = scaler.transform(input_data_encoded)

    # Make prediction
    predicted_chance = model.predict(input_data_scaled)[0]

    # Display prediction
    if predicted_chance == 1:
        st.subheader("🎓 Admission Prediction:")
        st.success(f"💯 High chance of admission!")
    else:
        st.subheader("🎓 Admission Prediction:")
        st.error(f"❌ Low chance of admission")

st.write("Loss Curve of the model:")
st.image("loss_curve.png", caption="Loss Curve", use_container_width=True)

