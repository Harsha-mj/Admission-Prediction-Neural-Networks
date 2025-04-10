import streamlit as st
import numpy as np
import joblib

# Set the page title
st.title("🎓 UCLA Admission Predictor")
st.write("""
This app predicts the likelihood of being admitted to UCLA based on various features such as GRE Score, TOEFL Score, SOP, LOR, CGPA, and more.
""")

# Load the pre-trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Form for user input
with st.form("admission_details"):
    st.subheader("Enter Applicant Details")

    # User inputs based on dataset columns
    gre_score = st.number_input("GRE Score", min_value=260, max_value=340, step=1, value=320)
    toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, step=1, value=110)
    university_rating = st.selectbox("University Rating (1 to 5)", [1, 2, 3, 4, 5], index=3)
    sop = st.number_input("Statement of Purpose (SOP) Score", min_value=0.0, max_value=5.0, step=0.1, value=4.5)
    lor = st.number_input("Letter of Recommendation (LOR) Score", min_value=0.0, max_value=5.0, step=0.1, value=4.5)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01, value=9.0)
    research = st.selectbox("Research Experience (1 = Yes, 0 = No)", [0, 1], index=1)

    # Submit button
    submitted = st.form_submit_button("Predict Admission")

if submitted:
    # Prepare input data
    input_data = np.array([
        gre_score, toefl_score, university_rating, sop, lor, cgpa, research
    ]).reshape(1, -1)

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    predicted_admission_prob = model.predict_proba(input_scaled)[0, 1]  # Probability of admission

    # Display prediction
    st.subheader("🎓 Admission Probability:")
    st.success(f"🎯 Probability of Admission: {predicted_admission_prob * 100:.2f}%")

st.write("Loss Curve of the model:")
st.image("loss_curve.png", caption="Loss Curve", use_column_width=True)
