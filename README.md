# Admission-Prediction-Neural-Networks

This app has been built using **Streamlit** and deployed with **Streamlit Community Cloud**.

Visit the app here: [Link to your app]

Password: `streamlit`

This application predicts whether a student is likely to be admitted to a university based on inputs such as GPA, university rating, research experience, and other relevant factors. The model aims to help users assess the chances of admission based on historical data.

## Features
- **User-friendly interface** powered by Streamlit.
- Input form to enter details such as GPA, university rating, research experience, and other relevant factors.
- Real-time prediction of admission chances based on the trained model.
- Accessible via Streamlit Community Cloud.

## Dataset
The application is trained on the **Admission Prediction dataset**. This dataset includes features such as:

- **GPA**: Grade Point Average of the applicant.
- **University Rating**: The rating of the university the applicant is applying to.
- **Research**: Whether the applicant has research experience.
- **TOEFL Score**: English proficiency test score.
- **GRE Score**: Graduate Record Examination score.
- **SOP**: Statement of Purpose score.
- **LOR**: Letter of Recommendation score.
- **CGPA**: Cumulative Grade Point Average.

## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).

## Model
The predictive model is trained using the **Admission Prediction dataset**. It applies preprocessing steps such as encoding categorical variables and scaling numerical features. The classification model used may include algorithms such as Logistic Regression, Random Forest, or XGBoost.

## Future Enhancements
- Adding support for more datasets.
- Incorporating explainability tools like **SHAP** to provide insights into predictions.
- Adding visualizations to better represent user input and model predictions.

## Installation (for local deployment)
If you want to run the application locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/admission_prediction_application.git
   cd admission_prediction_application
