import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('breast_cancer_rf_model.joblib')

def preprocess_data(df):
    # Define columns to encode
    ordered_columns = ['T Stage', 'Grade', 'N Stage', '6th Stage', 
                      'Estrogen Status', 'Progesterone Status', 'A Stage']
    unordered_columns = ['Race', 'Marital Status']
    
    # Apply Label Encoding for ordered columns
    label_encoders = {}
    for col in ordered_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # Apply One-Hot Encoding for unordered columns
    df = pd.get_dummies(df, columns=unordered_columns, drop_first=False, dtype=int)
    
    # Drop unnecessary columns
    columns_to_drop = ['Regional Node Examined', 'Regional Node Positive', 
                      'Differentiate', 'Tumor Size (mm)', 'Survival Months']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    return df

def predict_survival(data):
    model = load_model()
    prediction = model.predict_proba(data)
    return prediction

def main():
    st.title('Breast Cancer Survival Prediction')
    st.write('Upload a CSV file with patient data to predict survival probability')
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            input_df = pd.read_csv(uploaded_file)
            
            # Preprocess data
            processed_df = preprocess_data(input_df)
            
            # Make prediction
            predictions = predict_survival(processed_df)
            
            # Add predictions to dataframe
            results_df = input_df.copy()
            results_df['Survival Probability'] = predictions[:, 1]
            results_df['Death Probability'] = predictions[:, 0]
            
            # Display results
            st.subheader('Prediction Results')
            st.dataframe(results_df)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Sample file format
    st.sidebar.subheader("Required Input Format")
    st.sidebar.write("""
    Your CSV file should contain these columns:
    - Age
    - Race
    - Marital Status 
    - T Stage
    - N Stage
    - 6th Stage
    - Grade
    - A Stage
    - Estrogen Status
    - Progesterone Status
    """)

if __name__ == '__main__':
    main()
