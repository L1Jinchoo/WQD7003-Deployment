import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('breast_cancer_rf_model.joblib')

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
            
            # Required columns
            required_columns = ['Age', 'T Stage', 'N Stage', '6th Stage', 'Grade', 
                              'A Stage', 'Estrogen Status', 'Progesterone Status', 
                              'Race1', 'Race2', 'Race3',
                              'Marital Status1', 'Marital Status2', 'Marital Status3',
                              'Marital Status4', 'Marital Status5']
            
            # Check if all required columns are present
            missing_cols = set(required_columns) - set(input_df.columns)
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return
            
            # Make prediction
            predictions = predict_survival(input_df)
            
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
    st.sidebar.subheader("Sample File Format")
    st.sidebar.write("Your CSV file should contain the following columns:")
    for col in required_columns:
        st.sidebar.write(f"- {col}")
    
    # Add explanation of one-hot encoded features
    st.sidebar.markdown("""
    ### Note on Race and Marital Status
    These features are one-hot encoded:
    - Race1, Race2, Race3 represent different race categories
    - Marital Status1-5 represent different marital status categories
    
    For each group, use 1 for the applicable category and 0 for others.
    """)

if __name__ == '__main__':
    main()
