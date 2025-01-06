
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_model():
    return joblib.load('breast_cancer_rf_model.joblib')

def preprocess_data(df):
    # 标签编码
    ordered_columns = ['T Stage', 'Grade', 'N Stage', '6th Stage', 
                      'Estrogen Status', 'Progesterone Status', 'A Stage']
    for col in ordered_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # 手动创建独热编码
    # Race encoding
    df['Race_1'] = (df['Race'] == 'Black').astype(int)
    df['Race_2'] = (df['Race'] == 'Other').astype(int)
    df['Race_3'] = (df['Race'] == 'White').astype(int)
    
    # Marital Status encoding
    df['Marital Status_1'] = (df['Marital Status'] == 'Divorced').astype(int)
    df['Marital Status_2'] = (df['Marital Status'] == 'Married').astype(int)
    df['Marital Status_3'] = (df['Marital Status'] == 'Separated').astype(int)
    df['Marital Status_4'] = (df['Marital Status'] == 'Single').astype(int)
    df['Marital Status_5'] = (df['Marital Status'] == 'Widowed').astype(int)
    
    # 删除原始分类列和不需要的列
    columns_to_drop = ['Race', 'Marital Status', 'Regional Node Examined', 
                      'Regional Node Positive', 'Differentiate', 
                      'Tumor Size (mm)', 'Survival Months']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    return df

def predict_survival(data):
    model = load_model()
    prediction = model.predict_proba(data)
    return prediction

def main():
    st.title('Breast Cancer Survival Prediction')
    st.write('Upload a CSV file with patient data to predict survival probability')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            
            # 确保必需的列存在
            required_columns = ['Age', 'Race', 'Marital Status', 'T Stage', 
                              'N Stage', '6th Stage', 'Grade', 'A Stage',
                              'Estrogen Status', 'Progesterone Status']
            
            missing_cols = set(required_columns) - set(input_df.columns)
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return
            
            processed_df = preprocess_data(input_df)
            predictions = predict_survival(processed_df)
            
            results_df = input_df.copy()
            results_df['Survival Probability'] = predictions[:, 1]
            results_df['Death Probability'] = predictions[:, 0]
            
            st.subheader('Prediction Results')
            st.dataframe(results_df)
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    st.sidebar.subheader("Required Input Format")
    st.sidebar.write("Your CSV file should contain these columns:")
    for col in ['Age', 'Race', 'Marital Status', 'T Stage', 'N Stage', 
                '6th Stage', 'Grade', 'A Stage', 'Estrogen Status', 
                'Progesterone Status']:
        st.sidebar.write(f"- {col}")

if __name__ == '__main__':
    main()
