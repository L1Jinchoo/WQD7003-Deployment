
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
    label_encoders = {}
    for col in ordered_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # 独热编码
    # 使用数字而不是具体状态名称
    race_categories = ['Race_1', 'Race_2', 'Race_3']
    marital_categories = ['Marital Status_1', 'Marital Status_2',
                         'Marital Status_3', 'Marital Status_4',
                         'Marital Status_5']
    
    # 进行独热编码并重命名列
    df = pd.get_dummies(df, columns=['Race', 'Marital Status'], dtype=int)
    
    # 重命名列以匹配训练时的格式
    rename_dict = {
        'Race_Black': 'Race_1',
        'Race_Other': 'Race_2',
        'Race_White': 'Race_3',
        'Marital Status_Divorced': 'Marital Status_1',
        'Marital Status_Married': 'Marital Status_2',
        'Marital Status_Separated': 'Marital Status_3',
        'Marital Status_Single': 'Marital Status_4',
        'Marital Status_Widowed': 'Marital Status_5'
    }
    df = df.rename(columns=rename_dict)
    
    # 添加缺失的类别列，填充0
    for cat in race_categories + marital_categories:
        if cat not in df.columns:
            df[cat] = 0
    
    # 删除不需要的列
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
