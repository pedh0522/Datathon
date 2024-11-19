import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data():
    path = 'D:\\Downloads\\raw_datasets'
    
    # Load diabetes and alcohol datasets
    df_dia = pd.read_sas(os.path.join(path, 'diabetes.XPT'), format='xport')
    with open("D:\\Downloads\\variables.csv", mode='r', encoding='utf-8', errors='replace') as f:
        df_vars = pd.read_csv(f)
    df_alcohol = pd.read_sas(os.path.join(path, 'weight_history.XPT'), format='xport')
    
    # Rename columns based on df_vars
    relevant_rows = df_vars[(df_vars['Variable Name'].isin(df_dia.columns)) & (df_vars['Data File Name'] == 'DIQ_L')]
    df_dia.rename(columns=dict(zip(relevant_rows['Variable Name'], relevant_rows['Renamed_variables'])), inplace=True)
    
    relevant_rows = df_vars[(df_vars['Variable Name'].isin(df_alcohol.columns)) & (df_vars['Data File Name'] == 'WHQ_L')]
    df_alcohol.rename(columns=dict(zip(relevant_rows['Variable Name'], relevant_rows['Renamed_variables'])), inplace=True)
    
    return df_dia, df_vars, df_alcohol

def preprocess_csv(df):
    """Remove duplicates and NaN values."""
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df

def get_diabetes_status(df):
    """Get lists of diabetic and non-diabetic patients."""
    dia_patients = df[df['EverTold_Diabetes'] == 1]['sequence_no'].tolist()
    non_dia_patients = df[df['EverTold_Diabetes'] == 2]['sequence_no'].tolist()
    return dia_patients, non_dia_patients

def split_by_diabetes_status(df, dia_patients, non_dia_patients):
    """Split dataset based on diabetes status."""
    dia_df = df[df['sequence_no'].isin(dia_patients)]
    non_dia_df = df[df['sequence_no'].isin(non_dia_patients)]
    return dia_df, non_dia_df

def add_diabetes_status(df_feature, df_diabetes, target_column='EverTold_Diabetes'):
    """Add diabetes status to feature DataFrame."""
    return df_feature.merge(df_diabetes[['sequence_no', target_column]], on='sequence_no', how='left')
