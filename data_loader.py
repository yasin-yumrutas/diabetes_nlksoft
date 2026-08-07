"""
Data Loader Module for Diabetes Dataset.
Loads scikit-learn diabetes dataset, performs initial exploration, missing value checks,
outlier detection and handling.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

FEATURE_TR_MAP = {
    'age': 'Yaş (Age)',
    'sex': 'Cinsiyet (Sex)',
    'bmi': 'Vücut Kitle İndeksi (BMI)',
    'bp': 'Kan Basıncı (BP)',
    's1': 'Toplam Kolesterol (s1 - TCH)',
    's2': 'LDL Kolesterol (s2 - LDL)',
    's3': 'HDL Kolesterol (s3 - HDL)',
    's4': 'TCH/HDL Oranı (s4)',
    's5': 'Trigliserit Seviyesi (s5 - LTG)',
    's6': 'Kan Şekeri (s6 - GLU)',
    'target': 'Diyabet İlerleme Skoru (Target)'
}

def load_diabetes_data() -> pd.DataFrame:
    """
    Loads sklearn diabetes dataset as a pandas DataFrame with target column.
    """
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame.copy()
    return df

def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generates summary statistics, missing values report, and dataset info.
    """
    missing = df.isnull().sum().to_dict()
    stats = df.describe().T
    return {
        'shape': df.shape,
        'missing_values': missing,
        'summary_stats': stats
    }

def detect_outliers_iqr(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Detects outliers using Interquartile Range (IQR) method.
    Returns summary of outlier counts per feature.
    """
    if columns is None:
        columns = [c for c in df.columns if c != 'target']
        
    outlier_info = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': round((len(outliers) / len(df)) * 100, 2),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
    return pd.DataFrame(outlier_info).T

def cap_outliers_iqr(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Caps outliers using Winsorization based on IQR bounds.
    """
    df_capped = df.copy()
    if columns is None:
        columns = [c for c in df.columns if c != 'target']
        
    for col in columns:
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_capped[col] = np.clip(df_capped[col], lower_bound, upper_bound)
        
    return df_capped

if __name__ == '__main__':
    df = load_diabetes_data()
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    outlier_df = detect_outliers_iqr(df)
    print("\nOutlier Detection Summary:")
    print(outlier_df)
