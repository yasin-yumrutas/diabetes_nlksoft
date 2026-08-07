"""
Feature Engineering Module for Diabetes Dataset.
Creates domain-specific metabolic ratios, interaction terms, polynomial terms,
and binary/multi-class targets for classification interpretation.
"""

import pandas as pd
import numpy as np

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds domain-relevant features to the dataset.
    """
    df_fe = df.copy()
    
    # 1. Metabolic Load Index (BMI * BP)
    df_fe['bmi_bp_interaction'] = df_fe['bmi'] * df_fe['bp']
    
    # 2. Glucose-BMI Load Index (BMI * Glucose s6)
    df_fe['bmi_glucose_interaction'] = df_fe['bmi'] * df_fe['s6']
    
    # 3. Lipid Composite Index (Triglyceride s5 * Glucose s6)
    df_fe['triglyceride_glucose_index'] = df_fe['s5'] * df_fe['s6']
    
    # 4. Cholesterol Ratio Proxy (s1 / (s3 + epsilon))
    # Note: features in sklearn dataset are centered/scaled around 0, so offset to avoid division by zero/negative issues
    s3_shifted = df_fe['s3'] - df_fe['s3'].min() + 0.01
    s1_shifted = df_fe['s1'] - df_fe['s1'].min() + 0.01
    df_fe['cholesterol_ratio_proxy'] = s1_shifted / s3_shifted
    
    # 5. Non-linear BMI term (BMI^2) to capture accelerating risk
    df_fe['bmi_squared'] = df_fe['bmi'] ** 2
    
    return df_fe

def add_classification_targets(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """
    Adds classification target columns based on target threshold for risk analysis.
    - risk_class: 0 (Low Risk) vs 1 (High Risk) based on median threshold (~140).
    - risk_category: 'Düşük Risk', 'Orta Risk', 'Yüksek Risk'
    """
    df_class = df.copy()
    
    if threshold is None:
        threshold = df_class['target'].median()  # Median ~140.0
        
    df_class['risk_class'] = (df_class['target'] >= threshold).astype(int)
    
    # Multi-class categories for deeper clinical visualization
    q33 = df_class['target'].quantile(0.33)
    q66 = df_class['target'].quantile(0.66)
    
    def get_category(val):
        if val < q33:
            return 'Düşük Risk'
        elif val < q66:
            return 'Orta Risk'
        else:
            return 'Yüksek Risk'
            
    df_class['risk_category'] = df_class['target'].apply(get_category)
    return df_class, threshold

if __name__ == '__main__':
    from data_loader import load_diabetes_data
    df = load_diabetes_data()
    df_fe = add_engineered_features(df)
    df_class, th = add_classification_targets(df_fe)
    print("Original features:", df.shape[1])
    print("Engineered features:", df_fe.shape[1])
    print("Risk Class distribution (Threshold={:.1f}):".format(th))
    print(df_class['risk_class'].value_counts())
    print("\nRisk Category distribution:")
    print(df_class['risk_category'].value_counts())
