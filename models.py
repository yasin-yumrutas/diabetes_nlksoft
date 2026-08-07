"""
Models Module for Diabetes Regression and Classification.
Trains, tunes, and evaluates regression and classification models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def prepare_train_test_data(df: pd.DataFrame, target_col: str = 'target', test_size: float = 0.2, random_state: int = 42):
    """
    Splits features and target into train and test sets.
    Returns X_train, X_test, y_train, y_test, feature_names.
    """
    drop_cols = [c for c in ['target', 'risk_class', 'risk_category'] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def train_regression_models(X_train, X_test, y_train, y_test, random_state: int = 42):
    """
    Trains standard baseline, regularized, and tree-based regression models.
    Returns dictionary of trained models and predictions.
    """
    models = {
        'Linear Regression': Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())]),
        'Ridge Regression': Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))]),
        'Lasso Regression': Pipeline([('scaler', StandardScaler()), ('reg', Lasso(alpha=0.1, random_state=random_state))]),
        'ElasticNet': Pipeline([('scaler', StandardScaler()), ('reg', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state))]),
        'SVR (RBF Kernel)': Pipeline([('scaler', StandardScaler()), ('reg', SVR(C=10.0, epsilon=0.1))]),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=random_state),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_state),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_state, verbosity=0)
    }
    
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        
        results[name] = {
            'model': model,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
        
    return results

def train_classification_models(X_train, X_test, y_train, y_test, random_state: int = 42):
    """
    Trains classification models to interpret high vs low diabetes risk.
    """
    clf_models = {
        'Lojistik Regresyon': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=random_state))]),
        'Random Forest Sınıflandırıcı': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state),
        'XGBoost Sınıflandırıcı': XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_state, eval_metric='logloss')
    }
    
    clf_results = {}
    for name, model in clf_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        clf_results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
    return clf_results

if __name__ == '__main__':
    from data_loader import load_diabetes_data
    from feature_engineering import add_engineered_features, add_classification_targets
    
    df = load_diabetes_data()
    df_fe = add_engineered_features(df)
    df_class, th = add_classification_targets(df_fe)
    
    X_train, X_test, y_train, y_test, feats = prepare_train_test_data(df_class, target_col='target')
    results = train_regression_models(X_train, X_test, y_train, y_test)
    print("Trained", len(results), "regression models successfully.")
