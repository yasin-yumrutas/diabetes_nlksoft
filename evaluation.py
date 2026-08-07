"""
Evaluation Module for Diabetes Regression and Classification.
Calculates MAE, MSE, RMSE, R2, Cross-Validation Scores, and Classification Metrics.
Exports evaluation tables as CSV and Markdown files.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_regression_models(model_results: dict, y_test: pd.Series) -> pd.DataFrame:
    """
    Computes MAE, MSE, RMSE, R2, and CV R2 for each regression model.
    Returns a formatted pandas DataFrame.
    """
    summary_data = []
    
    for name, res in model_results.items():
        y_pred = res['y_pred_test']
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        cv_r2 = res['cv_r2_mean']
        cv_std = res['cv_r2_std']
        
        summary_data.append({
            'Model': name,
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R² Score': round(r2, 4),
            '5-Fold CV R²': f"{cv_r2:.4f} ± {cv_std:.4f}"
        })
        
    df_metrics = pd.DataFrame(summary_data)
    df_metrics = df_metrics.sort_values(by='R² Score', ascending=False).reset_index(drop=True)
    return df_metrics

def evaluate_classification_models(clf_results: dict, y_test: pd.Series) -> pd.DataFrame:
    """
    Computes Accuracy, Precision, Recall, F1, and ROC-AUC for classification models.
    """
    clf_summary = []
    for name, res in clf_results.items():
        y_pred = res['y_pred']
        y_prob = res['y_prob']
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        
        clf_summary.append({
            'Model': name,
            'Doğruluk (Accuracy)': round(acc, 4),
            'Hassasiyet (Precision)': round(prec, 4),
            'Duyarlılık (Recall)': round(rec, 4),
            'F1-Skoru': round(f1, 4),
            'ROC-AUC Score': round(auc, 4) if not np.isnan(auc) else 'N/A'
        })
        
    df_clf_metrics = pd.DataFrame(clf_summary)
    df_clf_metrics = df_clf_metrics.sort_values(by='ROC-AUC Score', ascending=False).reset_index(drop=True)
    return df_clf_metrics

def _df_to_markdown(df: pd.DataFrame) -> str:
    """Helper to convert DataFrame to Markdown format without external dependencies."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = df.columns.tolist()
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        sep_line = "| " + " | ".join("---" for _ in headers) + " |"
        row_lines = []
        for _, row in df.iterrows():
            row_lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
        return "\n".join([header_line, sep_line] + row_lines)

def export_reports(df_reg: pd.DataFrame, df_clf: pd.DataFrame, output_dir: str = 'outputs/reports'):
    """
    Saves metrics tables to CSV and Markdown.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    csv_reg_path = os.path.join(output_dir, 'regression_model_metrics.csv')
    md_reg_path = os.path.join(output_dir, 'regression_model_metrics.md')
    
    csv_clf_path = os.path.join(output_dir, 'classification_model_metrics.csv')
    md_clf_path = os.path.join(output_dir, 'classification_model_metrics.md')
    
    df_reg.to_csv(csv_reg_path, index=False, encoding='utf-8-sig')
    with open(md_reg_path, 'w', encoding='utf-8') as f:
        f.write("# Diyabet İlerleme Tahmini - Regresyon Modelleri Karşılaştırması\n\n")
        f.write(_df_to_markdown(df_reg))
        
    df_clf.to_csv(csv_clf_path, index=False, encoding='utf-8-sig')
    with open(md_clf_path, 'w', encoding='utf-8') as f:
        f.write("# Diyabet Risk Derecelendirmesi - Sınıflandırma Modelleri Karşılaştırması\n\n")
        f.write(_df_to_markdown(df_clf))
        
    print(f"Reports successfully saved to '{output_dir}'.")
