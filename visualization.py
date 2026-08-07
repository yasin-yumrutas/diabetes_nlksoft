"""
Visualization Module for Diabetes Project.
Generates publication-quality charts for EDA, Regression Diagnostics,
Model Comparison, Feature Importances, and Classification Risk Metrics.
Saves PNG files to outputs/charts/ and supports standalone interactive window pop-ups.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Set sleek modern theme & palette
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

CHARTS_DIR = 'outputs/charts'
os.makedirs(CHARTS_DIR, exist_ok=True)

def save_and_show_fig(fig, filename: str, show_window: bool = False):
    """
    Saves figure to PNG in outputs/charts/ and optionally displays it in a pop-up window.
    """
    filepath = os.path.join(CHARTS_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Chart saved: {filepath}")
    
    if show_window:
        plt.show()
    else:
        plt.close(fig)

def plot_correlation_heatmap(df: pd.DataFrame, show_window: bool = False):
    """
    1. Feature Correlation Heatmap
    """
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
        square=True, linewidths=.8, cbar_kws={"shrink": .8}, annot=True, fmt='.2f', ax=ax
    )
    
    ax.set_title('Klinik Özellikler ve Diyabet İlerleme Skoru Korelasyon Matrisi', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    save_and_show_fig(fig, '01_correlation_heatmap.png', show_window)
    return fig

def plot_target_distribution(df: pd.DataFrame, threshold: float, show_window: bool = False):
    """
    2. Target Distribution & Risk Classification Cutoff
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(df['target'], kde=True, color='#2b5c8f', bins=30, ax=ax, alpha=0.6)
    
    ax.axvline(threshold, color='#d9534f', linestyle='--', linewidth=2.5, label=f'Risk Eşiği (Medyan = {threshold:.1f})')
    ax.axvline(df['target'].mean(), color='#f0ad4e', linestyle=':', linewidth=2, label=f'Ortalama = {df["target"].mean():.1f}')
    
    ax.text(threshold - 35, ax.get_ylim()[1]*0.8, 'Düşük Risk Bölgesi\n(Target < 140)', fontsize=11, color='#1b4332', fontweight='bold')
    ax.text(threshold + 10, ax.get_ylim()[1]*0.8, 'Yüksek Risk Bölgesi\n(Target >= 140)', fontsize=11, color='#7209b7', fontweight='bold')
    
    ax.set_title('Diyabet İlerleme Skoru (Target) Dağılımı ve Risk Sınıflandırma Eşiği', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('1 Yıl Sonraki Diyabet İlerleme Skoru', fontsize=12)
    ax.set_ylabel('Hasta Sayısı (Frekans)', fontsize=12)
    ax.legend(fontsize=11)
    
    fig.tight_layout()
    save_and_show_fig(fig, '02_target_distribution_risk.png', show_window)
    return fig

def plot_actual_vs_predicted(y_test: pd.Series, reg_results: dict, top_n: int = 4, show_window: bool = False):
    """
    3. Actual vs Predicted Scatter Plots with Ideal y=x Fit Line for Regressors
    """
    top_models = sorted(reg_results.keys(), key=lambda k: reg_results[k]['cv_r2_mean'], reverse=True)[:top_n]
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()
    
    for i, name in enumerate(top_models):
        ax = axes[i]
        y_pred = reg_results[name]['y_pred_test']
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.75, color='#1f77b4', edgecolors='k', linewidth=0.5, s=45)
        
        # Ideal y=x line
        min_val = min(y_test.min(), y_pred.min()) - 10
        max_val = max(y_test.max(), y_pred.max()) + 10
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Mükemmel Tahmin (y = x)')
        
        r2 = reg_results[name]['cv_r2_mean']
        ax.set_title(f'{name}\n(5-Fold CV R² = {r2:.3f})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Gerçek Değerler (y_test)', fontsize=10)
        ax.set_ylabel('Tahmin Edilen Değerler (y_pred)', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
    fig.suptitle('Gerçek vs Tahmin Edilen Değerler Kıyaslaması (Actual vs Predicted)', fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_and_show_fig(fig, '03_actual_vs_predicted.png', show_window)
    return fig

def plot_residuals_analysis(y_test: pd.Series, y_pred: np.ndarray, model_name: str = 'XGBoost', show_window: bool = False):
    """
    4. Residual Analysis: Residuals vs Fitted & Residual Distribution
    """
    residuals = y_test - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Residuals vs Fitted
    ax1.scatter(y_pred, residuals, alpha=0.7, color='#d9534f', edgecolors='k', linewidth=0.5)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.8)
    ax1.set_title(f'{model_name} - Artıklar (Residuals) vs Tahmin Edilenler', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Tahmin Edilen Değerler (y_pred)', fontsize=11)
    ax1.set_ylabel('Hata / Artık Değer (y_test - y_pred)', fontsize=11)
    
    # 2. Residual Distribution (Normality Check)
    sns.histplot(residuals, kde=True, color='#4b6b94', ax=ax2, bins=25)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_title(f'{model_name} - Hataların Dağılımı (Normallik Kontrolü)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Hata Miktarı (Residual)', fontsize=11)
    ax2.set_ylabel('Frekans', fontsize=11)
    
    fig.tight_layout()
    save_and_show_fig(fig, f'04_residuals_{model_name.lower().replace(" ", "_")}.png', show_window)
    return fig

def plot_metrics_comparison(df_metrics: pd.DataFrame, show_window: bool = False):
    """
    5. Regression Model Performance Metrics Bar Charts (MAE, RMSE, R2)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    df_sorted = df_metrics.sort_values(by='R² Score', ascending=True)
    
    # R2 Score comparison
    bars1 = ax1.barh(df_sorted['Model'], df_sorted['R² Score'], color='#2a9d8f', edgecolor='black', alpha=0.85)
    ax1.set_title('Modellerin Açıklayıcılık Oranı (R² Score - Yüksek İstenir)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('R² Score', fontsize=11)
    ax1.set_xlim(0, max(df_sorted['R² Score']) * 1.15)
    for bar in bars1:
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.3f}",
                 va='center', fontsize=10, fontweight='bold')
                 
    # RMSE & MAE comparison
    y_pos = np.arange(len(df_sorted))
    width = 0.35
    
    ax2.barh(y_pos - width/2, df_sorted['RMSE'], width, label='RMSE (Kök Ortalama Kare Hata)', color='#e76f51', alpha=0.85)
    ax2.barh(y_pos + width/2, df_sorted['MAE'], width, label='MAE (Ortalama Mutlak Hata)', color='#f4a261', alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_sorted['Model'])
    ax2.set_title('Hata Metrikleri Karşılaştırması (MAE & RMSE - Düşük İstenir)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Hata Miktarı', fontsize=11)
    ax2.legend(loc='lower right', fontsize=10)
    
    fig.suptitle('Regresyon Modelleri Performans Metrikleri Kıyaslaması', fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_and_show_fig(fig, '05_metrics_comparison.png', show_window)
    return fig

def plot_feature_importances(model, feature_names: list, model_name: str = 'XGBoost', show_window: bool = False):
    """
    6. Feature Importances / Coefficients Analysis
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps['reg'], 'coef_'):
        importances = np.abs(model.named_steps['reg'].coef_)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        print(f"Model {model_name} does not have feature importances.")
        return None
        
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='#457b9d', edgecolor='black', alpha=0.85)
    ax.set_title(f'{model_name} - En Önemli Klinik Özellikler (Feature Importances)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Önem Derecesi (Importance Score)', fontsize=11)
    
    fig.tight_layout()
    save_and_show_fig(fig, f'06_feature_importances_{model_name.lower().replace(" ", "_")}.png', show_window)
    return fig

def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, model_name: str = 'Random Forest Classifier', show_window: bool = False):
    """
    7. Classification Confusion Matrix (Anlaşılır Risk Çıktısı)
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    
    labels = ['Düşük Risk (<140)', 'Yüksek Risk (>=140)']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
                
    ax.set_title(f'{model_name}\nKarmaşıklık Matrisi (Confusion Matrix)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Tahmin Edilen Risk Sınıfı', fontsize=11, fontweight='bold')
    ax.set_ylabel('Gerçek Risk Sınıfı', fontsize=11, fontweight='bold')
    
    fig.tight_layout()
    save_and_show_fig(fig, f'07_confusion_matrix_{model_name.lower().replace(" ", "_")}.png', show_window)
    return fig

def plot_roc_auc_curve(clf_results: dict, y_test: pd.Series, show_window: bool = False):
    """
    8. ROC-AUC and Precision-Recall Curves for Risk Classification
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (name, res) in enumerate(clf_results.items()):
        if res['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2.5, label=f'{name} (AUC = {roc_auc:.3f})')
            
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Rastgele Tahmin (AUC = 0.500)')
    ax.set_title('Diyabet Risk Sınıflandırması ROC-AUC Eğrileri', fontsize=13, fontweight='bold')
    ax.set_xlabel('Yanlış Pozitif Oranı (1 - Özgüllük / FPR)', fontsize=11)
    ax.set_ylabel('Doğru Pozitif Oranı (Duyarlılık / TPR)', fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    
    fig.tight_layout()
    save_and_show_fig(fig, '08_roc_auc_curves.png', show_window)
    return fig

if __name__ == '__main__':
    print("Visualization module ready.")
