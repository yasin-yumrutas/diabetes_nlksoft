"""
Main Execution Script for Diabetes Machine Learning Project.
Executes complete pipeline: Data Loading -> Outlier Check -> Feature Engineering ->
Model Training (Regression & Classification) -> Metrics Evaluation & Reports ->
High-Resolution Plot Generation -> Interactive GUI Launch Option.
"""

import os
import sys
import argparse
import pandas as pd

# Reconfigure stdout for Windows console UTF-8 support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from data_loader import load_diabetes_data, get_data_summary, detect_outliers_iqr, cap_outliers_iqr
from feature_engineering import add_engineered_features, add_classification_targets
from models import prepare_train_test_data, train_regression_models, train_classification_models
from evaluation import evaluate_regression_models, evaluate_classification_models, export_reports
from visualization import (
    plot_correlation_heatmap,
    plot_target_distribution,
    plot_actual_vs_predicted,
    plot_residuals_analysis,
    plot_metrics_comparison,
    plot_feature_importances,
    plot_confusion_matrix,
    plot_roc_auc_curve
)

def run_pipeline(launch_gui_after: bool = False):
    print("=" * 80)
    print(" 🏥 DİYABET İLERLEME TAHMİNİ & RİSK SINIFLANDIRMA PROJESİ ")
    print("=" * 80)
    
    # 1. Load Data
    print("\n[1/6] Veri Seti Yükleniyor ve İnceleniyor...")
    df = load_diabetes_data()
    summary = get_data_summary(df)
    print(f" -> Veri Seti Boyutu: {summary['shape'][0]} hasta, {summary['shape'][1]} sütun")
    print(f" -> Eksik Değer (Missing Values) Sayısı: {sum(summary['missing_values'].values())}")
    
    # Outlier check
    outlier_df = detect_outliers_iqr(df)
    print("\n[2/6] Aykırı Değer (Outlier) Tespiti (IQR Yöntemi):")
    print(outlier_df.to_string())
    
    # Optional capping for extreme outliers
    df = cap_outliers_iqr(df)
    
    # 3. Feature Engineering
    print("\n[3/6] Özellik Mühendisliği (Feature Engineering) Uygulanıyor...")
    df_fe = add_engineered_features(df)
    df_class, risk_threshold = add_classification_targets(df_fe)
    print(f" -> Yeni türetilen klinik özellikler eklendi (Toplam Özellik: {df_fe.shape[1] - 1})")
    print(f" -> Risk Sınıflandırma Eşiği (Medyan Target): {risk_threshold:.1f}")
    
    # 4. Train Models
    print("\n[4/6] Regresyon ve Sınıflandırma Modelleri Eğitiliyor...")
    # Regression
    X_tr, X_te, y_tr, y_te, feature_names = prepare_train_test_data(df_class, target_col='target')
    reg_results = train_regression_models(X_tr, X_te, y_tr, y_te)
    
    # Classification
    X_tr_c, X_te_c, y_tr_c, y_te_c, _ = prepare_train_test_data(df_class, target_col='risk_class')
    clf_results = train_classification_models(X_tr_c, X_te_c, y_tr_c, y_te_c)
    
    # 5. Evaluate & Export Reports
    print("\n[5/6] Model Performans Metrikleri Hesaplanıyor ve Raporlanıyor...")
    df_reg_metrics = evaluate_regression_models(reg_results, y_te)
    df_clf_metrics = evaluate_classification_models(clf_results, y_te_c)
    
    print("\n--- REGRESYON MODELLERİ PERFORMANS TABLOSU (regresyon.pdf Standartları) ---")
    print(df_reg_metrics.to_string(index=False))
    
    print("\n--- RISK SINIFLANDIRMA MODELLERİ PERFORMANS TABLOSU (Anlaşılır Çıktı) ---")
    print(df_clf_metrics.to_string(index=False))
    
    export_reports(df_reg_metrics, df_clf_metrics)
    
    # 6. Generate & Save High-Res Charts
    print("\n[6/6] Yüksek Çözünürlüklü Grafik PNG Dosyaları Üretiliyor...")
    plot_correlation_heatmap(df_fe, show_window=False)
    plot_target_distribution(df_fe, risk_threshold, show_window=False)
    plot_actual_vs_predicted(y_te, reg_results, show_window=False)
    
    best_model_name = df_reg_metrics.iloc[0]['Model']
    best_y_pred = reg_results[best_model_name]['y_pred_test']
    plot_residuals_analysis(y_te, best_y_pred, model_name=best_model_name, show_window=False)
    plot_metrics_comparison(df_reg_metrics, show_window=False)
    
    best_model_obj = reg_results[best_model_name]['model']
    plot_feature_importances(best_model_obj, feature_names, model_name=best_model_name, show_window=False)
    
    best_clf_name = df_clf_metrics.iloc[0]['Model']
    best_clf_pred = clf_results[best_clf_name]['y_pred']
    plot_confusion_matrix(y_te_c, best_clf_pred, model_name=best_clf_name, show_window=False)
    plot_roc_auc_curve(clf_results, y_te_c, show_window=False)
    
    print("\n" + "=" * 80)
    print(" ✅ TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
    print(" Raporlar: 'outputs/reports/' dizinine kaydedildi.")
    print(" Grafikler: 'outputs/charts/' dizinine 300 DPI PNG olarak kaydedildi.")
    print("=" * 80)
    
    if launch_gui_after:
        print("\n🖥️ Masaüstü İnteraktif Grafik Pencereleri Başlatılıyor...")
        from gui_viewer import launch_gui
        launch_gui()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diyabet ML Projesi Ana Scripti")
    parser.add_argument('--gui', action='store_true', help="Grafik pencerelerini canlı masaüstü GUI olarak açar")
    args = parser.parse_args()
    
    run_pipeline(launch_gui_after=args.gui)
