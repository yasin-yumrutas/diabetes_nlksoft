"""
Desktop Interactive Chart Window Viewer (Tkinter GUI).
Provides a desktop dashboard interface to launch each machine learning chart
in an individual, standalone, interactive Matplotlib pop-up window.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_diabetes_data
from feature_engineering import add_engineered_features, add_classification_targets
from models import prepare_train_test_data, train_regression_models, train_classification_models
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
from evaluation import evaluate_regression_models, evaluate_classification_models

class ChartViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diyabet ML Projesi - İnteraktif Grafik Pencereleri")
        self.root.geometry("620x720")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)
        
        # Load data & run light pipeline once for interactive popup generation
        self.prepare_data_and_models()
        
        # UI Layout
        self.create_widgets()
        
    def prepare_data_and_models(self):
        """Prepares dataset and model artifacts for live window popups."""
        self.df = load_diabetes_data()
        self.df_fe = add_engineered_features(self.df)
        self.df_class, self.threshold = add_classification_targets(self.df_fe)
        
        # Regression
        self.X_tr, self.X_te, self.y_tr, self.y_te, self.feats = prepare_train_test_data(self.df_class, target_col='target')
        self.reg_results = train_regression_models(self.X_tr, self.X_te, self.y_tr, self.y_te)
        self.df_reg_metrics = evaluate_regression_models(self.reg_results, self.y_te)
        
        # Classification
        self.X_tr_c, self.X_te_c, self.y_tr_c, self.y_te_c, _ = prepare_train_test_data(self.df_class, target_col='risk_class')
        self.clf_results = train_classification_models(self.X_tr_c, self.X_te_c, self.y_tr_c, self.y_te_c)

    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#313244", pady=15)
        header_frame.pack(fill="x")
        
        title_lbl = tk.Label(
            header_frame, 
            text="Diyabet İlerleme & Risk Analizi", 
            font=("Helvetica", 16, "bold"), 
            fg="#cdd6f4", 
            bg="#313244"
        )
        title_lbl.pack()
        
        sub_lbl = tk.Label(
            header_frame, 
            text="Grafikleri Bağımsız Pencerelerde Açmak İçin Butonlara Tıklayın", 
            font=("Helvetica", 10, "italic"), 
            fg="#a6adc8", 
            bg="#313244"
        )
        sub_lbl.pack(pady=3)
        
        # Buttons Container Frame
        btn_frame = tk.Frame(self.root, bg="#1e1e2e", padx=30, pady=20)
        btn_frame.pack(fill="both", expand=True)
        
        buttons = [
            ("📊 1. Korelasyon Isı Haritası (Correlation Heatmap)", self.show_corr),
            ("📈 2. Diyabet Skoru Dağılımı ve Risk Eşiği", self.show_target_dist),
            ("🎯 3. Gerçek vs Tahmin Grafikleri (y = x Uyum Doğrusu)", self.show_actual_pred),
            ("📉 4. Hata/Artık (Residuals) Analizi & Normallik", self.show_residuals),
            ("🏆 5. Modellerin Metrik Kıyaslaması (MAE, RMSE, R²)", self.show_metrics),
            ("🔍 6. En Önemli Klinik Özellikler (Feature Importances)", self.show_feat_imp),
            ("📋 7. Risk Sınıflandırması Karmaşıklık Matrisi (Confusion Matrix)", self.show_confusion),
            ("⚡ 8. Sınıflandırma ROC-AUC Performans Eğrileri", self.show_roc),
            ("🚀 TÜM GRAFİKLERİ PENCERELERDE AÇ (Open All)", self.show_all)
        ]
        
        for text, cmd in buttons:
            is_main = "TÜM" in text
            btn = tk.Button(
                btn_frame, 
                text=text, 
                command=cmd,
                font=("Helvetica", 10, "bold" if is_main else "normal"),
                fg="#ffffff" if is_main else "#cdd6f4",
                bg="#89b4fa" if is_main else "#45475a",
                activebackground="#b4befe" if is_main else "#585b70",
                activeforeground="#11111b",
                bd=0,
                padx=10,
                pady=8,
                cursor="hand2"
            )
            btn.pack(fill="x", pady=5)
            
        # Footer
        footer_lbl = tk.Label(
            self.root,
            text="Tüm grafikler ayrıca 'outputs/charts/' klasörüne PNG olarak kaydedilmiştir.",
            font=("Helvetica", 9),
            fg="#6c7086",
            bg="#1e1e2e",
            pady=10
        )
        footer_lbl.pack(side="bottom")

    # Command handlers
    def show_corr(self):
        plot_correlation_heatmap(self.df, show_window=True)

    def show_target_dist(self):
        plot_target_distribution(self.df, self.threshold, show_window=True)

    def show_actual_pred(self):
        plot_actual_vs_predicted(self.y_te, self.reg_results, show_window=True)

    def show_residuals(self):
        best_model = self.df_reg_metrics.iloc[0]['Model']
        y_pred = self.reg_results[best_model]['y_pred_test']
        plot_residuals_analysis(self.y_te, y_pred, model_name=best_model, show_window=True)

    def show_metrics(self):
        plot_metrics_comparison(self.df_reg_metrics, show_window=True)

    def show_feat_imp(self):
        best_model = self.df_reg_metrics.iloc[0]['Model']
        model_obj = self.reg_results[best_model]['model']
        plot_feature_importances(model_obj, self.feats, model_name=best_model, show_window=True)

    def show_confusion(self):
        best_clf = list(self.clf_results.keys())[0]
        y_pred = self.clf_results[best_clf]['y_pred']
        plot_confusion_matrix(self.y_te_c, y_pred, model_name=best_clf, show_window=True)

    def show_roc(self):
        plot_roc_auc_curve(self.clf_results, self.y_te_c, show_window=True)

    def show_all(self):
        self.show_corr()
        self.show_target_dist()
        self.show_actual_pred()
        self.show_residuals()
        self.show_metrics()
        self.show_feat_imp()
        self.show_confusion()
        self.show_roc()

def launch_gui():
    root = tk.Tk()
    app = ChartViewerApp(root)
    root.mainloop()

if __name__ == '__main__':
    launch_gui()
