# 🏥 Diyabet İlerleme Tahmini ve Risk Analizi Makine Öğrenmesi Projesi

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red?logo=xgboost)
![License](https://img.shields.io/badge/License-MIT-green)

Klinik ve demografik verileri kullanarak hastaların **1 yıl sonraki diyabet ilerleme skorunu** tahmin eden ve **risk derecelendirmesi** yapan profesyonel makine öğrenmesi uygulamasıdır. `regresyon.pdf` isterlerine %100 uyumludur.

---

## ⚡ Hızlı Başlangıç & Kurulum

<details open>
<summary><b>🚀 Projeyi Çalıştırma Adımları (Genişletmek / Kapatmak İçin Tıklayın)</b></summary>

<br>

1. **Depoyu Klonlayın:**
   ```bash
   git clone https://github.com/yasin-yumrutas/diabetes_nlksoft.git
   cd diabetes_nlksoft
   ```

2. **Kütüphaneleri Yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Eğitimini ve Masaüstü Pop-Up Grafik Arayüzünü Başlatın:**
   ```bash
   python main.py --gui
   ```
</details>

---

## 🖥️ İnteraktif Masaüstü Grafik Pencereleri (`gui_viewer.py`)

<details>
<summary><b>🖥️ Canlı Pop-Up Grafik Görüntüleyici Kullanımı</b></summary>

<br>

Grafiklerin sadece klasörde resim olarak kalmaması, **tek tek canlı pencereler halinde ekranda açılıp incelenebilmesi** için geliştirilen GUI uygulamasını çalıştırmak için:

```bash
python gui_viewer.py
```
* **Özellikler:** Tıklanan grafiği bağımsız pencerede açar; **zoom (yakınlaştırma), pan (ekranda kaydırma) ve yüksek kalitede kaydetme** araçları sunar.

</details>

---

## 📊 Grafik Analizleri ve Anlamları (Açılır Sekmeler)

Aşağıdaki başlıklara tıklayarak her bir grafiği ve ne anlama geldiğini detaylıca görüntüleyebilirsiniz:

<details>
<summary><b>📊 1. Korelasyon Isı Haritası (Correlation Heatmap)</b></summary>

<br>

![Korelasyon Haritası](outputs/charts/01_correlation_heatmap.png)

> **💡 Ne Anlama Geliyor?:** Klinik özelliklerin (BMI, Kan Basıncı vb.) diyabet ilerlemesiyle ilişkisini gösterir. Sayılar $+1$'e yaklaştıkça o klinik değer arttığında hastanın diyabet ilerleme riski de doğrudan artmaktadır.
</details>

<details>
<summary><b>📈 2. Diyabet Skoru Dağılımı ve Risk Eşik Çizgisi</b></summary>

<br>

![Hedef Dağılımı](outputs/charts/02_target_distribution_risk.png)

> **💡 Ne Anlama Geliyor?:** Hastaların diyabet ilerleme skorlarının dağılımını gösterir. Kırmızı kesikli çizgi ($140.5$ medyan değeri), hastaları karar kolaylığı sağlamak adına **"Düşük Risk"** ve **"Yüksek Risk"** olarak 2 gruba ayırır.
</details>

<details>
<summary><b>🎯 3. Gerçek vs Tahmin Edilen Değerler (y = x Doğrusu)</b></summary>

<br>

![Gerçek vs Tahmin](outputs/charts/03_actual_vs_predicted.png)

> **💡 Ne Anlama Geliyor?:** Gerçek skorlar ile modellerin tahminlerini kıyaslar. Kırmızı çizgi ($y=x$) **mükemmel tahmini** temsil eder. Noktalar kırmıza çizgiye ne kadar yakınsa model o kadar az hata yapıyor demektir.
</details>

<details>
<summary><b>📉 4. Artık / Hata Analizi (Residuals Analysis)</b></summary>

<br>

![Hata Analizi](outputs/charts/04_residuals_elasticnet.png)

> **💡 Ne Anlama Geliyor?:** Modelin yaptığı hataların ($Hata = Gerçek - Tahmin$) sıfır etrafında tarafsız dağılıp dağılmadığını test eder. Çan eğrisine (normal dağılıma) uyum modelin güvenilir olduğunu gösterir.
</details>

<details>
<summary><b>🏆 5. Regresyon Modelleri Metrik Kıyaslaması</b></summary>

<br>

![Metrik Kıyaslaması](outputs/charts/05_metrics_comparison.png)

> **💡 Ne Anlama Geliyor?:** Tüm modellerin performansını kıyaslar. Yeşil çubuk **R² Score** (açıklayıcılık oranı - yüksek istenir), turuncu çubuklar ise **MAE ve RMSE** (hata miktarları - düşük istenir) değerlerini gösterir.
</details>

<details>
<summary><b>🔍 6. En Önemli Klinik Özellikler (Feature Importances)</b></summary>

<br>

![Özellik Önemleri](outputs/charts/06_feature_importances_elasticnet.png)

> **💡 Ne Anlama Geliyor?:** Yapay zeka modelinin tahminde bulunurken **en çok hangi klinik verilere önem verdiğini** sıralar. Grafik incelendiğinde **BMI (Vücut Kitle İndeksi)** ve **s5 (Serum Trigliserit Seviyesi)** en kritik 2 faktördür.
</details>

<details>
<summary><b>📋 7. Risk Sınıflandırması Karmaşıklık Matrisi (Confusion Matrix)</b></summary>

<br>

![Karmaşıklık Matrisi](outputs/charts/07_confusion_matrix_lojistik_regresyon.png)

> **💡 Ne Anlama Geliyor?:** Sınıflandırma modelinin kaç yüksek riskli ve düşük riskli hastayı doğru bildiğini gösterir. Sol üst ve sağ alt köşedeki yüksek sayılar modelin yüksek doğruluğunu kanıtlar.
</details>

<details>
<summary><b>⚡ 8. Sınıflandırma ROC-AUC Performans Eğrileri</b></summary>

<br>

![ROC-AUC Eğrileri](outputs/charts/08_roc_auc_curves.png)

> **💡 Ne Anlama Geliyor?:** Modelin sağlıklı ve yüksek riskli hastaları **birbirinden ayırma gücünü** ölçer. Eğri sol üst köşeye ne kadar yakınsa ($AUC \rightarrow 1.0$) modelin ayırt etme başarısı o kadar kusursuzdur (Lojistik Regresyon: $0.847$ AUC).
</details>

---

## 🏆 Model Performans Tabloları

<details>
<summary><b>📊 1. Regresyon Modelleri Performans Tablosu (Tıklayarak Açın)</b></summary>

<br>

| Model | MAE | MSE | RMSE | R² Score | 5-Fold CV R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ElasticNet** | **42.2134** | **2760.5132** | **52.5406** | **0.4790** | 0.4769 ± 0.0547 |
| **Lasso Regression** | 41.8664 | 2773.6090 | 52.6651 | 0.4765 | **0.4824 ± 0.0549** |
| **Ridge Regression** | 41.9155 | 2779.2911 | 52.7190 | 0.4754 | 0.4821 ± 0.0544 |
| **SVR (RBF Kernel)** | 42.2231 | 2780.6908 | 52.7323 | 0.4752 | 0.3867 ± 0.0534 |
| **Linear Regression** | 41.8349 | 2799.1066 | 52.9066 | 0.4717 | 0.4808 ± 0.0586 |
| **XGBoost** | 43.3862 | 2807.5814 | 52.9866 | 0.4701 | 0.4060 ± 0.0542 |
| **Gradient Boosting** | 43.5660 | 2827.2965 | 53.1723 | 0.4664 | 0.3982 ± 0.0576 |
| **Random Forest** | 43.2315 | 2883.4538 | 53.6978 | 0.4558 | 0.4237 ± 0.0723 |
| **Decision Tree** | 51.6773 | 4166.2606 | 64.5466 | 0.2136 | 0.1588 ± 0.2693 |

</details>

<details>
<summary><b>📋 2. Sınıflandırma (Risk Derecelendirme) Tablosu (Tıklayarak Açın)</b></summary>

<br>

| Model | Doğruluk (Accuracy) | Hassasiyet (Precision) | Duyarlılık (Recall) | F1-Skoru | ROC-AUC Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Lojistik Regresyon** | **0.7753** | **0.7381** | **0.7750** | **0.7561** | **0.8474** |
| **XGBoost Sınıflandırıcı** | 0.7753 | 0.7381 | 0.7750 | 0.7561 | 0.8235 |
| **Random Forest Sınıflandırıcı**| 0.7303 | 0.6818 | 0.7500 | 0.7143 | 0.8184 |

</details>

---

## 📚 Kullanılan Algoritmalar ve Açıklamaları

<details>
<summary><b>🤖 Model Terimleri ve Kullanım Amaçları (Tıklayarak Açın)</b></summary>

<br>

1. **Linear Regression:** Özellikler ile hedef skor arasına temel doğrusal çizgi çizer.
2. **Ridge Regression (L2):** Aşırı korelasyonlu değişkenlerde ezberlemeyi (overfitting) önlemek için ceza ekler.
3. **Lasso Regression (L1):** Önemsiz özellikleri sıfırlayarak otomatik özellik seçimi yapar.
4. **ElasticNet:** Ridge ve Lasso'nun birleşimidir; projemizde en yüksek R² skorunu ($0.4790$) vermiştir.
5. **Decision Tree:** Veriyi EVET/HAYIR sorularıyla dallandıran karar ağacı modelidir.
6. **Random Forest:** Yüzlerce karar ağacının oylamasını birleştiren güvenilir topluluk modelidir.
7. **Gradient Boosting:** Her yeni ağacı bir önceki ağacın hatalarını düzeltecek şekilde eğiten algoritmadır.
8. **XGBoost:** Gradient Boosting'in donanım ve hız açısından aşırı optimize edilmiş versiyonudur.
9. **SVR (RBF Kernel):** Veriyi yüksek boyutlu matematiksel uzaya taşıyarak karmaşık ilişkileri modeller.

</details>

---

### 👨‍💻 Geliştirici
* **Yasin Yumrutaş** - [VARUNET]((https://www.varunet.com/))
