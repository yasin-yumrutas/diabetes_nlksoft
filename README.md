# Diyabet İlerleme Model Laboratuvarı

Scikit-learn diyabet veri seti üzerinde bir yıllık hastalık ilerleme ölçümünü inceleyen, regresyon ve deneysel ikili sınıflandırma yaklaşımlarını ortak bir değerlendirme hattında karşılaştıran makine öğrenmesi çalışmasıdır.

> Bu proje eğitim ve model karşılaştırma amaçlıdır. Klinik tanı, tedavi veya bireysel risk kararı üretmez. “Düşük/yüksek” sınıfları, hedef ilerleme skorunun medyanından türetilen deneysel etiketlerdir.

## Canlı model raporu

Canlı web raporu model kapsamını, sonuç tablolarını, doğrulama yaklaşımını ve analiz grafiklerini teknik ekiplerin hızla inceleyebileceği tek bir arayüzde sunar:

- https://diyabet-model-laboratuvari.vercel.app

## Doğrulanmış kapsam

- 442 gözlem ve 10 standartlaştırılmış temel özellik
- 5 türetilmiş etkileşim özelliği
- 9 regresyon ve 3 sınıflandırma yaklaşımı
- Hold-out test ve 5-fold çapraz doğrulama
- Residual, korelasyon, özellik katkısı, confusion matrix ve ROC-AUC analizi
- ElasticNet test R²: **0.4790**
- Lojistik regresyon ROC-AUC: **0.8474**

## Pipeline

```text
Veri yükleme
  → IQR aykırı değer kontrolü
  → özellik mühendisliği
  → train/test ayrımı
  → model eğitimi
  → 5-fold CV ve hold-out değerlendirme
  → rapor ve 300 DPI grafik üretimi
```

## Yerel çalıştırma

```bash
python -m pip install -r requirements.txt
python main.py
```

Masaüstü grafik inceleyicisi:

```bash
python gui_viewer.py
```

Statik web raporu için repository kökünü herhangi bir HTTP sunucusuyla açabilirsiniz:

```bash
python -m http.server 4173
```

Ardından `http://localhost:4173` adresine gidin.

## Üretilen çıktılar

- `outputs/reports/`: Regresyon ve sınıflandırma metrik tabloları
- `outputs/charts/`: Sekiz yüksek çözünürlüklü analiz grafiği
- `index.html`, `web.css`, `web.js`: Bağımlılıksız canlı model raporu

## Teknolojiler

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn ve bağımlılıksız HTML/CSS/JavaScript.

## Geliştirici

[Yasin Yumrutaş](https://github.com/yasin-yumrutas)
