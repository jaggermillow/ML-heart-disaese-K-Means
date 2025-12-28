# K-Means ile Kalp Hastalığı Teşhisi Projesi Raporu


**Öğrencinin adı-soyadı:** *Danial Pourrashidi Alibeiglou*


**Ders adi:** *Yapay zeka ve Makine Öğrenimine giriş*


**Öğrenci numarası:** 250121007


## Giriş

Bu proje, K-Means kümeleme algoritması kullanarak kalp hastalığı teşhisi yapmayı amaçlamaktadır. Kullanılan veri seti "heart.csv" dosyası olup, hastaların yaş, cinsiyet, göğüs ağrısı tipi (cp), dinlenme kan basıncı (trestbps), kolesterol seviyesi (chol), açlık kan şekeri (fbs), dinlenme EKG sonuçları (restecg), maksimum kalp atış hızı (thalach), egzersiz kaynaklı anjina (exang), eski tepe (oldpeak), eğim (slope), büyük damar sayısı (ca), thalassemia (thal) ve hedef değişken (target: 0 - hastalık yok, 1 - hastalık var) gibi özelliklerini içermektedir. Veri seti 1025 satır ve 14 sütundan oluşmaktadır.

K-Means, unsupervised öğrenme yöntemi olup, veriyi etiketsiz olarak kümelere ayırır. Bu projede, kalp hastalığı varlığını temsil etmek üzere 2 küme (n_clusters=2) oluşturulmuştur. Proje, K-Means'in genel kümeleme sağladığını ve sonuçların lojistik regresyona göre daha az kesin olabileceğini vurgular.

Proje adımları: Kütüphane yükleme, veri analizi, model eğitimi ve tahmin, küme merkezleri incelemesi, doğruluk hesaplama.

## Yöntem

### 1. Kütüphane Yükleme ve Veri Analizi
Kullanılan kütüphaneler:
- Pandas: Veri çerçeveleri için.
- NumPy: Sayısal işlemler için.
- Scikit-learn: K-Means modeli, veri bölme (train_test_split), ölçeklendirme (StandardScaler), ayarlanmış Rand indeksi (adjusted_rand_score) ve ortalama kare hata (mean_squared_error).

Veri seti pandas ile okunmuş ve analiz edilmiştir:
- Veri başlığı (`df.head(30)`): İlk 30 satır görüntülenmiştir, örneğin yaş ortalaması, cinsiyet dağılımı vb. incelenmiştir.

Veri ön işleme:
- Özellik matrisi (X): Tüm sütunlar (target dahil, çünkü unsupervised).
- Ölçeklendirme: StandardScaler ile özellikler standartlaştırılmış (`X_scaled = scaler.fit_transform(X)`), modelin daha iyi performans göstermesi sağlanmıştır.

### 2. Model Eğitimi ve Testi
Model: KMeans(n_clusters=2, random_state=25) nesnesi oluşturulmuş ve ölçeklenmiş veri ile eğitilmiş/tahmin yapılmış (`cluster_labels = kmeans.fit_predict(X_scaled)`).

Tahmin: Küme etiketleri veri setine eklenmiş (`df["cluster"] = cluster_labels`).

Küme İncelemesi:
- Küme merkezleri (`centers = kmeans.cluster_centers_`): Her kümenin ortalama değerleri hesaplanmış.
  - Küme 0: Ortalama yaş = -0.26, ortalama kalp basıncı = -0.06, ortalama kolesterol = 0.40.
  - Küme 1: Ortalama yaş = 0.45, ortalama kalp basıncı = 0.11, ortalama kolesterol = -0.68.

- Küme 1 verileri görüntülenmiş: 381 satır, hastalık riski yüksek bireyleri temsil ettiği varsayılmıştır.

Sonuç Tablosu Örneği (İlk 30 satırdan seçili):

| age | sex | cp | trestbps | chol | fbs | restecg | thalach | exang | oldpeak | slope | ca | thal | target | cluster |
|-----|-----|----|----------|------|-----|---------|---------|-------|---------|-------|----|------|--------|---------|
| 52  | 1   | 0  | 125      | 212  | 0   | 1       | 168     | 0     | 1.0     | 2     | 2  | 3    | 0      | 0       |
| 53  | 1   | 0  | 140      | 203  | 1   | 0       | 155     | 1     | 3.1     | 0     | 0  | 3    | 0      | 1       |
| 70  | 1   | 0  | 145      | 174  | 0   | 1       | 125     | 1     | 2.6     | 0     | 0  | 3    | 0      | 1       |
| 61  | 1   | 0  | 148      | 203  | 0   | 1       | 161     | 0     | 0.0     | 2     | 1  | 3    | 0      | 0       |
| 62  | 0   | 0  | 138      | 294  | 1   | 1       | 106     | 0     | 1.9     | 1     | 3  | 2    | 0      | 1       |
| 58  | 0   | 0  | 100      | 248  | 0   | 0       | 122     | 0     | 1.0     | 1     | 0  | 2    | 1      | 0       |
| 58  | 1   | 0  | 114      | 318  | 0   | 2       | 140     | 0     | 4.4     | 0     | 3  | 1    | 0      | 1       |
| 55  | 1   | 0  | 160      | 289  | 0   | 0       | 145     | 1     | 0.8     | 1     | 1  | 3    | 0      | 1       |
| 46  | 1   | 0  | 120      | 249  | 0   | 0       | 144     | 0     | 0.8     | 2     | 0  | 3    | 0      | 0       |
| 54  | 1   | 0  | 122      | 286  | 0   | 0       | 116     | 1     | 3.2     | 1     | 2  | 2    | 0      | 1       |

Doğruluk Hesabı: Mean squared error (MSE) ile küme etiketleri ve gerçek target karşılaştırılmış, sonuç 0.793'tür.

## Sonuç ve Değerlendirme

K-Means modeli, veriyi 2 kümeye ayırmış ve MSE değeri 0.793 olarak hesaplanmıştır. Bu, kümelemenin gerçek etiketlerle kısmen uyumlu olduğunu gösterir, ancak unsupervised doğası nedeniyle sonuçlar genel olup, lojistik regresyona göre daha düşük performanslıdır.

Danial Pourrashidi Alibeiglou,


Thanks to Enes Gorgulu.

Github Linki : https://github.com/jaggermillow/ML-heart-disaese-K-Means.git
