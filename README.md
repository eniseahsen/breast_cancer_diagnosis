# Breast Health Diagnostic App
# Tıbbi Tanı Verileri ile Erken Evre Tümör Tespiti İçin Makine Öğrenmesi Tabanlı Tanı Sistemi  
Bu proje, **Breast Cancer Wisconsin (Diagnostic) Data Set** kullanılarak geliştirilmiş bir meme kanseri teşhis uygulamasıdır. Kullanıcılar, tümör özelliklerini girerek bu tümörün **malign (kötü huylu)** ya da **benign (iyi huylu)** olup olmadığını öğrenebilir. Ayrıca farklı makine öğrenimi modelleriyle tahmin yapılabilir, veriler görselleştirilebilir ve derin öğrenme ile daha güçlü tahminler alınabilir.

---

##  Özellikler

- Gelişmiş veri görselleştirme (korelasyon haritası, dağılım, histogramlar)
- Farklı makine öğrenimi modelleriyle karşılaştırmalı tahminler:
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Gaussian Naive Bayes
  - XGBoost
- MLP (Multi-Layer Perceptron) ile derin öğrenme tahmini (TensorFlow)
- Performans analizleri: accuracy, confusion matrix, classification report
- Streamlit ile sade ve sezgisel kullanıcı arayüzü

---
## Özet

Bu çalışma, “Wisconsin Tanısal Göğüs Sağlığı” veriseti kullanılarak geliştirilen bir tanı tahmin sistemi uygulamasıdır. Göğüs kanseri ve erken teşhisin önemi hakkında kullanıcıları bilinçlendirirken aynı zamanda makine öğrenmesi destekli modellerin tümör sınıflandırma üzerindeki performansını analiz etmeyi amaçlamaktadır. Geliştirilen uygulama, makine öğrenmesi algoritmalarından yararlanarak kullanıcıların girdiği tümör özelliklerine göre iyi huylu (benign) veya kötü huylu (malign) olasılıklarını tahmin etmektedir. Kullanıcılar, parametre seçimlerini yaptıkları makine öğrenmesi ve derin öğrenme algoritmalarını kullanarak bu algoritmaların tümör sınıflandırma üzerindeki performanslarını karşılaştırabilmektedir. Uygulama aynı zamanda verisetine ait istatistiksel bilgileri kullanarak veri analizi, veri görselleştirmesi ile tümör sınıflandırmada etkili olan faktörlerin incelenmesine olanak sağlamaktadır.

---

## I. Giriş

Bu uygulama, Python programlama dili, makine öğrenmesi ve derin öğrenme yöntemleri kullanılarak Streamlit üzerinde geliştirilmiştir. Estetik ve temaya uygun bir görüntü için arkaplan görselini URL üzerinden getiren bir fonksiyon kullanılmıştır. Sidebar temaya uygun olarak renklendirilmiştir. `@st.cache_data` dekoratörü ile uygulamada kullanılacak olan veri seti yüklenip ön işlemeye tabi tutulur. Bu ön işlemde makine öğrenmesi işlemleri için gerekli olmayan sütunlar silinir. Bu işlem yalnızca bir kez yapılır ve sonuç önbelleğe alınarak uygulamanın performansı artırılır. Uygulama; Welcome, Breast Cancer, Dictionary, Applications with Dataset ve Prediction sayfalarından oluşmaktadır. Bu sayfalar selectbox ile kullanıcı tarafından seçilmektedir.

---

## II. Welcome Sayfası

Bu sayfa uygulamaya giriş sayfasıdır. Uygulamanın amacından bahsetmektedir.

---

## III. Breast Cancer Sayfası

Bu sayfa göğüs kanserinden ve erken teşhisin öneminden bahsetmektedir.

---

## IV. Dictionary Sayfası

Bu sayfa veri setinde geçen ve tümör sınıflandırması için ihtiyaç duyulan değişkenlerin açıklamalarını içermektedir. Her bir özelliğin (feature) açıklamasını içeren iki sütunlu bir DataFrame oluşturulur. Ardından `st.dataframe()` fonksiyonu ile Streamlit arayüzünde etkileşimli bir tablo olarak görüntülenir.

---

## V. Applications with Dataset Sayfası

- **Dataset Preview:** Veri seti ile ilgili genel bilgiler kullanıcıya sunulur.  
- **Data Visualization:** Kullanıcılar "Pairplot", "Boxplot", "Violinplot", "Correlation Maps" görselleştirmelerini seçtiği özellik üzerinden inceleyebilmektedir. Seçilen özelliğe göre, Seaborn kütüphanesi kullanılarak `diagnosis` (tanı) sınıfına göre renklerle ayrılmış grafikler oluşturulur. Bu grafikler, her iki sınıf için (malignant ve benign) seçilen özelliğin merkezi eğilimi, dağılımı ve aykırı değerlerini karşılaştırmalı olarak gösterir. Grafikler `st.pyplot()` ile Streamlit arayüzünde sunulur ve kullanıcıya özelliklerin sınıflar üzerindeki etkisini analiz etme imkânı tanır.

- **Machine Learning Applications:**  
  Göğüs kanseri teşhisi amacıyla çeşitli makine öğrenmesi algoritmalarının eğitilmesini ve test edilmesini sağlar.  
  - `diagnosis` sütunu LabelEncoder ile sayısal forma dönüştürülür (M: 1, B: 0).  
  - Özellikler (X) ve hedef değişken (y) ayrıştırılır.  
  - `StandardScaler` ile özellikler ölçeklendirilir.  
  - Veri eğitim ve test kümelerine ayrılır.  
  - Kullanıcı arayüzden model seçimi yapabilir (SVM, Random Forest, XGBoost gibi).  
  - Model ve hiperparametreler seçildikten sonra eğitim yapılır.  
  - Eğitim sonrası test verisi ile tahmin yapılır; doğruluk, sınıflandırma raporu ve karışıklık matrisi görsel olarak sunulur.

- **Deep Learning Applications:**  
  - **MLP (Multilayer Perceptron):** İki ReLU aktivasyonlu gizli katman ve bir sigmoid aktivasyonlu çıkış katmanı kullanılır. Adam optimizasyon algoritması ve binary crossentropy kayıp fonksiyonu ile derlenir. Erken durdurma (EarlyStopping) stratejisi ile aşırı öğrenme önlenir.  
  - **Dropout Katmanlı ANN:** İlk katmanda Dropout uygulanarak aşırı öğrenme azaltılır. Kullanıcı dropout oranı ve epoch sayısını seçebilir. Eğitim sonrası doğruluk ve eğitim süreci grafiklerle gösterilir.

---

## VI. Prediction Sayfası

Bu sayfa, kullanıcıdan belirli tümör özelliklerini alarak eğitilmiş bir lojistik regresyon modeli ile tümörün iyi huylu (benign) veya kötü huylu (malignant) olup olmadığını tahmin eder.  

- Önceden eğitim yapılmış ve `joblib` ile kaydedilmiş lojistik regresyon modeli (`logistic_modelfinal.pkl`) ve standartlaştırıcı (`scaler.pkl`) yüklenir.  
- Kullanıcıya, modelde kullanılan 30 farklı özellik için minimum, maksimum ve ortalama değer aralıklarında kaydırıcılar (slider) sunulur.  
- Girilen veriler numpy dizisine dönüştürülür ve eğitimde kullanılan ölçeklendirme yöntemi ile standartlaştırılır.  
- Model bu standartlaştırılmış veriyi kullanarak tahmin yapar.  
- Sonuç `1` ise kötü huylu (malignant), `0` ise iyi huylu (benign) olarak gösterilir.  
- Uygulama sonunda bu tahminlerin profesyonel tıbbi değerlendirmelerin yerini almadığına dair bir uyarı mesajı gösterilir.

---
## Kullanılan Teknolojiler

- **Python 3.11**
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [TensorFlow](https://www.tensorflow.org/)

---

## Kurulum Adımları
Aşağıdaki adımları izleyerek projeyi yerel ortamınızda çalıştırabilirsiniz:

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/eniseahsen/breast_cancer_diagnosis.git
cd breast_cancer_diagnosis
```

### 2. Sanal Ortam Oluşturun
*Windows için*
```bash
python -m venv envs
```
*macOS/Linux için*
```bash
python3 -m venv envs
source envs/bin/activate
```
### 3. Gerekli Kütüphaneleri Kurun
```bash
pip install -r requirements.txt
```
### 4. Uygulamayı Başlatın
```bash
streamlit run stream_app.py
```
 *veya*
```bash
 python -m streamlit run stream_app.py
```
