# Breast Health Diagnostic App

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
# Widnows için
```bash
python -m venv envs
```
# macOS/Linux için
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
# veya
```bash
 python -m streamlit run stream_app.py
```
