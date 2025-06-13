import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
#yardımcı sayfalar çağırılır
from utils.tasarim import set_page_config, add_background, style_sidebar
from utils.data import load_data
#yardımcı fonksiyonlar çağırılır
set_page_config()
add_background()
style_sidebar()
df = load_data()
#kullanıcının seçim yapabilmesi için bir sidebar
page = st.sidebar.selectbox("Menu", ["🌸Welcome","🌸Breast Cancer","🌸Dictionary","🌸Applications with Dataset", "🌸Prediction"])
#**************************************************************** WELCOME SAYFASİ ***************************************************************************************
#kullanıcıyı karşılayan sayfa. uygulamanın amacından bahseder.
if page == "🌸Welcome":
    st.markdown(
        '<h1 style="text-align:center;color:#ff9999;font-weight:bolder;font-size:30px;">🌸 Welcome 🌸</h1>',
        unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:18px'>
    This app helps you explore breast cancer data and understand the differences between 
    <font color='#e85a79'><b>Benign</b></font> and 
    <font color='#e85a79'><b>Malignant</b></font> tumors. 
    It allows users to explore and analyze the Breast Cancer Wisconsin (Diagnostic) dataset through visualizations and machine learning models. 
    It also uses machine learning models to make predictions based on the given features. 
    Easy to use and helpful for early detection.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<h1 style="text-align:center;color:#ff9999;font-weight:bolder;font-size:30px;">Don’t Wait – Check, Detect, Protect !</h1>',
        unsafe_allow_html=True)
#**************************************************************** BREAST CANCER SAYFASİ ***************************************************************************************
#göğüs kanseri hakkında bilgilenderime yapan sayfa
if page == "🌸Breast Cancer":
    st.markdown("""
        <div style='font-size:18px'>
        <h1 style="color:#ff9999;font-weight:bolder;font-size:30px;">Breast cancer,</h1>
        is one of the most common types of cancer affecting women worldwide. It occurs when abnormal cells in the breast grow uncontrollably, forming a tumor. While it can also affect men, it is far more prevalent in women.
        Early detection through regular self-exams, clinical screenings, and mammography significantly increases the chances of successful treatment. 
        Common warning signs include a lump in the breast, changes in breast shape, dimpling of the skin, or discharge from the nipple.
        Raising awareness, encouraging regular check-ups, and supporting research are vital steps in the fight against breast cancer. 
        <br>
        <font color='#e85a79'><b>Remember: early detection saves lives.</b></font> 
        </div>
        """, unsafe_allow_html=True)
#**************************************************************** DICTIONARY SAYFASİ ***************************************************************************************
#verisetindeki değişkenleri açıklayan sözlük sayfası
if page == "🌸Dictionary":
    st.markdown("<h3 style='color: #F08080;'>Data Dictionary</h2>", unsafe_allow_html=True)
    data_dict_en = {
        'diagnosis': 'Diagnosis of the breast tissue (M = malignant, B = benign)',
    'radius_mean': 'Mean of distances from center to points on the perimeter',
    'texture_mean': 'Mean of the standard deviation of gray-scale values',
    'perimeter_mean': 'Mean size of the core tumor perimeter',
    'area_mean': 'Mean area of the core tumor',
    'smoothness_mean': 'Mean of local variation in radius lengths',
    'compactness_mean': 'Mean of (perimeter squared divided by area) minus 1.0, a measure of tumor compactness',
    'concavity_mean': 'Mean severity of concave portions of the contour',
    'concave points_mean': 'Mean number of concave portions of the contour',
    'symmetry_mean': 'Mean symmetry of the tumor',
    'fractal_dimension_mean': 'Mean fractal dimension ("coastline approximation")',
    'radius_se': 'Standard error of the radius',
    'texture_se': 'Standard error of the texture',
    'perimeter_se': 'Standard error of the perimeter',
    'area_se': 'Standard error of the area',
    'smoothness_se': 'Standard error of the smoothness',
    'compactness_se': 'Standard error of the compactness',
    'concavity_se': 'Standard error of the concavity',
    'concave points_se': 'Standard error of the number of concave points',
    'symmetry_se': 'Standard error of the symmetry',
    'fractal_dimension_se': 'Standard error of the fractal dimension',
    'radius_worst': 'Worst (largest) value of radius',
    'texture_worst': 'Worst (largest) value of texture',
    'perimeter_worst': 'Worst (largest) value of perimeter',
    'area_worst': 'Worst (largest) value of area',
    'smoothness_worst': 'Worst (largest) value of smoothness',
    'compactness_worst': 'Worst (largest) value of compactness',
    'concavity_worst': 'Worst (largest) value of concavity',
    'concave points_worst': 'Worst (largest) value of concave points',
    'symmetry_worst': 'Worst (largest) value of symmetry',
    'fractal_dimension_worst': 'Worst (largest) value of fractal dimension'
    }

    data_en_df = pd.DataFrame(data_dict_en.items(),columns=["Feature","Description"])
    st.dataframe(data_en_df.style.set_table_styles(
        [{'selector': 'td', 'props': [('white-space', 'nowrap')]}]
    )) #td: hücre  nowrap: metin taşarsa kaydırma çubuğu gelir, alt satıra geçmez
    #st.table(data_en_df) #farklı bir format

#**************************************************************** APPLICATIONS WITH DATASET SAYFASİ ***************************************************************************************
#veriseti hakkında işlemler, veriseti ile görselleştirme, makine öğrenmesi ve derin öğrenme ile ilgili uygulamalar
if page == "🌸Applications with Dataset":
    menu = st.selectbox("Please Select", ["Dataset Preview", "Data Visualization", "Machine Learning Applications",
                                           "Deep Learning Applications"])

    if menu == "Dataset Preview": #veriseti ile ilgili bilgiler ve açıklamalar
        st.subheader("Dataset Preview")
        st.write("""
        - **Diagnosis**: Indicates whether the tumor is malignant (M) or benign (B).
        - **Mean features** (e.g., `radius_mean`, `texture_mean`): Average values of the feature across all nuclei in the image.
        - **Standard error features** (e.g., `radius_se`, `texture_se`): Standard error of the feature.
        - **Worst features** (e.g., `radius_worst`, `texture_worst`): Worst (largest) value for the feature.

        The features describe:
        - **Radius**: Distance from center to points on the perimeter.
        - **Texture**: Standard deviation of gray-scale values.
        - **Perimeter, Area, Smoothness**: Shape-related metrics.
        - **Compactness, Concavity, Symmetry, Fractal dimension**: Descriptors of the cell shapes.

        These features help in distinguishing between benign and malignant tumors.
        """)

        st.write(df.head(21)) #ilk 20 satır kullanıcıya gösterilir
        st.subheader("Diagnosis Class Distribution")
        st.write(df['diagnosis'].value_counts()) #verisetindeki Bening ve Malignant değerlerinin sayıları görselleştirlir
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x="diagnosis", hue="diagnosis", ax=ax1, palette="Set2", legend=False)
        st.pyplot(fig1)
        st.write(
            "**M = Malignant:** A malignant tumor is cancerous. It can grow aggressively and spread to other parts of the body (metastasize). *(GREEN)*")
        st.write(
            "**B = Benign**: A benign tumor is non-cancerous. It generally grows slowly and does not spread to other tissues. *(ORANGE)*")


    elif menu == "Data Visualization": #veri görselleştirmeleri
        # st.subheader("")

        plot_type = st.selectbox("Choose a graphic", ["Pairplot", "Boxplot", "Violinplot", "Correlation Maps"])

        if plot_type == "Pairplot":
            st.subheader("Pairplot of Selected Features")
            st.write(
                "A pairplot to visualize the relationships between selected features like radius mean, texture mean, perimeter mean, area mean, and smoothness mean and their correlation with the diagnosis (malignant or benign).")
            features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
            fig2 = sns.pairplot(df[features + ["diagnosis"]], hue="diagnosis", palette="Set2")
            fig2.fig.suptitle("Pairplot of Selected Features", y=1.02) #y: başlığın dikey eksendeki konumu
            st.pyplot(fig2)

        elif plot_type == "Boxplot":
            st.subheader("Boxplots of Selected Features by Diagnosis")
            st.write(
                "Boxplots to compare the distribution of 10 selected features between malignant and benign diagnoses in the Breast Cancer Wisconsin dataset.")
            features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                        'compactness_mean',
                        'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
            selected_feature = st.selectbox("Select a feature", features)
            fig3, ax3 = plt.subplots()
            sns.boxplot(x="diagnosis", y=selected_feature, hue="diagnosis", data=df, ax=ax3, palette="Set2")
            st.pyplot(fig3)

        elif plot_type == "Violinplot":
            st.subheader("Violinplot of Selected Features by Diagnosis")
            st.write(
                "Violinplots to visualize the distribution and density of selected features for malignant and benign diagnoses.")
            features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                        'compactness_mean',
                        'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
            selected_feature = st.selectbox("Select a feature", features)
            fig4, ax4 = plt.subplots()
            sns.violinplot(x="diagnosis", y=selected_feature, data=df, ax=ax4, color="skyblue")
            st.pyplot(fig4)

        elif plot_type == "Correlation Maps":
            st.subheader("Correlation Matrix Heatmap")
            st.write("This heatmap displays the correlation between numerical features in the dataset.")
            df_corr = df.copy()
            df_corr["diagnosis"] = df_corr["diagnosis"].map({"M": 1, "B": 0})
            fig5, ax5 = plt.subplots(figsize=(24, 20))
            sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", ax=ax5) #annot: değerlerin kutucuklara yazımı
            ax5.set_title("Correlation Matrix")
            st.pyplot(fig5)

            st.subheader("Correlation of Features with Diagnosis")
            st.write(
                "Visualizing the correlation of each feature with the diagnosis variable.This helps identify the most predictive features for distinguishing between benign and malignant tumors.")
            corr_matrix = df_corr.corr()
            diagnosis_corr = corr_matrix["diagnosis"].drop("diagnosis").sort_values(key=lambda x: abs(x), #key'e göre sıralar
                                                                                    ascending=False)
            fig6, ax6 = plt.subplots(figsize=(24, 20))
            sns.heatmap(diagnosis_corr.to_frame(), annot=True, cmap="coolwarm", ax=ax6)
            st.pyplot(fig6)

    elif menu == "Machine Learning Applications":  #makine öğrenmesi algoritmalaarı ile model eğitimi uygulamaları
        st.subheader("Prediction with Machine Learning Algorithms")

        from sklearn.preprocessing import LabelEncoder #kategoorik değerleri sayısal değerlere dönüştürmek için
        le = LabelEncoder()
        df["diagnosis"] = le.fit_transform(df["diagnosis"])
        #özellikler ve hedef değişkenin ayrılması
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']  # M: 1, B: 0
        #verilerin standartlaştırılması (ortalama=0, std=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        #veri setinin eğitim ve test olarak bölünmesi
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        #kullanıcının model seçimi yapması için seçim kutusu
        model_name = st.selectbox("Choose a Model and open the sidebar to arrange paramaters.",
                                  ["Random Forest", "SVM", "Logistic Regression", "KNN", "XGBoost", "Naive Bayes"])
        #seçilen modele göre sidebar’da parametre ayarlarının gösterilmesi
        if model_name == "SVM":
            st.sidebar.subheader("SVM Parameters")
            svm_c = st.sidebar.slider("C (Regularization)",0.01,10.0,1.0)
            svm_kernel = st.sidebar.selectbox("Kernel", ["linear","rbf","poly","sigmoid"])
            svm_gamma = st.sidebar.selectbox("Gamma",["scale","auto"])
        elif  model_name == "KNN":
            st.sidebar.subheader("KNN Parameters")
            knn_k = st.sidebar.slider("Number of Neighbors (k)",1,15,5)
            knn_metric = st.sidebar.selectbox("Distance Metric",["minkowski", "euclidean", "manhattan"])
        elif model_name == "Random Forest":
            st.sidebar.subheader("Random Forest Parameters")
            rf_n_estimators = st.sidebar.slider("Number of Trees",10,200,100)
            rf_max_depth = st.sidebar.slider("Max Depth",1,20,5)
            rf_criterion = st.sidebar.selectbox("Criterion",["gini","entropy"])
        elif model_name == "Logistic Regression":
            st.sidebar.subheader("Logistic Regression Parameters")
            logreg_c = st.sidebar.slider("C (Inverse oof Regularization)", 0.01,10.0,1.0)
            logreg_solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
        elif model_name == "XGBoost":
            st.sidebar.subheader("XGBoost Parameters")
            xgb_n_estimators = st.sidebar.slider("Number of Estimators",50,300,100)
            xgb_learning_rate = st.sidebar.slider("Learning Rate",0.01,0.5,0.1)
            xgb_max_depth = st.sidebar.slider("Max Depth",1,15,3)
        if st.button("Train the Model"): #"Modeli Eğit" butonuna basıldığında işlemleri başlat
            #seçilen modele göre uygun sınıf çağrılır ve parametreler set edilir
            if model_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=rf_n_estimators,
                                       max_depth=rf_max_depth,
                                       criterion=rf_criterion)
            elif model_name == "SVM":
                model = SVC(C=svm_c, kernel=svm_kernel, gamma = svm_gamma)
            elif model_name == "Logistic Regression":
                model = LogisticRegression(C=logreg_c,solver=logreg_solver,max_iter=1000)
            elif model_name == "KNN":
                model = KNeighborsClassifier(n_neighbors = knn_k, metric = knn_metric)
            elif model_name == "XGBoost":
                model = XGBClassifier(n_estimators=xgb_n_estimators,
                              learning_rate=xgb_learning_rate,
                              max_depth=xgb_max_depth,
                              use_label_encoder=False, eval_metric='logloss')
            elif model_name == "Naive Bayes":
                model = GaussianNB()
            #model eğitilir
            model.fit(X_train, y_train)
            #test verileri üzerinde tahmin yapılır
            y_pred = model.predict(X_test)
            #eğitim tamamlandı mesajı ve başarı metriği (accuracy)
            st.success("Model trained successfully!")
            st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
            #sınıflandırma raporu oluşturulur ve tablo olarak gösterilir
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()

            st.subheader("Classification Report")
            st.table(report_df.round(2)) #round2: virgülden sonra 2 basamak
            #sınıfların ne anlama geldiği belirtilir
            st.write(
                "**M: 1 (*Malignant*)**")
            st.write(
                "**B: 0 (*Benign*)**")
            #karışıklık matrisi görselleştirilir
            st.subheader(f"Confusion Matrix {model_name}")
            fig5, ax5 = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax5)
            st.pyplot(fig5)



    elif menu == "Deep Learning Applications":
        st.subheader("Prediction with Deep Learning Algorithms")
        #kullanıcı algoritma seçebilir
        plot_type_dl = st.selectbox("Choose an algorithm", ["MLP Fully Connected Neural Network", "ANN with Dropout"])
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder
        from tensorflow.keras import Input
        #kategorik hedef sütunu (diagnosis) sayısal değere dönüştürülür
        le = LabelEncoder()
        df["diagnosis"] = le.fit_transform(df["diagnosis"])
        #özellikler ve hedef değişkenin ayrılması
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']  # M: 1, B: 0
        #eğitim ve test setine ayırma işlemi
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #verilerin standartlaştırılması (derin öğrenme için önemlidir)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        #eğer kullanıcı MLP (katmanlı tam bağlı yapay sinir ağı) seçtiyse
        if plot_type_dl == "MLP Fully Connected Neural Network":
            #eğitim için epoch sayısı seçimi
            epochs = st.selectbox("Number of epochs", [10, 25,50,100,200],index=3)
            train_button = st.button("Train the Model")
            if train_button:
                # MLP (Multilayer Perceptron) modelinin oluşturulması
                model = Sequential()
                model.add(Input(shape=(X_train.shape[1],))) #giriş katmanı (özellik sayısı kadar) 1: sütun
                model.add(Dense(16, activation="relu")) #gizli katman 1
                model.add(Dense(8, activation="relu"))  #gizli katman 2
                model.add(Dense(1, activation="sigmoid")) #çıkış katmanı (binary sınıflandırma)
                #modelin derlenmesi (binary crossentropy kullanılır)
                model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                #erken durdurma: doğrulama başarımı değişmezse eğitimi erken bitirir
                from tensorflow.keras.callbacks import EarlyStopping
                early_stop = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
                #modelin eğitilmesi
                with st.spinner("Training the Model..."):
                    history = model.fit(
                        X_train_scaled, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(X_test_scaled, y_test),
                        callbacks=[early_stop],
                        verbose=0  # konsol çıktısını bastırma
                    )
                #eğitim ve doğrulama başarılarının gösterilmesi
                st.subheader("Model Performance")
                train_acc = history.history["accuracy"][-1]
                val_acc = history.history["val_accuracy"][-1]
                st.write(f"**Training Accuracy:** {train_acc:4f}")
                st.write(f"**Validation Accuracy:** {val_acc:4f}")
                #başarı grafiğinin çizdirilmesi
                st.subheader("Training History")
                fig, ax = plt.subplots()
                ax.plot(history.history["accuracy"], label="Train Accuracy")
                ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
                ax.legend()
                st.pyplot(fig)
        #eğer kullanıcı dropout içeren ANN (yapay sinir ağı) seçtiyse
        elif plot_type_dl == "ANN with Dropout":
            from tensorflow.keras.callbacks import EarlyStopping
            #kullanıcıdan dropout oranı seçimi
            plot_type_dr = st.selectbox("Dropout Rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], index=4)
            #epoch sayısı seçimi
            epochs2 = st.selectbox("Select number of epochs", [10, 25, 50, 100, 200], index=3)
            if plot_type_dr:
                train_button = st.button("Train the Model")
                if train_button:
                    #dropout içeren ANN modelinin kurulması
                    model = Sequential()
                    model.add(Dense(32, activation="relu", input_dim=X_train.shape[1])) #Giriş + gizli katman
                    model.add(Dropout(plot_type_dr)) #dropout ile overfitting azaltılır
                    model.add(Dense(1, activation="sigmoid")) #çıkış katmanı
                    #modelin derlenmesi
                    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
                    #erken durdurma callback'i
                    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
                    #modelin eğitilmesi
                    with st.spinner("Training the Model..."):
                        history = model.fit(
                            X_train_scaled, y_train, epochs=epochs2, batch_size=32,
                            validation_data=(X_test_scaled, y_test),
                            callbacks=[early_stop]
                        )
                    #model başarısının gösterilmesi
                    st.subheader("Model Performance")
                    train_acc = history.history["accuracy"][-1]
                    val_acc = history.history["val_accuracy"][-1]
                    st.write(f"**Training Accuracy:** {train_acc:4f}")
                    st.write(f"**Validation Accuracy:** {val_acc:4f}")
                    #eğitim geçmişinin grafiksel gösterimi
                    st.subheader("Training History")
                    fig, ax = plt.subplots()
                    ax.plot(history.history["accuracy"], label="Train Accuracy")
                    ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
                    ax.legend()
                    st.pyplot(fig)

#**************************************************************** PREDICTION SAYFASİ ***************************************************************************************
#kullanıcının kendi girdileriyle tümör tahmini yapabileceği sayfa
if page == "🌸Prediction":
    #önceden eğitilen logistic regression modeli ve scaler
    model = joblib.load("logistic_modelfinal.pkl")
    scaler = joblib.load("scaler.pkl")

    #tahmin için gerekli olan özellikler
    st.write("Select the values below to predict whether the tumor is benign or malignant.")
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

    #kullanıcının girdilerini depolayacak bir liste
    user_input = []
    for feature in feature_names:
        min_val = float(df[feature].min())  #sliderda göstermek için
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        #slider
        val = st.slider(label=feature, min_value=min_val,max_value=max_val,value=mean_val,step=(max_val - min_val)/1000)
        user_input.append(val)
    if st.button("Predict"): #Kullanıcı "Tahmin Et" butonuna bastığında
        input_array = np.array(user_input).reshape(1,-1) #girdiyi diziye çevir ve modele uygun boyuta getir
        input_scaled = scaler.transform(input_array) #girdiyi normalize et
        prediction = model.predict(input_scaled) #tahmini yap

        if prediction[0] == 1:
            st.error("Prediction: Malignant ") # 1:Kötü huylu
        else:
            st.success("Prediction: Bening")    # 0:İyi huylu

    st.write("While this app aids in diagnosis, it is not a replacement for a professional medical evaluation.")