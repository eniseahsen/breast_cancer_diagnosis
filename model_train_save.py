import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib


df = pd.read_csv("data.csv")
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)


X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]


le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


joblib.dump(model, "logistic_modelfinal.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler have been saved")
