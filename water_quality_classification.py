# Langkah 1: Import Library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Langkah 2: Memuat Dataset dari Path Lokal
# Gantilah path ke lokasi dataset di komputer Anda
file_path = r'C:\Users\DELL\water_quality_classification.csv'  # Pastikan nama file dataset benar
data = pd.read_csv('C:\Users\DELL\water_quality_classification.py\water_quality.csv')

# Langkah 3: Analisis Awal Data
print(data.info())
print(data.describe())
print("Jumlah nilai hilang pada setiap kolom:\n", data.isnull().sum())

# Menangani Nilai Hilang (Imputasi dengan Mean)
data.fillna(data.mean(), inplace=True)

# Langkah 4: Visualisasi Data
# Visualisasi distribusi setiap fitur sebelum normalisasi
for column in data.columns[:-1]:
    plt.figure()
    sns.histplot(data[column], kde=True)
    plt.title(f"Distribusi dari {column}")
    plt.show()

# Visualisasi matriks korelasi
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap Korelasi")
plt.show()

# Langkah 5: Menentukan Fitur dan Target
# Asumsi kolom target adalah 'Potability'
X = data.drop(columns=['Potability'])  
y = data['Potability']

# Normalisasi Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Langkah 6: Membagi Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Langkah 7: Pemodelan
# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# Support Vector Machine
svc = SVC()
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
svc_acc = accuracy_score(y_test, svc_pred)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# Langkah 8: Evaluasi
print("\nAkurasi Logistic Regression:", lr_acc)
print("Akurasi Support Vector Machine:", svc_acc)
print("Akurasi Random Forest:", rf_acc)

# Confusion Matrix
models = {'Logistic Regression': lr_pred, 'SVM': svc_pred, 'Random Forest': rf_pred}
for model_name, predictions in models.items():
    print(f"\nConfusion Matrix untuk {model_name}:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
