import pandas as pd

## COLLECTION
# Membaca file CSV
df = pd.read_csv("kelulusan_mahasiswa.csv")

# Menampilkan informasi dataset
print(df.info())
print(df.head())



## CLEANING
# Cek missing value
print(df.isnull().sum())

# Hapus duplikat (kalau ada)
df = df.drop_duplicates()

# Deteksi outlier dengan boxplot
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.boxplot(x=df['IPK'])
plt.show()


##Exploratory Data Analyze (EDA)
# Statistik deskriptif
print(df.describe())

# Histogram IPK
plt.figure(figsize=(6,4))
sns.histplot(df['IPK'], bins=10, kde=True)
plt.show()

# Scatterplot
plt.figure(figsize=(6,4))
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.show()

# Heatmap korelasi
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()


##Splitting Dataset
from sklearn.model_selection import train_test_split

X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Check class distribution first
print("Class distribution:")
print(y.value_counts())

# For small datasets, we'll use simple random split without stratification
# Train 70%, sisanya 30%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Dari 30% dibagi 2 (validation 15%, test 15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


