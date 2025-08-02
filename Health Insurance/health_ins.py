import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

#Load Dataset
df = pd.read_csv("train.csv")

#Initial Exploration
print("🔹 First 5 Rows:\n", df.head())
print("\n🔹 Info:\n")
df.info()
print("\n🔹 Descriptive Statistics:\n", df.describe())
print("\n🔹 Missing Values:\n", df.isnull().sum())

#Label Encoding for Categorical Features
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Vehicle_Age'] = le.fit_transform(df['Vehicle_Age'])
df['Vehicle_Damage'] = le.fit_transform(df['Vehicle_Damage'])

#Exploratory Data Analysis (EDA)
sns.countplot(x='Response', data=df)
plt.title("Insurance Offer Acceptance Distribution")
plt.show()

sns.boxplot(x='Response', y='Age', data=df)
plt.title("Age Distribution by Response")
plt.show()

sns.barplot(x='Vehicle_Damage', y='Response', data=df)
plt.title("Vehicle Damage vs Response")
plt.show()

#Feature and Target Separation
X = df.drop(['id', 'Response'], axis=1)
y = df['Response']

#Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Predictions and Evaluation
y_pred = model.predict(X_test)

print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix")
plt.show()

#Feature Importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance Ranking")
plt.xlabel("Importance Level")
plt.ylabel("Feature")
plt.show()