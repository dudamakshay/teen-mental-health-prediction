
# ============================================================
# Teen Mental Health Prediction Project
# Beginner Friendly Version
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

DATA_PATH = r"C:\Users\cwmbn\Downloads\teen-mental-health-analytics\data\Teen_Mental_Health_Dataset.csv"

# Read CSV file
df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!")

# ------------------------------------------------------------
# Dataset Overview
# ------------------------------------------------------------

print("Dataset Shape:", df.shape)
print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# ------------------------------------------------------------
# Data Cleaning
# ------------------------------------------------------------

# Remove duplicates
df = df.drop_duplicates()

# Remove missing values
df = df.dropna()

# ------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------

# Encode gender column
# male = 0, female = 1

df["gender_enc"] = df["gender"].map({
    "male": 0,
    "female": 1
})

# Create simple risk score
# Higher score = higher mental health risk

df["risk_score"] = (
    df["stress_level"] +
    df["anxiety_level"] +
    df["daily_social_media_hours"]
) / 3

# Sleep deficit feature
# Recommended sleep = 8 hours

df["sleep_deficit"] = np.maximum(0, 8 - df["sleep_hours"])

print("\nFeature Engineering Completed")

# ------------------------------------------------------------
# Exploratory Data Analysis (EDA)
# ------------------------------------------------------------

sns.set_style("darkgrid")

# 1. Depression Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='depression_label', data=df)
plt.title("Depression Distribution")
plt.xlabel("Depression Label")
plt.ylabel("Count")
plt.show()

# ------------------------------------------------------------
# 2. Sleep Hours vs Depression
# ------------------------------------------------------------

plt.figure(figsize=(7,5))
sns.boxplot(x='depression_label', y='sleep_hours', data=df)
plt.title("Sleep Hours vs Depression")
plt.xlabel("Depression Label")
plt.ylabel("Sleep Hours")
plt.show()

# ------------------------------------------------------------
# 3. Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------------------------
# Prepare Features and Target
# ------------------------------------------------------------

FEATURES = [
    "daily_social_media_hours",
    "sleep_hours",
    "stress_level",
    "anxiety_level",
    "addiction_level",
    "physical_activity",
    "gender_enc",
    "sleep_deficit"
]

X = df[FEATURES]
y = df["depression_label"]

# ------------------------------------------------------------
# Train Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Test Samples:", len(X_test))

# ------------------------------------------------------------
# Train Random Forest Model
# ------------------------------------------------------------

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

# Predictions
rf_pred = rf.predict(X_test)

# ------------------------------------------------------------
# Model Evaluation
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, rf_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, rf_pred))

# ------------------------------------------------------------
# Feature Importance
# ------------------------------------------------------------

importance = pd.Series(
    rf.feature_importances_,
    index=FEATURES
).sort_values(ascending=True)

plt.figure(figsize=(8,5))
importance.plot(kind='barh')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.show()

# ------------------------------------------------------------
# Save Model
# ------------------------------------------------------------

joblib.dump(rf, "mental_health_model.pkl")

print("\nModel saved successfully!")

# ------------------------------------------------------------
# Project Completed
# ------------------------------------------------------------

print("\nTeen Mental Health Prediction Project Completed Successfully!")
