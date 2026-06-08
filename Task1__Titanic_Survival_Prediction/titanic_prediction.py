# Titanic Survival Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Data/Titanic-Dataset.csv")

# -----------------------------
# Dataset Information
# -----------------------------
print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# -----------------------------
# Drop Unnecessary Columns
# -----------------------------
df.drop(
    ["PassengerId", "Name", "Ticket", "Cabin"],
    axis=1,
    inplace=True
)

# -----------------------------
# Handle Missing Values
# -----------------------------
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# -----------------------------
# Encode Categorical Columns
# -----------------------------
sex_encoder = LabelEncoder()
embarked_encoder = LabelEncoder()

df["Sex"] = pd.Series(
    sex_encoder.fit_transform(df["Sex"]),
    index=df.index
)

df["Embarked"] = pd.Series(
    embarked_encoder.fit_transform(df["Embarked"]),
    index=df.index
)

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("Survived", axis=1)
y = df["Survived"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Random Forest Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# Survival Distribution
# -----------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", data=df)

plt.title("Titanic Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Survival by Gender
# -----------------------------
plt.figure(figsize=(6, 4))
sns.countplot(
    x="Sex",
    hue="Survived",
    data=df
)

plt.title("Survival by Gender")
plt.xlabel("Gender (0=Female, 1=Male)")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Feature Importance Graph
# -----------------------------
importance = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(
    X.columns,
    importance
)

plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

# -----------------------------
# Feature Importance Table
# -----------------------------
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

# -----------------------------
# Model Comparison
# -----------------------------
print("\nModel Comparison:")

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}

for name, mdl in models.items():

    mdl.fit(X_train, y_train)

    pred = mdl.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    print(f"{name}: {acc:.4f}")

# -----------------------------
# ROC Curve
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(6, 4))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(
    model,
    "titanic_survival_model.pkl"
)

print(
    "\nModel saved as 'titanic_survival_model.pkl'"
)

print("\nProject Execution Completed Successfully!")