import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load Dataset
df = pd.read_csv("data/IRIS.csv")

# Basic Information
print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSpecies Count:")
print(df["species"].value_counts())

# Species Distribution Plot
plt.figure(figsize=(8,5))
sns.countplot(x="species", data=df)
plt.title("Species Distribution")
plt.savefig("plots/species_distribution.png")
plt.show()

# Pair Plot
pair_plot = sns.pairplot(df, hue="species")
pair_plot.savefig("plots/pairplot.png")
plt.show()

# Features and Target
X = df.drop("species", axis=1)
y = df["species"]
print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("\nModel Training Completed!")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("plots/confusion_matrix.png")
plt.show()
# Save Model
with open("iris_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("\nModel saved successfully!")

# Test Prediction

sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)
prediction = model.predict(sample)