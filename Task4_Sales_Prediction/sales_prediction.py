import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import pickle

# Load Dataset
df = pd.read_csv("advertising.csv")
print(df.head())

# Dataset Information
print(df.info())

# Check Missing Values
print(df.isnull().sum())

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")

plt.savefig("plots/correlation_heatmap.png")
plt.show()

# Pair Plot
sns.pairplot(df)
plt.savefig("plots/pairplot.png")
plt.show()

# Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Sales vs TV
plt.figure(figsize=(6,4))
sns.scatterplot(x='TV', y='Sales', data=df)
plt.title("Sales vs TV Advertising")
plt.savefig("plots/sales_vs_tv.png")
plt.show()

# Sales vs Radio
plt.figure(figsize=(6,4))
sns.scatterplot(x='Radio', y='Sales', data=df)
plt.title("Sales vs Radio Advertising")
plt.savefig("plots/sales_vs_radio.png")
plt.show()

# Sales vs Newspaper
plt.figure(figsize=(6,4))
sns.scatterplot(x='Newspaper', y='Sales', data=df)
plt.title("Sales vs Newspaper Advertising")
plt.savefig("plots/sales_vs_newspaper.png")
plt.show()

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
# Actual vs Predicted Graph
plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.savefig("plots/actual_vs_predicted.png")

plt.show()

# Example Prediction
new_data = pd.DataFrame({
    'TV':[150],
    'Radio':[25],
    'Newspaper':[30]
})

prediction = model.predict(new_data)

import pickle

with open("sales_model.pkl", "wb") as file:
    pickle.dump(model, file)
    
print("Predicted Sales:", prediction[0])
with open("output/model_output.txt", "w") as f:
    f.write(f"MAE: {mae}\n") 
    f.write(f"MSE: {mse}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R2 Score: {r2}\n")
    f.write(f"Predicted Sales: {prediction[0]}\n")