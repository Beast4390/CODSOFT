import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

# Select required columns
df = df[['Genre', 'Director', 'Actor 1', 'Rating']]

# Remove missing values
df = df.dropna().copy()

# Encode categorical columns
le_genre = LabelEncoder()
le_director = LabelEncoder()
le_actor = LabelEncoder()

df['Genre'] = le_genre.fit_transform(df['Genre'])
df['Director'] = le_director.fit_transform(df['Director'])
df['Actor 1'] = le_actor.fit_transform(df['Actor 1'])

# Features and Target
X = df[['Genre', 'Director', 'Actor 1']]
y = df['Rating']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMovie Rating Prediction Results")
print("-" * 35)
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

import matplotlib.pyplot as plt
import joblib

plt.figure(figsize=(8,5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Movie Ratings")

plt.show()

mse = mean_squared_error(y_test, predictions)

print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))

joblib.dump(model, "model.pkl")
print("Model saved successfully!")
sample_movie = pd.DataFrame(
    [[100, 50, 200]],
    columns=['Genre', 'Director', 'Actor 1']
)

predicted_rating = model.predict(sample_movie)
print("Predicted Rating:", round(predicted_rating[0], 2))


