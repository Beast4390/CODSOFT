import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Page Title
st.title("🎬 Movie Rating Predictor")

st.markdown("""
Predict movie ratings using **Random Forest Regression**
based on Genre, Director, and Actor.
""")

st.info(
    "Select a Genre, Director, and Lead Actor from the dropdowns and click Predict Rating."
)

# Genre Mapping
genre_map = {
    "Action": 0,
    "Comedy": 1,
    "Drama": 2,
    "Romance": 3,
    "Thriller": 4
}

# Director Mapping
director_map = {
    "S. S. Rajamouli": 50,
    "Christopher Nolan": 20,
    "Rajkumar Hirani": 30,
    "Sanjay Leela Bhansali": 40,
    "Rohit Shetty": 50
}

# Actor Mapping
actor_map = {
    "Shah Rukh Khan": 100,
    "Aamir Khan": 110,
    "Salman Khan": 120,
    "Prabhas": 190,
    "Allu Arjun": 140,
    "Ram Charan": 150,
    "Jr NTR": 160,
    "Deepika Padukone": 170
}

# Dropdown Inputs
genre = st.selectbox(
    "Genre 🎭",
    options=list(genre_map.keys())
)

director = st.selectbox(
    "Director 🎬",
    options=list(director_map.keys())
)

actor = st.selectbox(
    "Lead Actor ⭐",
    options=list(actor_map.keys())
)

# Prediction Button
if st.button("Predict Rating"):

    sample = pd.DataFrame(
        [[
            genre_map[genre],
            director_map[director],
            actor_map[actor]
        ]],
        columns=["Genre", "Director", "Actor 1"]
    )

    prediction = model.predict(sample)

    st.success(
        f"⭐ Predicted Movie Rating: {float(prediction[0]):.2f}/10"
    )