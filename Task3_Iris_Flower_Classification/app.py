import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# -----------------------------
# Custom CSS (Pop Art Theme)
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

.title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    color: #ff006e;
}

.subtitle {
    text-align: center;
    color: #333333;
    font-size: 1.1rem;
}

.pop-card {
    background: linear-gradient(135deg,#ffbe0b,#fb5607);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-weight: bold;
    margin-bottom: 15px;
}

.result-card {
    background: #8338ec;
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.footer {
    text-align:center;
    color:gray;
    padding-top:20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    "<div class='title'>🌸 Iris Flower Classification</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Machine Learning Powered Iris Species Prediction</div>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    use_container_width=True
)

# -----------------------------
# About
# -----------------------------
st.markdown(
    """
<div class='pop-card'>
🎯 Random Forest Classifier <br>
📊 Accuracy: 100% <br>
🌼 Predict Setosa, Versicolor & Virginica
</div>
""",
    unsafe_allow_html=True
)

# -----------------------------
# Input Section
# -----------------------------
st.subheader("🌿 Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length",
        min_value=0.0,
        value=5.1
    )

    petal_length = st.number_input(
        "Petal Length",
        min_value=0.0,
        value=1.4
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width",
        min_value=0.0,
        value=3.5
    )

    petal_width = st.number_input(
        "Petal Width",
        min_value=0.0,
        value=0.2
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Species", use_container_width=True):

    sample = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    prediction = model.predict(sample)[0]

    st.markdown(
        f"""
        <div class='result-card'>
        Prediction: {prediction}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dynamic Images
    if prediction == "Iris-setosa":

        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
            caption="🌸 Iris Setosa",
            use_container_width=True
        )

    elif prediction == "Iris-versicolor":

        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
            caption="🌼 Iris Versicolor",
            use_container_width=True
        )

    else:

        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
            caption="🌺 Iris Virginica",
            use_container_width=True
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown(
    """
<div class='footer'>
👨‍💻 Developed by <b>Nikhil Gembali</b><br>
CODSOFT Data Science Internship
</div>
""",
    unsafe_allow_html=True
)