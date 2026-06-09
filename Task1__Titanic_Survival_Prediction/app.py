import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("titanic_survival_model.pkl")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div style="
background:linear-gradient(90deg,#0f172a,#1e3a8a);
padding:20px;
border-radius:15px;
text-align:center;
margin-bottom:20px;
">
<h1 style="color:white;">
🚢 Titanic Survival Prediction
</h1>
<p style="color:white;font-size:18px;">
Machine Learning Project using Random Forest Classifier
</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 Project Information")

st.sidebar.success("""
Model Used: Random Forest

Accuracy: 82.12%

ROC-AUC Score: 0.90
""")

st.sidebar.markdown("## 🚢 Titanic Facts")

st.sidebar.info("""
📅 Maiden Voyage: 10 April 1912

🌊 Sank: 15 April 1912

👥 Total People Onboard: 2,224

🛟 Survivors: Approximately 706

⚓ Route:
Southampton → New York
""")

# -----------------------------
# Main Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    st.subheader("📝 Passenger Details")

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    sex = 0 if gender == "Female" else 1

    age = st.slider(
        "Age",
        1,
        80,
        25
    )

    sibsp = st.number_input(
        "Number of Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0
    )

    parch = st.number_input(
        "Number of Parents / Children",
        min_value=0,
        max_value=10,
        value=0
    )

    fare = st.number_input(
        "Fare (£)",
        min_value=0.0,
        max_value=600.0,
        value=32.0
    )

    embarked = st.selectbox(
        "Embarked Port",
        ["C", "Q", "S"]
    )

    embarked_map = {
        "C": 0,
        "Q": 1,
        "S": 2
    }

    embarked_encoded = embarked_map[embarked]

with col2:

    st.subheader("📊 Passenger Summary")

    st.metric("Passenger Class", pclass)
    st.metric("Gender", gender)
    st.metric("Age", age)

    st.metric(
        "Fare (£)",
        f"£{fare:.2f}"
    )

    approx_inr = fare * 115

    st.metric(
        "Approx Fare (₹)",
        f"₹{approx_inr:,.0f}"
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button(
    "🔮 Predict Survival",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[
            pclass,
            sex,
            age,
            sibsp,
            parch,
            fare,
            embarked_encoded
        ]],
        columns=[
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked"
        ]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.success(
            f"🎉 Passenger Likely Survived\n\nConfidence: {probability:.2%}"
        )
    else:
        st.error(
            f"❌ Passenger Likely Did Not Survive\n\nConfidence: {(1-probability):.2%}"
        )

    st.progress(float(probability))

    st.write(
        f"### Survival Probability: {probability:.2%}"
    )

    st.write(
        f"### Non-Survival Probability: {(1-probability):.2%}"
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown(
    """
    <center>
    <h4>🚢 Titanic Survival Prediction Project</h4>
    Developed using Python, Streamlit & Machine Learning
    </center>
    """,
    unsafe_allow_html=True
)
