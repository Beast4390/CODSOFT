import streamlit as st
import pickle
import pandas as pd

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📈",
    layout="centered"
)

# =====================================
# Load Trained Model
# =====================================
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "sales_model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# =====================================
# Custom CSS
# =====================================
st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #334155
    );
}

.block-container{
    padding-top: 3rem;
    max-width: 700px;
}

h1,h2,h3,p,label{
    color:white !important;
}

.stButton > button{
    width:100%;
    height:55px;
    border:none;
    border-radius:12px;
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );
    color:white;
    font-size:18px;
    font-weight:600;
}

.stButton > button:hover{
    opacity:0.9;
}

.footer{
    text-align:center;
    color:#cbd5e1;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# Header
# =====================================
st.markdown(
    """
    <h1 style='text-align:center;'>
        📈 Sales Prediction
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align:center;
              color:#cbd5e1;
              font-size:18px;'>
        Predict Product Sales Based on Advertising Budgets
    </p>
    """,
    unsafe_allow_html=True
)

# =====================================
# Information Box
# =====================================
st.info(
    """
    Enter your advertising budget for TV, Radio, and Newspaper promotions.
    
    The system will estimate the expected sales based on past advertising and sales trends.
    """
)
# =====================================
# Input Section
# =====================================
st.subheader("📊 Advertising Budget Input")

tv = st.number_input(
    "📺 TV Advertising Budget",
    min_value=0.0,
    value=150.0,
    step=1.0
)

radio = st.number_input(
    "📻 Radio Advertising Budget",
    min_value=0.0,
    value=25.0,
    step=1.0
)

newspaper = st.number_input(
    "📰 Newspaper Advertising Budget",
    min_value=0.0,
    value=30.0,
    step=1.0
)

predict = st.button("🚀 Predict Sales")

# =====================================
# Prediction
# =====================================
if predict:

    data = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })

    prediction = model.predict(data)

    st.success(
        f"📈 Estimated Sales: {prediction[0]:.2f}"
    )
# =====================================
# Footer
# =====================================
st.write("---")

st.markdown(
    """
    <div class="footer">
        <b>Developed by Nikhil Gembali</b><br>
        Built with Python, Scikit-Learn, Pandas and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

