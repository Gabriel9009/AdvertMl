import streamlit as st
import joblib
import os
import numpy as np

# Try multiple common locations for the model file
MODEL_PATHS = [
    "sales_model.pkl",                         # Same directory
    os.path.join("models", "sales_model.pkl"),  # models/ subdirectory
    os.path.join(os.path.dirname(__file__), "sales_model.pkl")  # Absolute path
]

def load_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                st.error(f"Error loading model from {path}: {str(e)}")
    st.error("Model file not found in any standard locations")
    return None

model = load_model()

if model:
    st.title("📈 Advertising Budget → Sales Predictor")
    
    tv = st.slider("TV Budget ($)", 0, 300, 100)
    radio = st.slider("Radio Budget ($)", 0, 50, 25)
    newspaper = st.slider("Newspaper Budget ($)", 0, 120, 30)
    
    if st.button("Predict Sales"):
        prediction = model.predict([[tv, radio, newspaper]])[0]
        st.success(f"Predicted Sales: ${prediction:,.2f}")
