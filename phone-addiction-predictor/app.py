"""
app.py — Phone Addiction Level Predictor
Streamlit web application for predicting smartphone addiction level.
"""

import streamlit as st
from src.model import load_artifacts
from src.preprocessing import preprocess_pipeline

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phone Addiction Predictor",
    page_icon="📱",
    layout="centered",
)

# ── title & description ───────────────────────────────────────────────────────
st.title("📱 Phone Addiction Level Predictor")
st.markdown(
    """
    Aplikasi ini memprediksi **tingkat kecanduan smartphone** kamu berdasarkan
    pola penggunaan dan kondisi kesehatan mental.

    Model yang digunakan: **CatBoost Regressor** (RMSE ≈ 0.37, R² ≈ 0.945)

    Isi semua field di bawah, lalu klik **Prediksi**.
    """
)
st.divider()

# ── load artifacts ────────────────────────────────────────────────────────────
try:
    model, scaler, ohe, num_medians, cat_modes, feature_order = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
