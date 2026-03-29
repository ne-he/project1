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


# ── input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    # ── Seksi 1: Informasi Dasar ─────────────────────────────────────────────
    st.subheader("👤 Informasi Dasar")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia (Age)", min_value=1, max_value=100, value=18, step=1)
        daily_usage = st.number_input("Jam Pakai HP / Hari (Daily_Usage_Hours)", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
        sleep_hours = st.number_input("Jam Tidur (Sleep_Hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    with col2:
        gender = st.selectbox("Jenis Kelamin (Gender)", options=["Male", "Female", "Other"])
        weekend_usage = st.number_input("Jam Pakai HP Akhir Pekan (Weekend_Usage_Hours)", min_value=0.0, max_value=24.0, value=6.0, step=0.5)

    st.divider()

    # ── Seksi 2: Aktivitas Smartphone ────────────────────────────────────────
    st.subheader("📲 Aktivitas Smartphone")
    col3, col4 = st.columns(2)
    with col3:
        phone_checks = st.number_input("Cek HP / Hari (Phone_Checks_Per_Day)", min_value=0, max_value=500, value=50, step=1)
        apps_daily = st.number_input("Jumlah App Dipakai / Hari (Apps_Used_Daily)", min_value=0, max_value=100, value=10, step=1)
        screen_before_bed = st.number_input("Layar Sebelum Tidur / Jam (Screen_Time_Before_Bed)", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
        time_social_media = st.number_input("Waktu Media Sosial / Jam (Time_on_Social_Media)", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
    with col4:
        time_gaming = st.number_input("Waktu Gaming / Jam (Time_on_Gaming)", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
        time_education = st.number_input("Waktu Edukasi / Jam (Time_on_Education)", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
        phone_purpose = st.selectbox(
            "Tujuan Utama Pakai HP (Phone_Usage_Purpose)",
            options=["Browsing", "Education", "Gaming", "Social Media", "Other"],
        )

    st.divider()

    # ── Seksi 3: Kesehatan & Sosial ───────────────────────────────────────────
    st.subheader("🧠 Kesehatan & Sosial")
    col5, col6 = st.columns(2)
    with col5:
        anxiety = st.number_input("Tingkat Kecemasan (Anxiety_Level) [0–10]", min_value=0, max_value=10, value=5, step=1)
        depression = st.number_input("Tingkat Depresi (Depression_Level) [0–10]", min_value=0, max_value=10, value=5, step=1)
        self_esteem = st.number_input("Harga Diri (Self_Esteem) [0–10]", min_value=0, max_value=10, value=5, step=1)
        intellectual = st.number_input("Performa Intelektual (Interllectual_Performance) [0–100]", min_value=0, max_value=100, value=70, step=1)
    with col6:
        social_interactions = st.number_input("Interaksi Sosial / Hari (Social_Interactions)", min_value=0, max_value=20, value=5, step=1)
        exercise_hours = st.number_input("Jam Olahraga / Hari (Exercise_Hours)", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
        family_comm = st.number_input("Komunikasi Keluarga / Hari (Family_Communication)", min_value=0, max_value=20, value=5, step=1)

    st.divider()
    submitted = st.form_submit_button("🔍 Prediksi", use_container_width=True)


# ── prediction logic ──────────────────────────────────────────────────────────
if submitted:
    input_dict = {
        "Age":                       float(age),
        "Gender":                    gender,
        "Daily_Usage_Hours":         float(daily_usage),
        "Sleep_Hours":               float(sleep_hours),
        "Interllectual_Performance": int(intellectual),
        "Social_Interactions":       int(social_interactions),
        "Exercise_Hours":            float(exercise_hours),
        "Screen_Time_Before_Bed":    float(screen_before_bed),
        "Phone_Checks_Per_Day":      int(phone_checks),
        "Anxiety_Level":             int(anxiety),
        "Depression_Level":          int(depression),
        "Self_Esteem":               int(self_esteem),
        "Apps_Used_Daily":           int(apps_daily),
        "Time_on_Social_Media":      float(time_social_media),
        "Time_on_Gaming":            float(time_gaming),
        "Time_on_Education":         float(time_education),
        "Phone_Usage_Purpose":       phone_purpose,
        "Family_Communication":      int(family_comm),
        "Weekend_Usage_Hours":       float(weekend_usage),
    }

    try:
        processed = preprocess_pipeline(
            input_dict, ohe, scaler, num_medians, cat_modes, feature_order
        )
        from src.model import predict as run_predict
        prediction = run_predict(model, processed)
    except Exception as e:
        st.error(f"Terjadi error saat memproses input: {e}")
        st.stop()
