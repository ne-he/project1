# Implementation Plan

## Task List

- [x] 1. Setup Struktur Proyek
  - [x] 1.1 Buat folder `phone-addiction-predictor/` dengan subfolder `src/` dan `models/`
  - [x] 1.2 Buat file kosong: `app.py`, `src/__init__.py`, `src/preprocessing.py`, `src/model.py`, `train_and_save.py`
  - [x] 1.3 Buat file placeholder: `requirements.txt`, `README.md`, `.gitignore`

- [x] 2. Ekstrak Preprocessing Code (`src/preprocessing.py`)
  - [x] 2.1 Implementasi `clean_sleep_hours(df)` — strip kutip, konversi ke float
  - [x] 2.2 Implementasi `handle_missing_values(df, num_medians, cat_modes)` — impute numerik dengan median, kategorikal dengan modus
  - [x] 2.3 Implementasi `encode_categorical(df, ohe)` — OHE untuk `Gender` dan `Phone_Usage_Purpose`
  - [x] 2.4 Implementasi `engineer_features(df)` — buat 10 fitur turunan sesuai notebook
  - [x] 2.5 Implementasi `log_transform(df)` — `np.log1p` pada 7 kolom skewed
  - [x] 2.6 Implementasi `scale_features(df, scaler)` — terapkan StandardScaler
  - [x] 2.7 Implementasi `preprocess_pipeline(input_dict, ohe, scaler, num_medians, cat_modes, feature_order)` — gabungkan semua langkah

- [-] 3. Ekstrak Training Code dan Simpan Artifact (`train_and_save.py` + `src/model.py`)
  - [x] 3.1 Implementasi `train_and_save.py`:
    - Load `Phone_Addiction.csv`
    - Data cleaning (drop cols, fix Sleep_Hours, fix Gender, replace Unknown→NaN, cap Age>150, drop duplicates)
    - Train/test split (`test_size=0.2, random_state=284091`)
    - Fit imputer (median numerik, modus kategorikal) pada X_train
    - Drop duplicates X_train setelah imputation
    - Fit OHE pada X_train[cat_cols] dengan `drop=["Other","Other"]`
    - Feature engineering + log transform pada X_train dan X_test
    - Fit StandardScaler pada X_train, transform keduanya
    - Train CatBoostRegressor dengan best params dari notebook
    - Simpan `models/catboost_model.cbm`, `models/scaler.pkl`, `models/encoders.pkl`
    - Print RMSE dan R² pada test set sebagai verifikasi
  - [x] 3.2 Implementasi `load_artifacts()` di `src/model.py` dengan `@st.cache_resource`
  - [ ] 3.3 Implementasi `predict(model, processed_df)` di `src/model.py`

- [ ] 4. Buat Aplikasi Streamlit (`app.py`)
  - [ ] 4.1 Setup layout dasar: judul, deskripsi, import artifact via `load_artifacts()`
  - [ ] 4.2 Implementasi form input 19 fitur dalam 3 seksi:
    - Seksi "Informasi Dasar": Age, Gender, Daily_Usage_Hours, Sleep_Hours, Weekend_Usage_Hours
    - Seksi "Aktivitas Smartphone": Phone_Checks_Per_Day, Apps_Used_Daily, Screen_Time_Before_Bed, Time_on_Social_Media, Time_on_Gaming, Time_on_Education, Phone_Usage_Purpose
    - Seksi "Kesehatan & Sosial": Anxiety_Level, Depression_Level, Self_Esteem, Interllectual_Performance, Social_Interactions, Exercise_Hours, Family_Communication
  - [ ] 4.3 Hubungkan tombol "Prediksi" ke `preprocess_pipeline()` dan `predict()`
  - [ ] 4.4 Tampilkan hasil prediksi: nilai numerik (2 desimal) + `st.progress` bar
  - [ ] 4.5 Tampilkan interpretasi kategorikal:
    - < 4.0 → `st.success` "Rendah – Penggunaan smartphone Anda tergolong sehat."
    - 4.0–6.9 → `st.warning` "Sedang – Perhatikan pola penggunaan smartphone Anda."
    - ≥ 7.0 → `st.error` "Tinggi – Disarankan untuk mengurangi penggunaan smartphone."
  - [ ] 4.6 Tambahkan error handling dengan `try/except` dan `st.error()` untuk kegagalan preprocessing/inferensi

- [ ] 5. Setup Dependencies dan Dokumentasi
  - [ ] 5.1 Tulis `requirements.txt` dengan versi yang kompatibel:
    ```
    streamlit>=1.28.0
    catboost>=1.2.0
    scikit-learn>=1.3.0
    pandas>=2.0.0
    numpy>=1.24.0
    joblib>=1.3.0
    ```
  - [ ] 5.2 Tulis `.gitignore`:
    ```
    __pycache__/
    *.pyc
    .venv/
    *.egg-info/
    .DS_Store
    ```
  - [ ] 5.3 Tulis `README.md` dengan:
    - Deskripsi proyek
    - Instruksi instalasi: `pip install -r requirements.txt`
    - Instruksi training: `python train_and_save.py`
    - Instruksi menjalankan app: `streamlit run app.py`
    - Penjelasan singkat fitur input dan output

- [ ] 6. Testing dan Verifikasi
  - [ ] 6.1 Jalankan `train_and_save.py` dan verifikasi RMSE ≈ 0.362, R² ≈ 0.947 pada test set
  - [ ] 6.2 Jalankan `streamlit run app.py` dan test dengan input default (nilai tengah dari range)
  - [ ] 6.3 Test edge cases:
    - `Daily_Usage_Hours = 0` (usage_zero_flag = 1, denom_usage = 1)
    - `Sleep_Hours` sangat kecil (eps mencegah division by zero)
    - Input dengan nilai minimum dan maksimum semua fitur
  - [ ] 6.4 Verifikasi urutan kolom output preprocessor identik dengan `feature_order` dari artifact
  - [ ] 6.5 Perbaiki error yang ditemukan pada langkah 6.1–6.4
