# Design Document

## Overview

Aplikasi Streamlit untuk prediksi tingkat kecanduan smartphone menggunakan model CatBoost yang telah dilatih. Arsitektur terdiri dari tiga lapisan: antarmuka pengguna (`app.py`), preprocessing (`src/preprocessing.py`), dan inferensi model (`src/model.py`). Semua artifact ML (model, scaler, encoder) disimpan di folder `models/` dan dimuat sekali saat startup.

---

## Architecture

```
phone-addiction-predictor/
├── app.py                    # Streamlit UI + orchestration
├── train_and_save.py         # Script training ulang + simpan artifact
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Pipeline preprocessing (identik notebook)
│   └── model.py              # Load artifact + predict
├── models/
│   ├── catboost_model.cbm    # CatBoost native format
│   ├── scaler.pkl            # StandardScaler (joblib)
│   └── encoders.pkl          # OneHotEncoder (joblib)
├── requirements.txt
├── README.md
└── .gitignore
```

### Data Flow

```
User Input (19 raw features)
        ↓
  app.py: collect_input() → dict
        ↓
  preprocessing.preprocess_pipeline(input_dict, ohe, scaler)
    ├── clean_sleep_hours()
    ├── handle_missing_values()
    ├── encode_categorical(ohe)
    ├── engineer_features()
    ├── log_transform()
    └── scale_features(scaler)
        ↓
  model.predict(processed_df) → float
        ↓
  app.py: display_result(prediction)
```

---

## Component Design

### `src/preprocessing.py`

Semua fungsi menerima dan mengembalikan `pd.DataFrame`. Fungsi `preprocess_pipeline` adalah entry point utama untuk inferensi.

```python
def clean_sleep_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Strip kutip dan konversi Sleep_Hours ke float."""
    df = df.copy()
    df["Sleep_Hours"] = df["Sleep_Hours"].astype(str).str.strip('"').astype(float)
    return df

def handle_missing_values(df: pd.DataFrame, num_medians: dict, cat_modes: dict) -> pd.DataFrame:
    """Impute missing values menggunakan statistik dari training set."""
    df = df.copy()
    for col, val in num_medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col, val in cat_modes.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df

def encode_categorical(df: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame:
    """Terapkan OHE pada Gender dan Phone_Usage_Purpose."""
    cat_cols = ["Gender", "Phone_Usage_Purpose"]
    encoded = ohe.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols), index=df.index)
    df = df.drop(columns=cat_cols)
    return pd.concat([df, encoded_df], axis=1)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Buat 10 fitur turunan sesuai notebook."""
    x = df.copy()
    eps = 1e-3
    x["usage_zero_flag"] = (x["Daily_Usage_Hours"] <= 0).astype(int)
    denom_usage = x["Daily_Usage_Hours"].clip(lower=1)
    x["checks_per_hour"] = x["Phone_Checks_Per_Day"] / denom_usage
    x["apps_per_hour"] = x["Apps_Used_Daily"] / denom_usage
    x["screen_before_bed_ratio"] = x["Screen_Time_Before_Bed"] / denom_usage
    x["usage_to_sleep_ratio"] = x["Daily_Usage_Hours"] / (x["Sleep_Hours"] + eps)
    x["late_screen_ratio"] = x["Screen_Time_Before_Bed"] / (x["Sleep_Hours"] + eps)
    solo_usage = x["Time_on_Gaming"] + x["Time_on_Social_Media"]
    social_use = x["Family_Communication"] + x["Social_Interactions"]
    x["social_to_solo_ratio"] = social_use / (solo_usage + eps)
    mental_strain = x["Anxiety_Level"] + x["Depression_Level"]
    x["resilience_gap"] = x["Self_Esteem"] - mental_strain / 2.0
    x["high_gaming_x_sleep"] = x["Time_on_Gaming"] * x["Sleep_Hours"]
    x["social_media_x_anxiety"] = x["Time_on_Social_Media"] * x["Anxiety_Level"]
    return x

def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Terapkan log1p pada kolom skewed."""
    skewed_cols = [
        "Age", "checks_per_hour", "apps_per_hour", "screen_before_bed_ratio",
        "usage_to_sleep_ratio", "social_to_solo_ratio", "social_media_x_anxiety"
    ]
    x = df.copy()
    for col in skewed_cols:
        if col in x.columns:
            x[col] = np.log1p(x[col].clip(lower=0))
    return x

def scale_features(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Terapkan StandardScaler yang sudah di-fit."""
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)

def preprocess_pipeline(
    input_dict: dict,
    ohe: OneHotEncoder,
    scaler: StandardScaler,
    num_medians: dict,
    cat_modes: dict,
    feature_order: list
) -> pd.DataFrame:
    """Entry point: raw input dict → scaled DataFrame siap inferensi."""
    df = pd.DataFrame([input_dict])
    df = clean_sleep_hours(df)
    df["Phone_Usage_Purpose"] = df["Phone_Usage_Purpose"].replace("Unknown", np.nan)
    df = handle_missing_values(df, num_medians, cat_modes)
    df = encode_categorical(df, ohe)
    df = engineer_features(df)
    df = log_transform(df)
    df = df[feature_order]  # pastikan urutan kolom identik
    df = scale_features(df, scaler)
    return df
```

### `src/model.py`

```python
import joblib
from catboost import CatBoostRegressor
import streamlit as st

MODELS_DIR = "models"

@st.cache_resource
def load_artifacts():
    """Muat semua artifact sekali saat startup."""
    model = CatBoostRegressor()
    model.load_model(f"{MODELS_DIR}/catboost_model.cbm")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    artifacts = joblib.load(f"{MODELS_DIR}/encoders.pkl")
    # artifacts berisi: ohe, num_medians, cat_modes, feature_order
    return model, scaler, artifacts

def predict(model: CatBoostRegressor, processed_df) -> float:
    """Jalankan inferensi dan kembalikan nilai prediksi."""
    prediction = model.predict(processed_df)
    return float(prediction[0])
```

### `train_and_save.py`

Script standalone yang mereproduksi pipeline training dari notebook dan menyimpan semua artifact.

```python
# Langkah-langkah:
# 1. Load Phone_Addiction.csv
# 2. Data cleaning (drop cols, fix Sleep_Hours, fix Gender, replace Unknown, cap Age, drop duplicates)
# 3. Split X/y dengan test_size=0.2, random_state=284091, stratify=y (binned)
# 4. Imputation (fit on X_train)
# 5. Drop duplicates X_train setelah imputation
# 6. OHE fit on X_train[cat_cols]
# 7. Feature engineering + log transform
# 8. Scaler fit on X_train
# 9. Train CatBoost dengan best params
# 10. Simpan: model.save_model("models/catboost_model.cbm")
#             joblib.dump(scaler, "models/scaler.pkl")
#             joblib.dump({ohe, num_medians, cat_modes, feature_order}, "models/encoders.pkl")
```

**CatBoost best params** (dari hasil Optuna di notebook):
```python
params = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": 1,
    "verbose": 0
}
```
> Catatan: Parameter eksak harus diambil dari output Optuna di notebook. Nilai di atas adalah placeholder yang perlu diverifikasi.

### `app.py`

```python
# Layout:
# - st.title("Phone Addiction Level Predictor")
# - st.markdown(deskripsi singkat)
# - Form input dalam 3 kolom / expander:
#     Seksi 1: Informasi Dasar (Age, Gender, Daily_Usage_Hours, Sleep_Hours, Weekend_Usage_Hours)
#     Seksi 2: Aktivitas Smartphone (Phone_Checks_Per_Day, Apps_Used_Daily, Screen_Time_Before_Bed,
#               Time_on_Social_Media, Time_on_Gaming, Time_on_Education, Phone_Usage_Purpose)
#     Seksi 3: Kesehatan & Sosial (Anxiety_Level, Depression_Level, Self_Esteem,
#               Interllectual_Performance, Social_Interactions, Exercise_Hours, Family_Communication)
# - Tombol "Prediksi"
# - Hasil: st.metric, st.progress, st.info/warning/error berdasarkan kategori
```

---

## Data Models

### Input Dict (raw, dari form Streamlit)

```python
{
    "Age": float,
    "Gender": str,                    # "Male" | "Female" | "Other"
    "Daily_Usage_Hours": float,
    "Sleep_Hours": float,             # akan di-clean oleh preprocessor
    "Interllectual_Performance": int,
    "Social_Interactions": int,
    "Exercise_Hours": float,
    "Screen_Time_Before_Bed": float,
    "Phone_Checks_Per_Day": int,
    "Anxiety_Level": int,
    "Depression_Level": int,
    "Self_Esteem": int,
    "Apps_Used_Daily": int,
    "Time_on_Social_Media": float,
    "Time_on_Gaming": float,
    "Time_on_Education": float,
    "Phone_Usage_Purpose": str,       # "Browsing" | "Education" | "Gaming" | "Social Media" | "Other"
    "Family_Communication": int,
    "Weekend_Usage_Hours": float
}
```

### Artifact Bundle (`encoders.pkl`)

```python
{
    "ohe": OneHotEncoder,             # fitted OHE
    "num_medians": dict,              # {col: median_value}
    "cat_modes": dict,                # {col: mode_value}
    "feature_order": list[str]        # urutan kolom setelah semua transformasi
}
```

---

## Error Handling

| Kondisi | Penanganan |
|---|---|
| File artifact tidak ada | `FileNotFoundError` dengan nama file, ditangkap di `app.py` dengan `st.error()` |
| Input di luar range | Validasi di form Streamlit via `min_value`/`max_value` |
| Prediksi di luar [1, 10] | Clip hasil ke range [1.0, 10.0] sebelum ditampilkan |
| Error preprocessing | `try/except` di `app.py`, tampilkan `st.error()` |

---

## Correctness Properties

### Property 1: Pipeline Idempotence
Menjalankan `preprocess_pipeline` dua kali pada input yang sama harus menghasilkan output yang identik (tidak ada side effect).

```python
result1 = preprocess_pipeline(input_dict, ohe, scaler, ...)
result2 = preprocess_pipeline(input_dict, ohe, scaler, ...)
assert result1.equals(result2)
```

### Property 2: Output Shape Invariant
Output `preprocess_pipeline` harus selalu memiliki shape `(1, N)` di mana N adalah jumlah fitur yang diharapkan model.

```python
result = preprocess_pipeline(input_dict, ...)
assert result.shape[0] == 1
assert result.shape[1] == len(feature_order)
```

### Property 3: Prediction Range
Nilai prediksi setelah clipping harus selalu berada dalam rentang [1.0, 10.0].

```python
pred = predict(model, processed_df)
clipped = max(1.0, min(10.0, pred))
assert 1.0 <= clipped <= 10.0
```

### Property 4: Feature Order Consistency
Urutan kolom output preprocessor harus identik dengan `feature_order` yang disimpan saat training.

```python
result = preprocess_pipeline(input_dict, ...)
assert list(result.columns) == feature_order
```

### Property 5: No NaN in Output
Output `preprocess_pipeline` tidak boleh mengandung nilai NaN (semua missing value sudah diimputasi).

```python
result = preprocess_pipeline(input_dict, ...)
assert not result.isnull().any().any()
```
