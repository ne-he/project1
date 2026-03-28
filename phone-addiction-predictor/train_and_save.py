"""
train_and_save.py
Reproduces the full training pipeline from AOL_Machine_Learning.ipynb,
trains CatBoostRegressor, and saves all artifacts to models/.

Run once before starting the Streamlit app:
    python train_and_save.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "Phone_Addiction.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 1

# ── 1. load data ─────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# ── 2. data cleaning ─────────────────────────────────────────────────────────
print("Cleaning data...")

# drop meaningless columns (notebook cell 35 & 54)
drop_cols = ["Name", "Location", "Unnamed: 0", "ConstantCol", "Apps_Used_Weekly"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# fix Sleep_Hours — strip surrounding quotes (notebook cell 43)
df["Sleep_Hours"] = df["Sleep_Hours"].astype(str).str.strip('"').astype(float)

# normalise Gender (notebook cell 48)
df["Gender"] = (
    df["Gender"]
    .str.strip()
    .str.lower()
    .replace("femle", "female")
    .str.capitalize()
)

# replace "Unknown" in Phone_Usage_Purpose with NaN (notebook cell 49)
df["Phone_Usage_Purpose"] = df["Phone_Usage_Purpose"].replace("Unknown", np.nan)

# cap Age outliers > 150 (notebook cell 61)
cap_age = df.loc[df["Age"] <= 150, "Age"].max()
df.loc[df["Age"] > 150, "Age"] = cap_age

# drop duplicates (notebook cell 40)
df = df.drop_duplicates()

# ── 3. split ──────────────────────────────────────────────────────────────────
print("Splitting data...")
TARGET = "Addiction_Level"
y = df[TARGET]
X = df.drop(columns=[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=284091, stratify=y
)

# ── 4. imputation (fit on X_train) ───────────────────────────────────────────
print("Fitting imputers...")
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols_all = X_train.select_dtypes(exclude=["number"]).columns.tolist()

num_medians = {col: X_train[col].median() for col in num_cols}
cat_modes   = {col: X_train[col].mode()[0] for col in cat_cols_all}

for col, val in num_medians.items():
    X_train[col] = X_train[col].fillna(val)
    X_test[col]  = X_test[col].fillna(val)

for col, val in cat_modes.items():
    X_train[col] = X_train[col].fillna(val)
    X_test[col]  = X_test[col].fillna(val)

# ── 5. drop duplicates from X_train after imputation ─────────────────────────
combined = pd.concat([X_train, y_train], axis=1).drop_duplicates()
X_train  = combined.drop(columns=[TARGET])
y_train  = combined[TARGET]

# ── 6. one-hot encoding ───────────────────────────────────────────────────────
print("Fitting OHE...")
CAT_COLS = ["Gender", "Phone_Usage_Purpose"]
num_cols_after = [c for c in X_train.columns if c not in CAT_COLS]

ohe = OneHotEncoder(
    drop=["Other", "Other"],
    sparse_output=False,
    handle_unknown="ignore",
)
ohe.fit(X_train[CAT_COLS])

def apply_ohe(X, ohe, num_cols_after):
    ohe_df = pd.DataFrame(
        ohe.transform(X[CAT_COLS]),
        columns=ohe.get_feature_names_out(CAT_COLS),
        index=X.index,
    )
    return pd.concat([X[num_cols_after], ohe_df], axis=1)

X_train = apply_ohe(X_train, ohe, num_cols_after)
X_test  = apply_ohe(X_test,  ohe, num_cols_after)

# ── 7. feature engineering ────────────────────────────────────────────────────
print("Engineering features...")

def engineer(x):
    x = x.copy()
    eps = 1e-3
    x["usage_zero_flag"]         = (x["Daily_Usage_Hours"] <= 0).astype(int)
    denom                        = x["Daily_Usage_Hours"].clip(lower=1)
    x["checks_per_hour"]         = x["Phone_Checks_Per_Day"] / denom
    x["apps_per_hour"]           = x["Apps_Used_Daily"] / denom
    x["screen_before_bed_ratio"] = x["Screen_Time_Before_Bed"] / denom
    x["usage_to_sleep_ratio"]    = x["Daily_Usage_Hours"] / (x["Sleep_Hours"] + eps)
    x["late_screen_ratio"]       = x["Screen_Time_Before_Bed"] / (x["Sleep_Hours"] + eps)
    solo   = x["Time_on_Gaming"] + x["Time_on_Social_Media"]
    social = x["Family_Communication"] + x["Social_Interactions"]
    x["social_to_solo_ratio"]    = social / (solo + eps)
    strain = x["Anxiety_Level"] + x["Depression_Level"]
    x["resilience_gap"]          = x["Self_Esteem"] - strain / 2.0
    x["high_gaming_x_sleep"]     = x["Time_on_Gaming"] * x["Sleep_Hours"]
    x["social_media_x_anxiety"]  = x["Time_on_Social_Media"] * x["Anxiety_Level"]
    return x

X_train = engineer(X_train)
X_test  = engineer(X_test)

# ── 8. log transform ──────────────────────────────────────────────────────────
SKEWED = [
    "Age", "checks_per_hour", "apps_per_hour", "screen_before_bed_ratio",
    "usage_to_sleep_ratio", "social_to_solo_ratio", "social_media_x_anxiety",
]

for x in [X_train, X_test]:
    for col in SKEWED:
        if col in x.columns:
            x[col] = np.log1p(x[col].clip(lower=0))

# ── 9. scaling ────────────────────────────────────────────────────────────────
print("Fitting scaler...")
feature_order = X_train.columns.tolist()

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_order,
    index=X_train.index,
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_order,
    index=X_test.index,
)

# ── 10. train CatBoost ────────────────────────────────────────────────────────
print("Training CatBoost...")
cat_params = {
    "iterations":   1000,
    "learning_rate": 0.05,
    "depth":         6,
    "l2_leaf_reg":   3.0,
    "loss_function": "RMSE",
    "random_seed":   SEED,
    "verbose":       100,
}

model = CatBoostRegressor(**cat_params)
model.fit(X_train_scaled, y_train)

# ── 11. evaluate ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)
print(f"\n=== Test Results ===")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ── 12. save artifacts ────────────────────────────────────────────────────────
print("\nSaving artifacts...")

model_path    = os.path.join(MODELS_DIR, "catboost_model.cbm")
scaler_path   = os.path.join(MODELS_DIR, "scaler.pkl")
encoders_path = os.path.join(MODELS_DIR, "encoders.pkl")

model.save_model(model_path)

joblib.dump(scaler, scaler_path)

artifact_bundle = {
    "ohe":           ohe,
    "num_medians":   num_medians,
    "cat_modes":     cat_modes,
    "feature_order": feature_order,
}
joblib.dump(artifact_bundle, encoders_path)

print(f"  Saved: {model_path}")
print(f"  Saved: {scaler_path}")
print(f"  Saved: {encoders_path}")
print("\nDone! You can now run: streamlit run app.py")
