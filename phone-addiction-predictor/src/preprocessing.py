"""
src/preprocessing.py
Preprocessing pipeline — identik dengan notebook AOL_Machine_Learning.ipynb
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ---------------------------------------------------------------------------
# 2.1  clean_sleep_hours
# ---------------------------------------------------------------------------

def clean_sleep_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding quotes from Sleep_Hours and convert to float.

    Notebook cell 43:
        df["Sleep_Hours"] = df["Sleep_Hours"].astype(str).str.strip('"').astype(float)
    """
    df = df.copy()
    df["Sleep_Hours"] = df["Sleep_Hours"].astype(str).str.strip('"').astype(float)
    return df


# ---------------------------------------------------------------------------
# 2.2  handle_missing_values
# ---------------------------------------------------------------------------

def handle_missing_values(
    df: pd.DataFrame,
    num_medians: dict,
    cat_modes: dict,
) -> pd.DataFrame:
    """Impute missing values using statistics computed from the training set.

    Notebook cells 74 & 80:
        - Numerical: fillna(median from X_train)
        - Categorical: fillna(mode from X_train)

    Args:
        df: Input DataFrame.
        num_medians: {column_name: median_value} from training set.
        cat_modes:   {column_name: mode_value}   from training set.
    """
    df = df.copy()
    for col, val in num_medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col, val in cat_modes.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df


# ---------------------------------------------------------------------------
# 2.3  encode_categorical
# ---------------------------------------------------------------------------

CAT_COLS = ["Gender", "Phone_Usage_Purpose"]


def encode_categorical(df: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame:
    """Apply fitted OneHotEncoder to Gender and Phone_Usage_Purpose.

    Notebook cell 94:
        ohe = OneHotEncoder(drop=["Other", "Other"], sparse_output=False,
                            handle_unknown="ignore")
        ohe.fit(X_train[cat_cols])

    The encoded columns replace the original categorical columns.
    """
    df = df.copy()
    encoded = ohe.transform(df[CAT_COLS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(CAT_COLS),
        index=df.index,
    )
    df = df.drop(columns=CAT_COLS)
    return pd.concat([df, encoded_df], axis=1)


# ---------------------------------------------------------------------------
# 2.4  engineer_features  (+ log_transform bundled as 2.5)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create 10 derived features exactly as in notebook cell 96.

    Features created:
        usage_zero_flag, checks_per_hour, apps_per_hour,
        screen_before_bed_ratio, usage_to_sleep_ratio, late_screen_ratio,
        social_to_solo_ratio, resilience_gap,
        high_gaming_x_sleep, social_media_x_anxiety
    """
    x = df.copy()
    eps = 1e-3  # small constant to prevent division by zero

    # flag rows where phone is barely used
    x["usage_zero_flag"] = (x["Daily_Usage_Hours"] <= 0).astype(int)

    # clip to 1 so we never divide by zero even when Daily_Usage_Hours=0
    denom_usage = x["Daily_Usage_Hours"].clip(lower=1)
    x["checks_per_hour"] = x["Phone_Checks_Per_Day"] / denom_usage
    x["apps_per_hour"] = x["Apps_Used_Daily"] / denom_usage
    x["screen_before_bed_ratio"] = x["Screen_Time_Before_Bed"] / denom_usage

    # sleep-based ratios — eps prevents inf when Sleep_Hours=0
    x["usage_to_sleep_ratio"] = x["Daily_Usage_Hours"] / (x["Sleep_Hours"] + eps)
    x["late_screen_ratio"] = x["Screen_Time_Before_Bed"] / (x["Sleep_Hours"] + eps)

    # social vs solo usage balance
    solo_usage = x["Time_on_Gaming"] + x["Time_on_Social_Media"]
    social_use = x["Family_Communication"] + x["Social_Interactions"]
    x["social_to_solo_ratio"] = social_use / (solo_usage + eps)

    # mental health composite: positive self-esteem vs negative strain
    mental_strain = x["Anxiety_Level"] + x["Depression_Level"]
    x["resilience_gap"] = x["Self_Esteem"] - mental_strain / 2.0

    # interaction terms
    x["high_gaming_x_sleep"] = x["Time_on_Gaming"] * x["Sleep_Hours"]
    x["social_media_x_anxiety"] = x["Time_on_Social_Media"] * x["Anxiety_Level"]

    return x


# ---------------------------------------------------------------------------
# 2.5  log_transform
# ---------------------------------------------------------------------------

SKEWED_COLS = [
    "Age",
    "checks_per_hour",
    "apps_per_hour",
    "screen_before_bed_ratio",
    "usage_to_sleep_ratio",
    "social_to_solo_ratio",
    "social_media_x_anxiety",
]


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply np.log1p to skewed columns (notebook cell 102).

    Columns transformed: Age, checks_per_hour, apps_per_hour,
    screen_before_bed_ratio, usage_to_sleep_ratio, social_to_solo_ratio,
    social_media_x_anxiety.
    """
    x = df.copy()
    for col in SKEWED_COLS:
        if col in x.columns:
            x[col] = np.log1p(x[col].clip(lower=0))
    return x


# ---------------------------------------------------------------------------
# 2.6  scale_features
# ---------------------------------------------------------------------------

def scale_features(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a pre-fitted StandardScaler to the DataFrame (notebook cell 106).

    Returns a DataFrame with the same columns and index as the input.
    """
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)


# ---------------------------------------------------------------------------
# 2.7  preprocess_pipeline  — main entry point for inference
# ---------------------------------------------------------------------------

def preprocess_pipeline(
    input_dict: dict,
    ohe: OneHotEncoder,
    scaler: StandardScaler,
    num_medians: dict,
    cat_modes: dict,
    feature_order: list,
) -> pd.DataFrame:
    """Transform a raw user-input dict into a scaled DataFrame ready for inference.

    Steps (identical to notebook training pipeline):
        1. Build single-row DataFrame from input_dict
        2. clean_sleep_hours()
        3. Replace "Unknown" in Phone_Usage_Purpose with NaN
        4. handle_missing_values()
        5. encode_categorical()
        6. engineer_features()
        7. log_transform()
        8. Reorder columns to match training feature_order
        9. scale_features()

    Args:
        input_dict:    Raw values from the Streamlit form (19 features).
        ohe:           Fitted OneHotEncoder (loaded from encoders.pkl).
        scaler:        Fitted StandardScaler (loaded from scaler.pkl).
        num_medians:   {col: median} computed on X_train.
        cat_modes:     {col: mode}   computed on X_train.
        feature_order: Ordered list of column names after all transformations.

    Returns:
        pd.DataFrame of shape (1, len(feature_order)), scaled and ready for model.predict().
    """
    df = pd.DataFrame([input_dict])

    df = clean_sleep_hours(df)

    df["Phone_Usage_Purpose"] = df["Phone_Usage_Purpose"].replace("Unknown", np.nan)

    df = handle_missing_values(df, num_medians, cat_modes)
    df = encode_categorical(df, ohe)
    df = engineer_features(df)
    df = log_transform(df)

    # Ensure column order matches training exactly
    df = df[feature_order]

    df = scale_features(df, scaler)
    return df
