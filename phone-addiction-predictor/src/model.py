"""
src/model.py
Artifact loading and model inference.
"""

import os
import joblib
import streamlit as st
from catboost import CatBoostRegressor

# Path to models/ relative to this file's parent (phone-addiction-predictor/)
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


# ---------------------------------------------------------------------------
# 3.2  load_artifacts
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    """Load all ML artifacts once at startup and cache them.

    Returns:
        model        : Fitted CatBoostRegressor
        scaler       : Fitted StandardScaler
        ohe          : Fitted OneHotEncoder
        num_medians  : dict {col: median} for numerical imputation
        cat_modes    : dict {col: mode}   for categorical imputation
        feature_order: list[str] — column order expected by the model
    """
    model_path    = os.path.join(_MODELS_DIR, "catboost_model.cbm")
    scaler_path   = os.path.join(_MODELS_DIR, "scaler.pkl")
    encoders_path = os.path.join(_MODELS_DIR, "encoders.pkl")

    for path in (model_path, scaler_path, encoders_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                "Run `python train_and_save.py` first to generate model artifacts."
            )

    model = CatBoostRegressor()
    model.load_model(model_path)

    scaler = joblib.load(scaler_path)

    bundle = joblib.load(encoders_path)
    ohe           = bundle["ohe"]
    num_medians   = bundle["num_medians"]
    cat_modes     = bundle["cat_modes"]
    feature_order = bundle["feature_order"]

    return model, scaler, ohe, num_medians, cat_modes, feature_order
