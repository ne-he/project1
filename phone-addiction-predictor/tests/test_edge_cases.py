"""
tests/test_edge_cases.py
Edge case tests for the preprocessing pipeline.
Run with: python phone-addiction-predictor/tests/test_edge_cases.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
from catboost import CatBoostRegressor
from src.preprocessing import preprocess_pipeline
from src.model import predict

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

model = CatBoostRegressor()
model.load_model(os.path.join(MODELS_DIR, "catboost_model.cbm"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
bundle = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
ohe, num_medians, cat_modes, feature_order = (
    bundle["ohe"], bundle["num_medians"], bundle["cat_modes"], bundle["feature_order"]
)


def run(input_dict):
    processed = preprocess_pipeline(
        input_dict, ohe, scaler, num_medians, cat_modes, feature_order
    )
    return predict(model, processed)


def base(**overrides):
    d = {
        "Age": 18.0, "Gender": "Male", "Daily_Usage_Hours": 5.0,
        "Sleep_Hours": 7.0, "Interllectual_Performance": 70,
        "Social_Interactions": 5, "Exercise_Hours": 1.0,
        "Screen_Time_Before_Bed": 1.0, "Phone_Checks_Per_Day": 50,
        "Anxiety_Level": 5, "Depression_Level": 5, "Self_Esteem": 5,
        "Apps_Used_Daily": 10, "Time_on_Social_Media": 2.0,
        "Time_on_Gaming": 1.0, "Time_on_Education": 1.0,
        "Phone_Usage_Purpose": "Browsing", "Family_Communication": 5,
        "Weekend_Usage_Hours": 6.0,
    }
    d.update(overrides)
    return d


# ── edge case 1: Daily_Usage_Hours = 0 (usage_zero_flag=1, denom_usage=1) ────
def test_zero_daily_usage():
    pred = run(base(Daily_Usage_Hours=0.0))
    assert 1.0 <= pred <= 10.0
    print(f"[PASS] zero daily usage: pred = {pred:.2f}")


# ── edge case 2: Sleep_Hours very small (eps prevents division by zero) ───────
def test_tiny_sleep_hours():
    pred = run(base(Sleep_Hours=0.01))
    assert 1.0 <= pred <= 10.0
    print(f"[PASS] tiny sleep hours (0.01): pred = {pred:.2f}")


# ── edge case 3: Sleep_Hours = 0 ─────────────────────────────────────────────
def test_zero_sleep_hours():
    pred = run(base(Sleep_Hours=0.0))
    assert 1.0 <= pred <= 10.0
    print(f"[PASS] zero sleep hours: pred = {pred:.2f}")


# ── edge case 4: all minimum values ──────────────────────────────────────────
def test_all_min_values():
    pred = run(base(
        Age=1.0, Daily_Usage_Hours=0.0, Sleep_Hours=0.0,
        Interllectual_Performance=0, Social_Interactions=0,
        Exercise_Hours=0.0, Screen_Time_Before_Bed=0.0,
        Phone_Checks_Per_Day=0, Anxiety_Level=0, Depression_Level=0,
        Self_Esteem=0, Apps_Used_Daily=0, Time_on_Social_Media=0.0,
        Time_on_Gaming=0.0, Time_on_Education=0.0,
        Family_Communication=0, Weekend_Usage_Hours=0.0,
    ))
    assert 1.0 <= pred <= 10.0
    print(f"[PASS] all min values: pred = {pred:.2f}")


# ── edge case 5: all maximum values ──────────────────────────────────────────
def test_all_max_values():
    pred = run(base(
        Age=100.0, Daily_Usage_Hours=24.0, Sleep_Hours=24.0,
        Interllectual_Performance=100, Social_Interactions=20,
        Exercise_Hours=24.0, Screen_Time_Before_Bed=24.0,
        Phone_Checks_Per_Day=500, Anxiety_Level=10, Depression_Level=10,
        Self_Esteem=10, Apps_Used_Daily=100, Time_on_Social_Media=24.0,
        Time_on_Gaming=24.0, Time_on_Education=24.0,
        Family_Communication=20, Weekend_Usage_Hours=24.0,
    ))
    assert 1.0 <= pred <= 10.0
    print(f"[PASS] all max values: pred = {pred:.2f}")


# ── edge case 6: Phone_Usage_Purpose = Other (dropped by OHE) ────────────────
def test_purpose_other():
    pred = run(base(Phone_Usage_Purpose="Other"))
    assert 1.0 <= pred <= 10.0
    print(f"[PASS] Phone_Usage_Purpose=Other: pred = {pred:.2f}")


# ── edge case 7: Gender = Other (dropped by OHE) ─────────────────────────────
def test_gender_other():
    pred = run(base(Gender="Other"))
    assert 1.0 <= pred <= 10.0
    print(f"[PASS] Gender=Other: pred = {pred:.2f}")


# ── edge case 8: Sleep_Hours with quoted string (clean_sleep_hours) ───────────
def test_quoted_sleep_hours():
    pred = run(base(Sleep_Hours='"7.5"'))
    assert 1.0 <= pred <= 10.0
    print(f"[PASS] quoted Sleep_Hours '\"7.5\"': pred = {pred:.2f}")


# ── edge case 9: high addiction profile ───────────────────────────────────────
def test_high_addiction_profile():
    pred = run(base(
        Daily_Usage_Hours=12.0, Phone_Checks_Per_Day=200,
        Anxiety_Level=9, Depression_Level=9, Self_Esteem=1,
        Time_on_Social_Media=8.0, Time_on_Gaming=6.0,
        Sleep_Hours=4.0, Exercise_Hours=0.0,
    ))
    assert 1.0 <= pred <= 10.0
    print(f"[PASS] high addiction profile: pred = {pred:.2f}")


# ── edge case 10: low addiction profile ───────────────────────────────────────
def test_low_addiction_profile():
    pred = run(base(
        Daily_Usage_Hours=1.0, Phone_Checks_Per_Day=5,
        Anxiety_Level=1, Depression_Level=1, Self_Esteem=9,
        Time_on_Social_Media=0.2, Time_on_Gaming=0.1,
        Sleep_Hours=8.0, Exercise_Hours=2.0,
        Phone_Usage_Purpose="Education",
    ))
    assert 1.0 <= pred <= 10.0
    print(f"[PASS] low addiction profile: pred = {pred:.2f}")


if __name__ == "__main__":
    print("=" * 50)
    print("Running edge case tests...")
    print("=" * 50)
    test_zero_daily_usage()
    test_tiny_sleep_hours()
    test_zero_sleep_hours()
    test_all_min_values()
    test_all_max_values()
    test_purpose_other()
    test_gender_other()
    test_quoted_sleep_hours()
    test_high_addiction_profile()
    test_low_addiction_profile()
    print("=" * 50)
    print("All edge case tests passed!")
