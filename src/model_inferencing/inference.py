# src/model_training/inference.py
import joblib
from pathlib import Path
import pandas as pd
from typing import Any, Dict

DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")
DEFAULT_SCALER_PATH = Path("artifacts/scaler.pkl")

FEATURE_COLUMNS = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    return joblib.load(path)

def load_scaler(path: str | Path = DEFAULT_SCALER_PATH):
    return joblib.load(path)

def preprocess(features: Dict[str, Any], scaler):
    # Ensure all features are in correct order
    df = pd.DataFrame([[features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    return scaler.transform(df)

def predict_single(model: Any, scaler: Any, features: Dict[str, Any]) -> Dict[str, Any]:
    X_scaled = preprocess(features, scaler)
    preds = model.predict(X_scaled)
    result = {"prediction": int(preds[0])}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled).tolist()
        result["probabilities"] = proba[0]
    return result

def predict_batch(model: Any, scaler: Any, input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    # Select and scale only the known feature columns
    X = df[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    out = df.copy()
    out["prediction"] = preds
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        for i in range(proba.shape[1]):
            out[f"proba_{i}"] = proba[:, i]
    out.to_csv(output_csv, index=False)
    return output_csv
