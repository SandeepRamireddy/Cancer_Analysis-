from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
from model_inferencing.inference import (
    load_model,
    load_scaler,
    predict_single,
    FEATURE_COLUMNS,
)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = ROOT_DIR / "artifacts" / "best_model.pkl"   # <-- changed
SCALER_PATH = ROOT_DIR / "artifacts" / "scaler.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        app.state.model = load_model(MODEL_PATH)
        app.state.scaler = load_scaler(SCALER_PATH)
        app.state.load_error = None
        print("Model and scaler loaded successfully.")
    except Exception as e:
        app.state.model = None
        app.state.scaler = None
        app.state.load_error = str(e)
        print(f"Failed to load model/scaler: {e}")

    yield  # 👈 Application runs here

    # Shutdown
    print("Shutting down Cancer Analysis API...")


app = FastAPI(title="Cancer Analysis Inference API", lifespan=lifespan)

# Serve static files under /static
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


class SingleInput(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

@app.get("/")
def root():
    return {"message": "Cancer Analysis API is running.."}


@app.post("/predict")
def predict(input_data: SingleInput):   # 👈 use Pydantic model for validation
    try:
        # Access model and scaler from app state
        if app.state.model is None or app.state.scaler is None:
            raise HTTPException(status_code=500, detail=f"Model/Scaler not loaded: {app.state.load_error}")

        model = app.state.model
        scaler = app.state.scaler

        # Ensure feature order matches training
        features = [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
            "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
            "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
            "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
        ]

        # Convert request to numpy array in correct order
        input_dict = input_data.dict()   # 👈 convert Pydantic model to dict
        values = [input_dict[feature] for feature in features]
        values = np.array(values).reshape(1, -1)

        # Scale input
        scaled = scaler.transform(values)

        # Predict
        prediction = model.predict(scaled)[0]
        result = "Malignant" if prediction == 1 else "Benign"

        return {"prediction": int(prediction), "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
