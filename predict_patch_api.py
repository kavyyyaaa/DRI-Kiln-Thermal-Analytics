from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# ==========================
# LOAD MODEL & SCALER
# ==========================

model = load_model(
    "model/lstm_patch.h5",
    compile=False
)
scaler = joblib.load(
    "scaler/scaler.pkl"
)

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_excel(
    r"C:\Users\DELL\Downloads\kiln_patch_10day_avg (6).xlsx"
)

# Remove unwanted columns
drop_cols = ["Patch ID"]

for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

if "90-Day Slope" in df.columns:
    df = df.drop(columns=["90-Day Slope"])

# Only temperature columns remain
temp_data = df.values

# ==========================
# FAST API
# ==========================

app = FastAPI()


class PatchRequest(BaseModel):
    patch_id: int


@app.post("/predict-patch")
def predict_patch(request: PatchRequest):

    patch_id = request.patch_id

    if patch_id >= len(temp_data):
        return {
            "error": f"Patch {patch_id} not found"
        }

    # ==========================
    # GET PATCH HISTORY
    # ==========================

    patch_series = temp_data[patch_id]

    current_temp = float(patch_series[-1])

    # ==========================
    # LAST 20 DAYS
    # ==========================

    last_20 = patch_series[-20:]

    predictions = []

    sequence = last_20.copy()

    # ==========================
    # FORECAST 10 DAYS
    # ==========================

    for _ in range(10):

        scaled_seq = scaler.transform(
            sequence.reshape(-1,1)
        )

        X = scaled_seq.reshape(1, 20, 1)

        pred_scaled = model.predict(X, verbose=0)

        pred_temp = scaler.inverse_transform(
            [[pred_scaled[0][0]]]
        )[0][0]

        predictions.append(round(float(pred_temp), 2))

        sequence = np.append(sequence[1:], pred_temp)

    future_temp = predictions[-1]

    # ==========================
    # REAL RISK LOGIC
    # Operator Normal Range:
    # 225°C - 275°C
    # ==========================

    if future_temp < 225:
        risk_level = "LOW"

    elif future_temp < 275:
        risk_level = "NORMAL"

    elif future_temp < 325:
        risk_level = "WARNING"

    elif future_temp < 400:
        risk_level = "HIGH"

    else:
        risk_level = "CRITICAL"

    return {
        "patch_id": patch_id,
        "current_temperature": round(current_temp, 2),
        "future_temperature": round(future_temp, 2),
        "predictions": predictions,
        "risk_level": risk_level
    }