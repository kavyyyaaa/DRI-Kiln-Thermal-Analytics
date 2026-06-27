import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# =====================================================
# LOAD MODEL
# =====================================================

model = load_model(
    "model/lstm_patch.h5",
    compile=False
)

scaler = joblib.load(
    "scaler/scaler.pkl"
)

# =====================================================
# LOAD DATA
# =====================================================

EXCEL_FILE = r"C:\Users\DELL\Downloads\kiln_patch_10day_avg (6).xlsx"

df = pd.read_excel(EXCEL_FILE)

temps = df.drop(
    columns=["Patch ID", "90-Day Slope"]
)

# =====================================================
# PICK PATCH
# =====================================================

PATCH_INDEX = 0

row = temps.iloc[PATCH_INDEX].values.astype(float)

patch_id = df.iloc[PATCH_INDEX]["Patch ID"]

print(f"\nPatch ID: {patch_id}")

# =====================================================
# LAST 20 DAYS
# =====================================================

current = row[-20:]

print(
    f"\nCurrent Temperature: {current[-1]:.2f}°C"
)

predictions = []

# =====================================================
# FORECAST 10 DAYS
# =====================================================

for day in range(10):

    scaled_window = scaler.transform(
        current.reshape(-1,1)
    ).flatten()

    X = scaled_window.reshape(
        1,
        20,
        1
    )

    pred_scaled = model.predict(
        X,
        verbose=0
    )[0][0]

    pred_real = scaler.inverse_transform(
        [[pred_scaled]]
    )[0][0]

    # Thermal inertia smoothing
    if predictions:

        alpha = 0.20

        pred_real = (
            predictions[-1] * (1-alpha)
            + pred_real * alpha
        )

    predictions.append(
        float(pred_real)
    )

    current = np.append(
        current[1:],
        pred_real
    )

# =====================================================
# PRINT
# =====================================================

print("\n===== 10 DAY FORECAST =====")

for i,p in enumerate(predictions,1):

    print(
        f"Day {i}: {p:.2f}°C"
    )

print("===========================")