import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# =====================================================
# SETTINGS
# =====================================================

EXCEL_FILE = r"C:\Users\DELL\Downloads\kiln_patch_10day_avg (6).xlsx"

WINDOW = 20

# =====================================================
# LOAD
# =====================================================

df = pd.read_excel(EXCEL_FILE)

temps = df.drop(
    columns=["Patch ID", "90-Day Slope"]
)

print("Shape:", temps.shape)

# =====================================================
# GLOBAL SCALER
# =====================================================

all_values = temps.values.reshape(-1, 1)

scaler = MinMaxScaler()

scaler.fit(all_values)

# =====================================================
# BUILD TRAINING DATA
# =====================================================

X = []
y = []

for _, row in temps.iterrows():

    series = row.values.astype(float)

    scaled_series = scaler.transform(
        series.reshape(-1,1)
    ).flatten()

    for i in range(WINDOW, len(scaled_series)):

        X.append(
            scaled_series[i-WINDOW:i]
        )

        y.append(
            scaled_series[i]
        )

X = np.array(X)
y = np.array(y)

X = X.reshape(
    (X.shape[0], X.shape[1], 1)
)

print("Training Samples:", X.shape)

# =====================================================
# MODEL
# =====================================================

model = Sequential()

model.add(
    Input(shape=(WINDOW,1))
)

model.add(
    LSTM(
        64,
        return_sequences=True
    )
)

model.add(
    LSTM(
        32
    )
)

model.add(
    Dense(1)
)

model.compile(
    optimizer="adam",
    loss="mse"
)

# =====================================================
# TRAIN
# =====================================================

model.fit(
    X,
    y,
    epochs=15,
    batch_size=128,
    verbose=1
)

# =====================================================
# SAVE
# =====================================================

os.makedirs("model", exist_ok=True)
os.makedirs("scaler", exist_ok=True)

model.save(
    "model/lstm_patch.h5"
)

joblib.dump(
    scaler,
    "scaler/scaler.pkl"
)

print("\nDONE")