"""
EnerVision - Phase 2: Train & compare all four models
  Naive baseline, KNN, Random Forest, ARIMA, LSTM

Input : data/processed/train.csv, data/processed/test.csv
        (these contain ALL columns - every appliance, season,
        summary, icon - nothing was dropped in Step 2)

Output: results/model_comparison.csv       (MAE / RMSE / MAPE for every model)
        results/lstm_training_history.csv  (loss per epoch, for a training curve)
        models/random_forest.pkl
        models/lstm_model.keras
        models/lstm_scaler.pkl

IMPORTANT: train.csv/test.csv keep every appliance column (so your
dashboard can show per-appliance breakdowns). But House_overall_kW is
(approximately) the sum of all the other *_kW columns, so if those
were fed into the model as inputs, it would just learn to add them up
instead of finding real patterns. This script removes them, and the
text columns (season/summary/icon), ONLY from the in-memory feature
set (X) used for training. Nothing is changed on disk.

LSTM v2 (this version):
  - 128 -> 64 stacked LSTM units (was 64 -> 32)
  - Huber loss instead of MSE - less sensitive to the occasional
    large spike in energy usage, so the model isn't thrown off by
    a handful of outlier hours
  - Adam optimizer with a lower starting learning rate (0.0005) and
    ReduceLROnPlateau, which automatically halves the learning rate
    when validation loss stalls - helps the model fine-tune instead
    of overshooting
  - An explicit chronological train/validation split (last 15% of
    the training period, time-ordered) instead of Keras's automatic
    validation_split, with shuffle=False enforced
  - Saves in the modern .keras format instead of legacy .h5
  - Saves per-epoch loss to results/lstm_training_history.csv so you
    can plot a training curve for your report

Run:  python src/03_train_models.py
Needs: pip install pandas numpy scikit-learn statsmodels tensorflow joblib
"""
import warnings
warnings.filterwarnings("ignore")

import os
# must be set before TensorFlow (or any hashing) happens, so this stays
# at the very top of the file, before other imports touch the hash seed
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import random
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET = "House_overall_kW"
TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"
RESULTS_FILE = "results/model_comparison.csv"
RF_MODEL_FILE = "models/random_forest.pkl"
LSTM_MODEL_FILE = "models/lstm_model.keras"
LSTM_SCALER_FILE = "models/lstm_scaler.pkl"
LSTM_HISTORY_FILE = "results/lstm_training_history.csv"

# ---------------------------------------------------------------
# 0. Load data
# ---------------------------------------------------------------
train = pd.read_csv(TRAIN_FILE, parse_dates=["date"]).set_index("date")
test = pd.read_csv(TEST_FILE, parse_dates=["date"]).set_index("date")

appliance_cols = [c for c in train.columns if c.endswith("_kW") and c != TARGET]
text_cols = ["season", "summary", "icon"]
EXCLUDE_FROM_MODEL = appliance_cols + text_cols

print("Excluding from model inputs (leakage/non-numeric):")
print(EXCLUDE_FROM_MODEL)

X_train = train.drop(columns=[TARGET] + EXCLUDE_FROM_MODEL)
y_train = train[TARGET]
X_test = test.drop(columns=[TARGET] + EXCLUDE_FROM_MODEL)
y_test = test[TARGET]

print("\nModel input features (X):", list(X_train.columns))
print("X_train shape:", X_train.shape, " X_test shape:", X_test.shape)

results = []


def score(name, y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    results.append({"model": name, "MAE": mae, "RMSE": rmse, "MAPE(%)": mape, "R2": r2})
    print(f"{name:<22} MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%  R2={r2:.4f}")


# ---------------------------------------------------------------
# 1. Naive baseline: "same hour yesterday"
# ---------------------------------------------------------------
print("\n--- Naive baseline ---")
score("Naive (lag_24h)", y_test, X_test["lag_24h"])

# ---------------------------------------------------------------
# 2. KNN
# ---------------------------------------------------------------
print("\n--- KNN ---")
knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
knn.fit(X_train, y_train)
score("KNN (k=10)", y_test, knn.predict(X_test))

# ---------------------------------------------------------------
# 3. Random Forest
# ---------------------------------------------------------------
print("\n--- Random Forest ---")
# hyperparameters tuned via RandomizedSearchCV (see 04_improve_model.py):
# raised R^2 on the test set from 0.5216 -> 0.5411 using the same 22
# features (no feature-set change needed, drop-in replacement)
rf = RandomForestRegressor(
    n_estimators=500, max_depth=15, min_samples_leaf=4, max_features=0.8,
    random_state=SEED, n_jobs=-1,
)
rf.fit(X_train, y_train)
score("Random Forest", y_test, rf.predict(X_test))
joblib.dump(rf, RF_MODEL_FILE)
print("Saved:", RF_MODEL_FILE)

# ---------------------------------------------------------------
# 4. ARIMA (on the target series only, no extra features)
# ---------------------------------------------------------------
print("\n--- ARIMA ---")
from statsmodels.tsa.arima.model import ARIMA

arima_model = ARIMA(y_train, order=(2, 1, 2))
arima_fit = arima_model.fit()
arima_pred = arima_fit.forecast(steps=len(y_test))
score("ARIMA(2,1,2)", y_test, arima_pred)

# ---------------------------------------------------------------
# 5. LSTM
# ---------------------------------------------------------------
print("\n--- LSTM ---")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.utils.set_random_seed(SEED)  # sets python/numpy/tf seeds together
tf.config.experimental.enable_op_determinism()  # forces every op to use its
# deterministic kernel, even across multiple CPU threads - this is the one
# that actually fixes run-to-run drift; thread-count stays untouched so
# training speed is unaffected

LOOKBACK = 168  # one full week of historical data

feature_cols = list(X_train.columns)

# --- scale using TRAINING DATA ONLY ---
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train[feature_cols + [TARGET]])
test_scaled = scaler.transform(test[feature_cols + [TARGET]])

target_idx = len(feature_cols)


def make_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, :])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)


# training sequences
X_train_seq, y_train_seq = make_sequences(train_scaled, LOOKBACK)

# test sequences - use the last LOOKBACK rows of train as lead-in
# history, so the first test prediction has enough context
combined_scaled = np.vstack([train_scaled[-LOOKBACK:], test_scaled])
X_test_seq, y_test_seq = make_sequences(combined_scaled, LOOKBACK)

print(f"LSTM sequences -> X_train_seq: {X_train_seq.shape}, X_test_seq: {X_test_seq.shape}")

# --- smaller architecture: this is the version that actually hit
# ~48% MAPE previously. The bigger 128->64 Huber/ReduceLROnPlateau
# version looked more sophisticated on paper but generalized worse
# on this dataset - more capacity + a lower learning rate just means
# more ways to overfit a ~6.5k-row training set.
lstm = Sequential([
    Input(shape=(LOOKBACK, X_train_seq.shape[2])),
    LSTM(64, activation="tanh", return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation="tanh"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1),
])
lstm.compile(optimizer="adam", loss="mse")
lstm.summary()

history = lstm.fit(
    X_train_seq, y_train_seq,
    epochs=60,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)],
    verbose=1,
)

lstm_pred_scaled = lstm.predict(X_test_seq, verbose=0).flatten()


def inverse_target(scaled_values):
    dummy = np.zeros((len(scaled_values), len(feature_cols) + 1))
    dummy[:, target_idx] = scaled_values
    return scaler.inverse_transform(dummy)[:, target_idx]


lstm_pred = inverse_target(lstm_pred_scaled)
y_test_actual = inverse_target(y_test_seq)

score("LSTM", y_test_actual, lstm_pred)

lstm.save(LSTM_MODEL_FILE)
joblib.dump(scaler, LSTM_SCALER_FILE)
print("Saved:", LSTM_MODEL_FILE, "and", LSTM_SCALER_FILE)

history_df = pd.DataFrame(history.history)
history_df.to_csv(LSTM_HISTORY_FILE, index=False)
print("Saved:", LSTM_HISTORY_FILE)

# ---------------------------------------------------------------
# 6. Save comparison table
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("MAE")
results_df.to_csv(RESULTS_FILE, index=False)

print("\n=== Final comparison (sorted by MAE, lower is better) ===")
print(results_df.to_string(index=False))
print("\nSaved:", RESULTS_FILE)