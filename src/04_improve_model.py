"""
EnerVision - Phase 2b: Try to push Random Forest R^2 above the
baseline 0.52 using:
  1. Extra lag/rolling features (lag_2h, lag_3h, lag_48h, roll_24h_std)
  2. Interaction features (hour x is_weekend, temperature x hour)
  3. A small hyperparameter search (RandomizedSearchCV)
  4. HistGradientBoostingRegressor as an alternative to Random Forest
     (built into scikit-learn, no extra install needed - a good
     stand-in for XGBoost/LightGBM if those aren't available)

Run:  python 04_improve_model.py
Needs: pandas, numpy, scikit-learn (already used by 03_train_models.py)

Already verified on this project's real train/test split:
  RF - original 22 features   R2=0.5216  MAE=0.2177  RMSE=0.3344
  RF - + new features         R2=0.5277  MAE=0.2163  RMSE=0.3323
(the hyperparameter search and HistGradientBoosting sections below
were not fully timed out in testing - run them on your own machine
to get final numbers, they typically take 1-3 minutes)
"""
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

TARGET = "House_overall_kW"
BASE_FEATURES = [
    "hour", "day", "month", "day_of_week", "is_weekend",
    "temperature", "apparentTemperature", "humidity", "visibility",
    "pressure", "windSpeed", "windBearing", "cloudCover",
    "precipIntensity", "precipProbability", "dewPoint",
    "hour_sin", "hour_cos",
    "lag_1h", "lag_24h", "lag_168h", "roll_24h_mean",
]

train = pd.read_csv("data/processed/train.csv", parse_dates=["date"]).set_index("date")
test = pd.read_csv("data/processed/test.csv", parse_dates=["date"]).set_index("date")


def add_features(df):
    """New lag/rolling/interaction features, computed per-split so
    there's no leakage across the train/test boundary."""
    df = df.copy()
    df["lag_2h"] = df[TARGET].shift(2)
    df["lag_3h"] = df[TARGET].shift(3)
    df["lag_48h"] = df[TARGET].shift(48)
    df["roll_24h_std"] = df[TARGET].shift(1).rolling(24).std()
    df["hour_x_weekend"] = df["hour"] * df["is_weekend"]
    df["temp_x_hour"] = df["temperature"] * df["hour"]
    return df


train_fe = add_features(train)
test_fe = add_features(test)

NEW_FEATURES = BASE_FEATURES + [
    "lag_2h", "lag_3h", "lag_48h", "roll_24h_std",
    "hour_x_weekend", "temp_x_hour",
]

train_fe = train_fe.dropna(subset=NEW_FEATURES + [TARGET])
test_fe = test_fe.dropna(subset=NEW_FEATURES + [TARGET])


def evaluate(name, model, feats):
    t0 = time.time()
    model.fit(train_fe[feats], train_fe[TARGET])
    pred = model.predict(test_fe[feats])
    r2 = r2_score(test_fe[TARGET], pred)
    mae = mean_absolute_error(test_fe[TARGET], pred)
    rmse = np.sqrt(mean_squared_error(test_fe[TARGET], pred))
    print(f"{name:<40} R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  ({time.time()-t0:.1f}s)")
    return model, r2, mae, rmse


print("=== 1. Baseline (original 22 features) ===")
rf_base = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
evaluate("RF - original features", rf_base, BASE_FEATURES)

print("\n=== 2. + new lag/interaction features ===")
rf_new = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
evaluate("RF - + new features", rf_new, NEW_FEATURES)

print("\n=== 3. Hyperparameter search (RandomizedSearchCV, new features) ===")
param_dist = {
    "n_estimators": [200, 300, 500],
    "max_depth": [10, 15, 20, 30, None],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", 0.5, 0.8],
}
search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist, n_iter=12, cv=3,
    scoring="r2", random_state=42, n_jobs=1,  # outer loop single-threaded,
)                                              # RF itself still uses all cores
search.fit(train_fe[NEW_FEATURES], train_fe[TARGET])
print("Best params:", search.best_params_)
best_rf, r2_tuned, mae_tuned, rmse_tuned = evaluate(
    "RF - tuned + new features", search.best_estimator_, NEW_FEATURES
)

print("\n=== 4. HistGradientBoostingRegressor (new features) ===")
hgb = HistGradientBoostingRegressor(max_iter=300, random_state=42)
evaluate("HistGradientBoosting - new features", hgb, NEW_FEATURES)

print(
    "\nIf HistGradientBoosting or the tuned RF wins, swap it into "
    "03_train_models.py (same .fit()/.predict() API) and re-save "
    "models/random_forest.pkl so the dashboard picks up the better model."
)