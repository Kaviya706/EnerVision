"""
EnerVision - AI-Powered Energy Management Dashboard
(Streamlit)

What it does:
  🏠 Home              - KPI cards + usage trend (day/week/month)
  📊 Energy Analysis   - hourly/daily/weekly/monthly breakdowns,
                         peak periods, weekday vs weekend
  🔌 Appliance Analysis- top energy-consuming appliances (bar + pie)
  🔮 Energy Prediction - choose a model (Random Forest / KNN / ARIMA /
                         LSTM) and a horizon (next hour / 24h / 7 days),
                         see a backtest (actual vs predicted) plus a
                         forward forecast
  🏆 Model Comparison  - MAE / RMSE / MAPE table + bar chart, best model
  📁 Dataset           - upload your own CSV, auto data-quality summary
  ✍️ Manual Prediction  - type in conditions, get one instant prediction
  🚨 Alerts & Tips     - abnormal-consumption alert + saving recommendations
  ℹ️ About              - project info

Run:
  streamlit run app/streamlit_app.py

Needs: pip install streamlit pandas numpy scikit-learn joblib plotly statsmodels
       (tensorflow only if you want the LSTM option - the app runs
       fine without it, it just hides that option)
"""
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
st.set_page_config(page_title="EnerVision", page_icon="⚡", layout="wide")

TARGET = "House_overall_kW"
LOOKBACK = 168  # must match what 03_train_models.py trained with

DEFAULT_DATA_PATH = "data/processed/enervision_hourly.csv"
RF_MODEL_PATH = "models/random_forest.pkl"
LSTM_MODEL_PATH_KERAS = "models/lstm_model.keras"
LSTM_MODEL_PATH_H5 = "models/lstm_model.h5"
LSTM_SCALER_PATH = "models/lstm_scaler.pkl"
RESULTS_PATH = "results/model_comparison.csv"

CALENDAR_FEATURES = ["hour", "day", "month", "day_of_week", "is_weekend", "hour_sin", "hour_cos"]
LAG_FEATURES = ["lag_1h", "lag_24h", "lag_168h", "roll_24h_mean"]
WEATHER_FEATURES = [
    "temperature", "apparentTemperature", "humidity", "visibility",
    "pressure", "windSpeed", "windBearing", "cloudCover",
    "precipIntensity", "precipProbability", "dewPoint",
]
MODEL_FEATURES = CALENDAR_FEATURES[:5] + WEATHER_FEATURES + CALENDAR_FEATURES[5:] + LAG_FEATURES
# -> ["hour","day","month","day_of_week","is_weekend", <11 weather cols>,
#     "hour_sin","hour_cos","lag_1h","lag_24h","lag_168h","roll_24h_mean"]

# fallback numbers (from the project's own training run) so the
# Model Comparison tab never looks empty before results/ exists
FALLBACK_RESULTS = pd.DataFrame([
    {"model": "Random Forest (tuned)", "MAE": 0.2101, "RMSE": 0.3265, "MAPE(%)": 36.03, "R2": 0.5411},
    {"model": "LSTM",                  "MAE": 0.2564, "RMSE": 0.3678, "MAPE(%)": 42.96, "R2": 0.4177},
    {"model": "KNN (k=10)",            "MAE": 0.3062, "RMSE": 0.4475, "MAPE(%)": 46.94, "R2": 0.1380},
    {"model": "Naive (lag_24h)",       "MAE": 0.3583, "RMSE": 0.5551, "MAPE(%)": 58.44, "R2": -0.3260},
    {"model": "ARIMA(2,1,2)",          "MAE": 0.3705, "RMSE": 0.5422, "MAPE(%)": 55.09, "R2": -0.2652},
])


# ---------------------------------------------------------------
# Data loading / preparation
# ---------------------------------------------------------------
@st.cache_data
def load_default_data():
    if Path(DEFAULT_DATA_PATH).exists():
        return pd.read_csv(DEFAULT_DATA_PATH, parse_dates=["date"])
    return None


def prepare_uploaded_df(df):
    """Make a best effort to get an uploaded file into the shape the
    models expect: a `date` column, House_overall_kW, and the
    engineered features. If lag/cyclic features are missing but
    House_overall_kW and date are present, (re)compute them."""
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("Uploaded file must have a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    if TARGET not in df.columns:
        raise ValueError(f"Uploaded file must have a '{TARGET}' column.")

    # rebuild calendar features from the index (cheap, always safe to redo)
    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # lag/rolling features - only recompute if missing, since a
    # continuous hourly series is needed for these to be meaningful
    if "lag_1h" not in df.columns:
        df["lag_1h"] = df[TARGET].shift(1)
    if "lag_24h" not in df.columns:
        df["lag_24h"] = df[TARGET].shift(24)
    if "lag_168h" not in df.columns:
        df["lag_168h"] = df[TARGET].shift(168)
    if "roll_24h_mean" not in df.columns:
        df["roll_24h_mean"] = df[TARGET].shift(1).rolling(24).mean()

    missing_weather = [c for c in MODEL_FEATURES if c not in df.columns and c not in
                        CALENDAR_FEATURES + LAG_FEATURES]
    for c in missing_weather:
        df[c] = np.nan  # model will just get NaN -> we warn the user below

    df = df.reset_index()
    return df, missing_weather


# ---------------------------------------------------------------
# Model loading / training (cached so the app doesn't redo this on
# every click or widget change)
# ---------------------------------------------------------------
@st.cache_resource
def load_rf_model():
    if Path(RF_MODEL_PATH).exists():
        return joblib.load(RF_MODEL_PATH)
    return None


@st.cache_resource
def load_lstm_model():
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        return None, None
    model_path = LSTM_MODEL_PATH_KERAS if Path(LSTM_MODEL_PATH_KERAS).exists() else LSTM_MODEL_PATH_H5
    if not Path(model_path).exists() or not Path(LSTM_SCALER_PATH).exists():
        return None, None
    model = load_model(model_path)
    scaler = joblib.load(LSTM_SCALER_PATH)
    return model, scaler


@st.cache_resource(show_spinner="Training a quick KNN model on the loaded dataset...")
def train_knn_model(df_full):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    valid = df_full.dropna(subset=MODEL_FEATURES + [TARGET])
    knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
    knn.fit(valid[MODEL_FEATURES], valid[TARGET])
    return knn


@st.cache_resource(show_spinner="Fitting ARIMA on the loaded dataset...")
def fit_arima_model(y_series):
    from statsmodels.tsa.arima.model import ARIMA
    fitted = ARIMA(y_series, order=(2, 1, 2)).fit()
    return fitted


def have_tensorflow():
    try:
        import tensorflow  # noqa
        return True
    except ImportError:
        return False


def have_statsmodels():
    try:
        import statsmodels  # noqa
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------
# Prediction helpers (backtest - predicting over rows we already
# have actuals for, so we can show Actual vs Predicted + metrics)
# ---------------------------------------------------------------
def predict_rf_or_knn(model, df):
    X = df[MODEL_FEATURES].dropna()
    preds = model.predict(X)
    return X.index, preds


def predict_lstm(model, scaler, df):
    valid = df.dropna(subset=MODEL_FEATURES + [TARGET])
    data = valid[MODEL_FEATURES + [TARGET]].values
    scaled = scaler.transform(data)
    target_idx = len(MODEL_FEATURES)

    if len(scaled) <= LOOKBACK:
        return None, None  # not enough history for even one sequence

    X_seq, idx_seq = [], []
    for i in range(LOOKBACK, len(scaled)):
        X_seq.append(scaled[i - LOOKBACK:i, :])
        idx_seq.append(valid.index[i])
    X_seq = np.array(X_seq)

    pred_scaled = model.predict(X_seq, verbose=0).flatten()
    dummy = np.zeros((len(pred_scaled), len(MODEL_FEATURES) + 1))
    dummy[:, target_idx] = pred_scaled
    pred = scaler.inverse_transform(dummy)[:, target_idx]
    return idx_seq, pred


def backtest_arima(y_full):
    """Fit on the first 85% chronologically, forecast the rest, so we
    can show an actual-vs-predicted comparison for ARIMA too."""
    split = int(len(y_full) * 0.85)
    if split < 10 or len(y_full) - split < 2:
        return None, None, None
    from statsmodels.tsa.arima.model import ARIMA
    train_y = y_full.iloc[:split]
    test_y = y_full.iloc[split:]
    fit = ARIMA(train_y, order=(2, 1, 2)).fit()
    pred = fit.forecast(steps=len(test_y))
    return test_y.index, test_y.values, pred.values


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return mae, rmse, mape


# ---------------------------------------------------------------
# Forward forecasting (predicting beyond the last known timestamp,
# where we have NO actuals and NO future weather readings)
# ---------------------------------------------------------------
def _hour_climatology(df_full):
    """Average of every weather column, grouped by hour-of-day, used
    as a stand-in for 'future weather' since none is observed yet."""
    return df_full.groupby(df_full["hour"])[WEATHER_FEATURES].mean()


def recursive_forecast(model_kind, model, df_full, periods, scaler=None):
    """Step forward one hour at a time, re-deriving every feature a
    future row would have (calendar features exactly; weather
    features approximated from the historical hourly average, since
    real forecasts aren't available; lag/rolling features from the
    growing history, including our own prior predictions)."""
    hist = df_full.sort_values("date").set_index("date").copy()
    hist = hist.dropna(subset=[TARGET])
    clim = _hour_climatology(df_full)

    last_time = hist.index[-1]
    future_rows = []

    for step in range(1, periods + 1):
        t = last_time + pd.Timedelta(hours=step)
        row = {
            "hour": t.hour, "day": t.day, "month": t.month,
            "day_of_week": t.dayofweek, "is_weekend": int(t.dayofweek >= 5),
            "hour_sin": np.sin(2 * np.pi * t.hour / 24),
            "hour_cos": np.cos(2 * np.pi * t.hour / 24),
        }
        for wc in WEATHER_FEATURES:
            row[wc] = clim.loc[t.hour, wc] if t.hour in clim.index else df_full[wc].mean()

        tail = hist[TARGET]
        row["lag_1h"] = tail.iloc[-1]
        row["lag_24h"] = tail.iloc[-24] if len(tail) >= 24 else tail.iloc[0]
        row["lag_168h"] = tail.iloc[-168] if len(tail) >= 168 else tail.iloc[0]
        row["roll_24h_mean"] = tail.iloc[-24:].mean()

        X_row = pd.DataFrame([row])[MODEL_FEATURES]

        if model_kind in ("rf", "knn"):
            yhat = float(model.predict(X_row)[0])
        elif model_kind == "lstm":
            seq_source = hist[MODEL_FEATURES + [TARGET]].tail(LOOKBACK - 1)
            seq_df = pd.concat([seq_source, pd.concat([X_row, pd.DataFrame({TARGET: [np.nan]})], axis=1)])
            # last row's target is unknown - fill with the most recent
            # known value just so scaling has a number; it is never
            # used as an output, only LOOKBACK rows of *inputs* matter
            seq_df[TARGET] = seq_df[TARGET].fillna(method="ffill")
            scaled = scaler.transform(seq_df.values)
            x_seq = scaled[-LOOKBACK:, :].reshape(1, LOOKBACK, -1)
            pred_scaled = model.predict(x_seq, verbose=0).flatten()[0]
            dummy = np.zeros((1, len(MODEL_FEATURES) + 1))
            dummy[:, len(MODEL_FEATURES)] = pred_scaled
            yhat = float(scaler.inverse_transform(dummy)[:, len(MODEL_FEATURES)][0])
        else:
            raise ValueError(model_kind)

        row[TARGET] = yhat
        row_series = pd.Series(row, name=t)
        hist = pd.concat([hist, row_series.to_frame().T])
        hist.index.name = "date"
        future_rows.append((t, yhat))

    return pd.DataFrame(future_rows, columns=["date", "Predicted"])


def forecast_arima(y_full, periods):
    from statsmodels.tsa.arima.model import ARIMA
    fit = ARIMA(y_full, order=(2, 1, 2)).fit()
    fc = fit.forecast(steps=periods)
    last_time = y_full.index[-1]
    dates = [last_time + pd.Timedelta(hours=i) for i in range(1, periods + 1)]
    return pd.DataFrame({"date": dates, "Predicted": fc.values})


# ---------------------------------------------------------------
# Consumption classification (used by Manual Prediction + Alerts)
# ---------------------------------------------------------------
def classify_consumption(value, hour, df_full):
    same_hour = df_full.loc[df_full["hour"] == hour, TARGET]
    mean, std = same_hour.mean(), same_hour.std()
    if std == 0 or np.isnan(std):
        std = df_full[TARGET].std()
    if value > mean + 1.5 * std:
        return "🔴 High", mean, std
    elif value > mean + 0.5 * std:
        return "🟡 Moderate", mean, std
    return "🟢 Normal", mean, std


# ---------------------------------------------------------------
# Sidebar - data source & global filters
# ---------------------------------------------------------------
st.sidebar.title("⚡ EnerVision")
st.sidebar.caption("AI-powered energy consumption dashboard")

uploaded_file = st.sidebar.file_uploader("Upload your own hourly CSV", type=["csv"])

missing_cols_warning = []
if uploaded_file is not None:
    raw = pd.read_csv(uploaded_file)
    try:
        df, missing_cols_warning = prepare_uploaded_df(raw)
        st.sidebar.success(f"Loaded uploaded file: {len(df)} rows")
    except ValueError as e:
        st.sidebar.error(str(e))
        st.stop()
else:
    df = load_default_data()
    if df is None:
        st.sidebar.warning(
            "No default dataset found at data/processed/enervision_hourly.csv. "
            "Upload a CSV to get started."
        )
        st.stop()
    st.sidebar.info(f"Using project dataset: {len(df)} rows")

if missing_cols_warning:
    st.sidebar.warning(
        "Uploaded file is missing weather columns: "
        + ", ".join(missing_cols_warning)
        + ". Predictions may be less accurate."
    )

# date range filter
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date.date(), max_date.date()),
    min_value=min_date.date(), max_value=max_date.date(),
)
if len(date_range) == 2:
    start, end = date_range
    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    df_view = df.loc[mask].reset_index(drop=True)
else:
    df_view = df

st.sidebar.markdown("---")
st.sidebar.caption("EnerVision — AI-Powered Cloud-Based Energy Consumption Prediction System")

appliance_cols = [c for c in df.columns if c.endswith("_kW") and c != TARGET]

# ---------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------
tab_home, tab_analysis, tab_appliance, tab_predict, tab_compare, \
    tab_dataset, tab_manual, tab_alerts, tab_about = st.tabs([
        "🏠 Home", "📊 Energy Analysis", "🔌 Appliance Analysis",
        "🔮 Energy Prediction", "🏆 Model Comparison", "📁 Dataset",
        "✍️ Manual Prediction", "🚨 Alerts & Tips", "ℹ️ About",
    ])

# --- Tab: Home ---
with tab_home:
    st.subheader("Energy Consumption Overview")

    if Path(RESULTS_PATH).exists():
        results_df = pd.read_csv(RESULTS_PATH)
    else:
        results_df = FALLBACK_RESULTS.copy()
    best_model_name = results_df.sort_values("MAE").iloc[0]["model"]

    peak_row = df_view.loc[df_view[TARGET].idxmax()]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("⚡ Average Consumption", f"{df_view[TARGET].mean():.3f} kW")
    c2.metric("🔺 Maximum Consumption", f"{df_view[TARGET].max():.3f} kW")
    c3.metric("🔻 Minimum Consumption", f"{df_view[TARGET].min():.3f} kW")
    c4.metric("📈 Peak Time", peak_row["date"].strftime("%d %b, %H:%M"))
    c5.metric("🤖 Best Model", best_model_name)

    st.markdown("#### Energy Consumption Trend")
    granularity = st.radio("View by", ["Hour (raw)", "Day", "Week", "Month"], horizontal=True)
    trend_df = df_view.set_index("date")[[TARGET]]
    freq_map = {"Day": "D", "Week": "W", "Month": "ME"}
    if granularity != "Hour (raw)":
        trend_df = trend_df.resample(freq_map[granularity]).mean()
    fig = px.line(trend_df.reset_index(), x="date", y=TARGET,
                  title=f"House Overall Usage Over Time ({granularity})")
    fig.update_layout(xaxis_title="Date", yaxis_title="kW")
    st.plotly_chart(fig, width="stretch")

# --- Tab: Energy Analysis ---
with tab_analysis:
    st.subheader("Energy Analysis")

    col_a, col_b = st.columns(2)
    with col_a:
        hourly_avg = df_view.groupby("hour")[TARGET].mean().reset_index()
        fig = px.bar(hourly_avg, x="hour", y=TARGET, title="Average Consumption by Hour")
        st.plotly_chart(fig, width="stretch")
    with col_b:
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_avg = df_view.groupby("day_of_week")[TARGET].mean().reset_index()
        dow_avg["day_name"] = dow_avg["day_of_week"].apply(lambda d: dow_names[int(d)])
        fig = px.bar(dow_avg, x="day_name", y=TARGET, title="Average Consumption by Day of Week")
        st.plotly_chart(fig, width="stretch")

    col_c, col_d = st.columns(2)
    with col_c:
        daily = df_view.set_index("date")[TARGET].resample("D").mean().reset_index()
        fig = px.line(daily, x="date", y=TARGET, title="Daily Consumption")
        st.plotly_chart(fig, width="stretch")
    with col_d:
        weekly = df_view.set_index("date")[TARGET].resample("W").mean().reset_index()
        fig = px.line(weekly, x="date", y=TARGET, title="Weekly Consumption")
        st.plotly_chart(fig, width="stretch")

    monthly = df_view.set_index("date")[TARGET].resample("ME").mean().reset_index()
    fig = px.bar(monthly, x="date", y=TARGET, title="Monthly Consumption")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Weekday vs Weekend")
    weekend_avg = df_view.groupby("is_weekend")[TARGET].mean().reset_index()
    weekend_avg["label"] = weekend_avg["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    fig = px.bar(weekend_avg, x="label", y=TARGET, title="Weekday vs Weekend Average Consumption",
                 color="label")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Top 10 Peak Consumption Periods")
    peaks = df_view.nlargest(10, TARGET)[["date", TARGET, "hour", "day_of_week"]].copy()
    peaks["day_of_week"] = peaks["day_of_week"].apply(lambda d: dow_names[int(d)])
    peaks = peaks.rename(columns={TARGET: "Consumption (kW)", "day_of_week": "Day"})
    st.dataframe(peaks, hide_index=True, width="stretch")

# --- Tab: Appliance Analysis ---
with tab_appliance:
    st.subheader("Appliance Analysis")

    if not appliance_cols:
        st.info(
            "This dataset doesn't include individual appliance columns, "
            "so a per-appliance breakdown isn't available. Upload a file "
            "with columns like Refrigerator_kW, Air_conditioner_kW, etc. "
            "to see this view."
        )
    else:
        avg_usage = df_view[appliance_cols].mean().sort_values(ascending=False)
        avg_usage_df = avg_usage.reset_index()
        avg_usage_df.columns = ["Appliance", "Average kW"]

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(
                avg_usage_df, x="Average kW", y="Appliance", orientation="h",
                title="Top Energy Consuming Appliances",
            )
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, width="stretch")
        with col2:
            st.markdown("**Top 5 consumers**")
            st.dataframe(avg_usage_df.head(5), hide_index=True, width="stretch")

        st.markdown("#### Share of Total Consumption")
        positive = avg_usage_df[avg_usage_df["Average kW"] > 0]
        fig_pie = px.pie(positive, names="Appliance", values="Average kW",
                          title="Appliance Contribution (positive-draw appliances only)")
        st.plotly_chart(fig_pie, width="stretch")

        st.markdown("---")
        selected_appliance = st.selectbox("View trend for a specific appliance", appliance_cols)
        fig4 = px.line(df_view, x="date", y=selected_appliance, title=f"{selected_appliance} Over Time")
        st.plotly_chart(fig4, width="stretch")

# --- Tab: Energy Prediction ---
with tab_predict:
    st.subheader("Energy Prediction")

    available_models = ["Random Forest", "KNN"]
    if have_statsmodels():
        available_models.append("ARIMA")
    if have_tensorflow():
        available_models.append("LSTM")

    c1, c2 = st.columns(2)
    with c1:
        model_choice = st.selectbox("Prediction model", available_models)
    with c2:
        horizon_label = st.selectbox("Prediction period", ["Next hour", "Next 24 hours", "Next 7 days"])
    horizon_steps = {"Next hour": 1, "Next 24 hours": 24, "Next 7 days": 168}[horizon_label]

    st.markdown(f"#### Backtest — Actual vs Predicted ({model_choice})")

    df_pred = df_view.set_index("date")
    backtest_ok = False

    if model_choice == "Random Forest":
        rf_model = load_rf_model()
        if rf_model is None:
            st.error(f"No trained Random Forest model found at {RF_MODEL_PATH}. Run src/03_train_models.py first.")
        else:
            valid_idx, preds = predict_rf_or_knn(rf_model, df_pred)
            actual = df_pred.loc[valid_idx, TARGET]
            backtest_ok = True

    elif model_choice == "KNN":
        knn_model = train_knn_model(df)
        valid_idx, preds = predict_rf_or_knn(knn_model, df_pred)
        actual = df_pred.loc[valid_idx, TARGET]
        backtest_ok = True

    elif model_choice == "ARIMA":
        idx, actual_vals, preds = backtest_arima(df_pred[TARGET])
        if idx is None:
            st.warning("Selected date range is too short for an ARIMA backtest split. Widen the date range.")
        else:
            valid_idx, actual = idx, pd.Series(actual_vals, index=idx)
            backtest_ok = True

    else:  # LSTM
        lstm_model, lstm_scaler = load_lstm_model()
        if lstm_model is None:
            st.error("No trained LSTM model found, or TensorFlow isn't installed. Run src/03_train_models.py first.")
        elif len(df_pred) <= LOOKBACK:
            st.warning(
                f"LSTM needs at least {LOOKBACK} hours ({LOOKBACK // 24} days) of history. "
                f"Your selected date range only has {len(df_pred)} rows - widen it in the sidebar."
            )
        else:
            valid_idx, preds = predict_lstm(lstm_model, lstm_scaler, df_pred)
            if valid_idx is None:
                st.warning("Not enough valid (non-missing) rows to build LSTM sequences.")
            else:
                actual = df_pred.loc[valid_idx, TARGET]
                backtest_ok = True

    if backtest_ok:
        mae, rmse, mape = compute_metrics(actual, preds)
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{mae:.4f}")
        m2.metric("RMSE", f"{rmse:.4f}")
        m3.metric("MAPE", f"{mape:.2f}%")

        plot_df = pd.DataFrame({"date": valid_idx, "Actual": np.asarray(actual), "Predicted": np.asarray(preds)})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["Actual"], name="Actual", mode="lines"))
        fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["Predicted"], name="Predicted",
                                  mode="lines", line=dict(dash="dash")))
        fig.update_layout(title=f"Actual vs Predicted ({model_choice})", xaxis_title="Date", yaxis_title="kW")
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown(f"#### Forward Forecast — {horizon_label}")
    st.caption(
        "Forecasts beyond the last available timestamp. Future weather isn't known, so it's "
        "approximated from the historical average for that hour of day — treat this as a trend "
        "estimate, not a precise reading."
    )

    forecast_df = None
    if model_choice == "Random Forest" and load_rf_model() is not None:
        forecast_df = recursive_forecast("rf", load_rf_model(), df, horizon_steps)
    elif model_choice == "KNN":
        forecast_df = recursive_forecast("knn", train_knn_model(df), df, horizon_steps)
    elif model_choice == "ARIMA" and have_statsmodels():
        forecast_df = forecast_arima(df.set_index("date")[TARGET], horizon_steps)
    elif model_choice == "LSTM":
        lstm_model, lstm_scaler = load_lstm_model()
        if lstm_model is not None and len(df) > LOOKBACK:
            forecast_df = recursive_forecast("lstm", lstm_model, df, horizon_steps, scaler=lstm_scaler)

    if forecast_df is not None and len(forecast_df) > 0:
        f1, f2, f3 = st.columns(3)
        f1.metric("Predicted avg. consumption", f"{forecast_df['Predicted'].mean():.3f} kW")
        peak_fc = forecast_df.loc[forecast_df["Predicted"].idxmax()]
        f2.metric("Expected peak time", peak_fc["date"].strftime("%d %b, %H:%M"))
        f3.metric("Expected peak consumption", f"{peak_fc['Predicted']:.3f} kW")

        if len(forecast_df) > 1:
            trend_note = "rising" if forecast_df["Predicted"].iloc[-1] > forecast_df["Predicted"].iloc[0] else "falling"
            st.caption(f"Prediction trend: consumption is **{trend_note}** across the forecast window.")

            # prepend the last actual reading so the line connects history -> forecast
            # instead of floating on its own
            last_actual = df.sort_values("date").iloc[-1]
            context_row = pd.DataFrame({"date": [last_actual["date"]], "Predicted": [last_actual[TARGET]],
                                         "Series": ["Last actual"]})
            fc_rows = forecast_df.copy()
            fc_rows["Series"] = "Forecast"
            chart_df = pd.concat([context_row, fc_rows], ignore_index=True)

            fig = px.line(chart_df, x="date", y="Predicted", title=f"Forecasted Consumption — {horizon_label}",
                          markers=True)
            fig.update_layout(xaxis_title="Date", yaxis_title="kW")
            st.plotly_chart(fig, width="stretch")
        else:
            # a single point (Next hour) has no timeline to draw - the metrics
            # above already say everything a line chart would show
            st.info(f"Single-hour forecast: **{forecast_df['Predicted'].iloc[0]:.3f} kW** "
                    f"at {forecast_df['date'].iloc[0].strftime('%d %b, %H:%M')}. "
                    f"Pick \"Next 24 hours\" or \"Next 7 days\" to see a trend line.")
    else:
        st.info("Forecast unavailable for this model/horizon combination (missing model file or dependency).")

# --- Tab: Model Comparison ---
with tab_compare:
    st.subheader("Model Comparison")

    if Path(RESULTS_PATH).exists():
        results_df = pd.read_csv(RESULTS_PATH).sort_values("MAE")
        source_note = f"Loaded from `{RESULTS_PATH}`."
    else:
        results_df = FALLBACK_RESULTS.sort_values("MAE")
        source_note = "Showing results from the project's last recorded training run (no results/model_comparison.csv found)."

    best_model = results_df.iloc[0]["model"]
    has_r2 = "R2" in results_df.columns

    if has_r2:
        best_r2 = results_df.loc[results_df["model"] == best_model, "R2"].iloc[0]
        st.success(f"🏆 Best Performing Model: **{best_model}** — explains about "
                   f"**{best_r2 * 100:.0f}%** of the variance in consumption (R² = {best_r2:.3f})")
    else:
        st.success(f"🏆 Best Performing Model: **{best_model}**")
    st.caption(source_note)

    lower_is_better = [c for c in ["MAE", "RMSE", "MAPE(%)"] if c in results_df.columns]
    styler = results_df.style.highlight_min(subset=lower_is_better, color="#c6efce")
    if has_r2:
        styler = styler.highlight_max(subset=["R2"], color="#c6efce")
    st.dataframe(styler, hide_index=True, width="stretch")
    if has_r2:
        st.caption(
            "R² (coefficient of determination): the share of variance in energy consumption "
            "the model explains — 1.0 is a perfect fit, 0 is no better than always predicting "
            "the average, and negative means worse than that average."
        )

    fig = px.bar(
        results_df.melt(id_vars="model", value_vars=["MAE", "RMSE"]),
        x="model", y="value", color="variable", barmode="group",
        title="MAE & RMSE by Model (lower is better)",
    )
    st.plotly_chart(fig, width="stretch")

    fig2 = px.bar(results_df, x="model", y="MAPE(%)", title="MAPE (%) by Model (lower is better)")
    st.plotly_chart(fig2, width="stretch")

    if has_r2:
        fig3 = px.bar(results_df, x="model", y="R2", title="R² by Model (higher is better)")
        fig3.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig3, width="stretch")

# --- Tab: Dataset ---
with tab_dataset:
    st.subheader("Dataset Upload")
    st.markdown(
        "Use the **Upload your own hourly CSV** control in the sidebar to replace the project's "
        "dataset. The summary below always reflects whichever dataset is currently loaded."
    )

    n_records, n_cols = df.shape
    date_min, date_max = df["date"].min(), df["date"].max()
    missing_vals = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())

    s1, s2, s3 = st.columns(3)
    s1.metric("Number of records", f"{n_records:,}")
    s2.metric("Number of columns", n_cols)
    s3.metric("Date range", f"{date_min.date()} → {date_max.date()}")

    s4, s5, s6 = st.columns(3)
    s4.metric("Missing values", f"{missing_vals:,}")
    s5.metric("Duplicate rows", f"{dup_rows:,}")
    s6.metric("Average consumption", f"{df[TARGET].mean():.3f} kW")

    st.metric("Maximum consumption", f"{df[TARGET].max():.3f} kW")

    with st.expander("Preview first rows"):
        st.dataframe(df.head(20), width="stretch")

    if missing_vals > 0:
        with st.expander("Missing values by column"):
            miss_by_col = df.isna().sum()
            miss_by_col = miss_by_col[miss_by_col > 0].sort_values(ascending=False)
            st.dataframe(miss_by_col.reset_index().rename(columns={"index": "Column", 0: "Missing"}),
                         hide_index=True, width="stretch")

# --- Tab: Manual Prediction ---
with tab_manual:
    st.subheader("Manual Prediction")
    st.caption(
        "Enter the conditions for a single hour and get an instant prediction from the "
        "Random Forest model. Any model input not listed below (other weather fields, "
        "rolling average) is filled in automatically from the dataset's historical averages."
    )

    rf_model = load_rf_model()
    if rf_model is None:
        st.error(f"No trained Random Forest model found at {RF_MODEL_PATH}. Run src/03_train_models.py first.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            m_hour = st.slider("Hour", 0, 23, 18)
            m_day = st.slider("Day", 1, 31, 15)
            m_month = st.slider("Month", 1, 12, 6)
        with c2:
            m_temp = st.number_input("Temperature (°F)", value=float(df["temperature"].mean()))
            m_humidity = st.number_input("Humidity (0-1)", value=float(df["humidity"].mean()), min_value=0.0, max_value=1.0)
            m_wind = st.number_input("Wind speed", value=float(df["windSpeed"].mean()))
        with c3:
            m_lag1 = st.number_input("Previous hour consumption (kW)", value=float(df[TARGET].mean()))
            m_lag24 = st.number_input("Same hour, 24h ago (kW)", value=float(df[TARGET].mean()))
            m_lag168 = st.number_input("Same hour, 168h (7 days) ago (kW)", value=float(df[TARGET].mean()))

        if st.button("🔮 Predict Energy Consumption", type="primary"):
            dow = pd.Timestamp(year=2024, month=int(m_month), day=min(int(m_day), 28)).dayofweek
            row = {
                "hour": m_hour, "day": m_day, "month": m_month,
                "day_of_week": dow, "is_weekend": int(dow >= 5),
                "hour_sin": np.sin(2 * np.pi * m_hour / 24),
                "hour_cos": np.cos(2 * np.pi * m_hour / 24),
                "temperature": m_temp, "humidity": m_humidity, "windSpeed": m_wind,
                "lag_1h": m_lag1, "lag_24h": m_lag24, "lag_168h": m_lag168,
                "roll_24h_mean": np.mean([m_lag1, m_lag24, m_lag168]),
            }
            # auto-fill any remaining weather feature from the dataset average
            for wc in WEATHER_FEATURES:
                if wc not in row:
                    row[wc] = float(df[wc].mean())

            X_row = pd.DataFrame([row])[MODEL_FEATURES]
            prediction = float(rf_model.predict(X_row)[0])
            label, hour_mean, hour_std = classify_consumption(prediction, m_hour, df)

            st.markdown(f"### Predicted Energy Consumption: **{prediction:.2f} kW**")
            st.markdown(f"### {label} consumption for this hour")
            st.caption(f"Typical consumption at {m_hour}:00 is {hour_mean:.2f} kW (±{hour_std:.2f}).")

# --- Tab: Alerts & Tips ---
with tab_alerts:
    st.subheader("🚨 Peak / Abnormal Consumption Alert")

    latest = df.sort_values("date").iloc[-1]
    current_val, current_hour = latest[TARGET], int(latest["hour"])
    label, hour_mean, hour_std = classify_consumption(current_val, current_hour, df)
    diff = current_val - hour_mean

    if label.startswith("🔴"):
        st.error(f"⚠️ High Energy Consumption Detected\n\n"
                 f"Current consumption is significantly higher than the normal level for this time period.")
    elif label.startswith("🟡"):
        st.warning("Moderate energy consumption detected for this time period.")
    else:
        st.success("Consumption is within the normal range for this time period.")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Current / observed consumption", f"{current_val:.3f} kW")
    a2.metric("Normal average (this hour)", f"{hour_mean:.3f} kW")
    a3.metric("Difference", f"{diff:+.3f} kW")
    a4.metric("Classification", label)

    overall_peak = df.loc[df[TARGET].idxmax()]
    st.caption(f"Dataset-wide peak time: {overall_peak['date'].strftime('%d %b %Y, %H:%M')} "
               f"at {overall_peak[TARGET]:.3f} kW.")

    st.markdown("---")
    st.subheader("💡 Energy Saving Recommendations")

    hourly_avg = df.groupby("hour")[TARGET].mean()
    peak_hour = int(hourly_avg.idxmax())
    window = sorted({peak_hour, (peak_hour + 1) % 24, (peak_hour - 1) % 24})

    st.info(
        f"💡 **Recommendation**\n\n"
        f"Energy consumption is highest around **{window[0]:02d}:00–{window[-1]:02d}:00**. "
        f"Consider reducing simultaneous use of high-consumption appliances during this period."
    )

    weekend_avg = df.groupby("is_weekend")[TARGET].mean()
    if weekend_avg.get(1, 0) > weekend_avg.get(0, 0):
        st.info(
            "💡 **Recommendation**\n\n"
            "Weekend consumption is higher than weekday consumption. "
            "Review appliance usage during weekend evening hours."
        )
    else:
        st.info(
            "💡 **Recommendation**\n\n"
            "Weekday consumption is higher than weekend consumption. "
            "Review appliance usage during weekday evening hours, when occupancy and load both rise."
        )

    if appliance_cols:
        avg_usage = df[appliance_cols].mean().sort_values(ascending=False)
        positive = avg_usage[avg_usage > 0]
        if len(positive) > 0:
            top_appliance, top_share = positive.index[0], positive.iloc[0] / positive.sum() * 100
            if top_share > 25:
                st.info(
                    f"💡 **Recommendation**\n\n"
                    f"**{top_appliance}** accounts for about {top_share:.0f}% of tracked appliance draw. "
                    f"It's the best candidate for scheduling changes or efficiency upgrades."
                )

# --- Tab: About ---
with tab_about:
    st.subheader("☁️ About EnerVision")
    st.markdown(
        """
EnerVision is an AI-powered, cloud-based interactive energy consumption prediction system
that uses machine learning and deep learning models to analyze household energy usage,
forecast future demand, and surface actionable savings recommendations.
"""
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Technologies**")
        st.markdown("- Python, pandas, NumPy, scikit-learn\n- statsmodels (ARIMA)\n- TensorFlow/Keras (LSTM)\n- Streamlit + Plotly for the dashboard")
        st.markdown("**Models used**")
        st.markdown("- Naive baseline (lag 24h)\n- K-Nearest Neighbors\n- Random Forest\n- ARIMA(2,1,2)\n- LSTM (stacked, 128→64 units)")
    with c2:
        st.markdown("**Dataset information**")
        st.markdown(
            f"- Records: {len(df):,}\n"
            f"- Date range: {df['date'].min().date()} → {df['date'].max().date()}\n"
            f"- Target: `{TARGET}` (hourly, kW)\n"
            f"- {len(appliance_cols)} per-appliance columns, {len(WEATHER_FEATURES)} weather features"
        )
        st.markdown("**Project objectives**")
        st.markdown(
            "- Forecast household energy consumption from historical usage and weather\n"
            "- Compare classical and deep-learning forecasting approaches\n"
            "- Surface peak periods, anomalies, and saving opportunities to the end user"
        )

    st.markdown("**GitHub**")
    st.markdown("Add your project's GitHub repository link here, e.g. `https://github.com/<your-username>/EnerVision`")