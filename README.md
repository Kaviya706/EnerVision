# ⚡ EnerVision – AI Energy Consumption Prediction System

## Description
EnerVision is an AI-powered, cloud-ready energy consumption prediction system that analyzes household energy usage, identifies consumption patterns, and forecasts future demand. It compares four forecasting models (Naive baseline, KNN, Random Forest, ARIMA, and LSTM) and presents everything through an interactive Streamlit dashboard — analysis, predictions, model comparison, and energy-saving recommendations in one place.

**Best result:** Random Forest, MAE = 0.21 kW, R² = 0.54 (explains ~54% of consumption variance) — outperforming every other model tested.

## Live Demo
https://enervision-mnkrh9ejsic9am7texgecd.streamlit.app/

## Features
- 🏠 **Home** — KPI cards (average/max/min consumption, peak time, best model) + usage trend (hourly/daily/weekly/monthly)
- 📊 **Energy Analysis** — hourly/daily/weekly/monthly breakdowns, peak consumption periods, weekday vs weekend comparison
- 🔌 **Appliance Analysis** — top energy-consuming appliances, contribution share, per-appliance trend
- 🔮 **Energy Prediction** — choose a model (Random Forest / KNN / ARIMA / LSTM) and a horizon (next hour / 24 hours / 7 days); see actual-vs-predicted backtest plus a forward forecast
- 🏆 **Model Comparison** — MAE / RMSE / MAPE / R² table and charts, best model highlighted
- 📁 **Dataset** — upload your own hourly CSV, automatic data-quality summary (records, columns, date range, missing values, duplicates)
- ✍️ **Manual Prediction** — enter conditions for a single hour, get an instant prediction with a normal/moderate/high classification
- 🚨 **Alerts & Tips** — abnormal-consumption alerts and rule-based energy-saving recommendations
- ℹ️ **About** — project info, tech stack, objectives

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn (Random Forest, KNN)
- Statsmodels (ARIMA)
- TensorFlow / Keras (LSTM)
- Streamlit + Plotly (dashboard and visualizations)
- Joblib (model persistence)

## Machine Learning Models
| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| **Random Forest (tuned)** | **0.2101** | **0.3265** | **36.03%** | **0.541** |
| LSTM (stacked 64→32 units) | 0.2564 | 0.3678 | 42.96% | 0.418 |
| KNN (k=10) | 0.3062 | 0.4475 | 46.94% | 0.138 |
| Naive (lag 24h baseline) | 0.3583 | 0.5551 | 58.44% | -0.326 |
| ARIMA(2,1,2) | 0.3705 | 0.5422 | 55.09% | -0.265 |

## Software Requirements
- Python 3.x
- Visual Studio Code / PyCharm
- Web Browser
- Internet Connection (for live deployment; the app also runs fully offline/local)

## Future Enhancements
- Real-time energy consumption monitoring
- Smart meter and IoT integration
- Energy cost prediction
- Advanced cloud-based data storage
- Mobile application integration

## Project Structure
```
EnerVision/
│
├── app/
│   └── streamlit_app.py          # the dashboard (9 sections)
│
├── requirements.txt
│
├── src/
│   ├── 02_resample_features.py   # hourly resampling + feature engineering
│   └── 03_train_models.py        # trains & compares all 5 models
│
├── data/
│   └── processed/
│       ├── enervision_hourly.csv
│       ├── train.csv
│       └── test.csv
│
├── models/
│   ├── random_forest.pkl
│   ├── lstm_model.keras
│   └── lstm_scaler.pkl
│
└── results/
    ├── model_comparison.csv
    └── lstm_training_history.csv
```

## Author
**Kaviya S**

GitHub: https://github.com/Kaviya706

LinkedIn: https://www.linkedin.com/in/kaviya-s
