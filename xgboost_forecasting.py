import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('household_power_consumption.txt', sep=';', 
                 parse_dates={'datetime': ['Date', 'Time']},
                 infer_datetime_format=True, 
                 low_memory=False, na_values='?')
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df = df.dropna(subset=['Global_active_power'])

# Set datetime index
df.set_index('datetime', inplace=True)
df_hourly = df['Global_active_power'].resample('H').mean()
df_daily = df['Global_active_power'].resample('D').mean()
df_monthly = df['Global_active_power'].resample('M').mean()

def create_features(df, lags, freq):
    df_feat = pd.DataFrame(df)
    for lag in lags:
        df_feat[f'lag_{lag}'] = df.shift(lag)
    df_feat['hour'] = df.index.hour if freq == 'H' else 0
    df_feat['dayofweek'] = df.index.dayofweek if freq in ['H', 'D'] else 0
    df_feat['month'] = df.index.month
    df_feat.dropna(inplace=True)
    return df_feat

def train_xgb_model(df_feat, title,threshold=5.0):
    X = df_feat.drop(columns=['Global_active_power'])
    y = df_feat['Global_active_power']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'{title} RMSE: {rmse:.3f}')

    plt.figure(figsize=(10, 4))
    plt.plot(y_test.values[:300], label='True')
    plt.plot(y_pred[:300], label='Predicted')
    plt.title(f'{title} Forecast')
    plt.legend()
    plt.show()

    alerts = []
    for i, value in enumerate(y_pred):
        if value > threshold:
            alerts.append((i, value, "⚠️ High consumption — consider reducing appliance usage."))
        elif value > 0.9 * threshold:
            alerts.append((i, value, "⚠️ Near limit — monitor usage carefully."))

    # Print recommendations
    if alerts:
        print("\n🔔 Energy Usage Alerts:")
        for idx, val, msg in alerts[:10]:
            print(f"Time Index {idx} → Predicted: {val:.2f} kW → {msg}")
    else:
        print("\n✅ All predicted values are within safe limits.")


# Hourly Forecast
"""print("📘 Hourly Forecast")
hourly_feat = create_features(df_hourly, lags=[1, 2, 24, 48, 168], freq='H')
train_xgb_model(hourly_feat, 'Hourly')"""

# Daily Forecast
print("📗 Daily Forecast")
daily_feat = create_features(df_daily, lags=[1, 2, 7, 14, 30], freq='D')
train_xgb_model(daily_feat, 'Daily')

# Monthly Forecast
"""print("📙 Monthly Forecast")
monthly_feat = create_features(df_monthly, lags=[1, 2, 12], freq='M')
train_xgb_model(monthly_feat, 'Monthly')"""
