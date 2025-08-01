import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

st.set_page_config(page_title="Scotland Birth Forecast", layout="wide")
st.title("Scotland Birth Forecasting with Deep Learning")

@st.cache_data
def load_data():
    df = pd.read_excel("clean_dataset.xlsx")
    df.drop(columns=['Year', 'Month', 'NHS_Board'], inplace=True, errors='ignore')
    df['Births'] = pd.to_numeric(df['Births'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Births', 'Date'], inplace=True)
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['log_births'] = np.log1p(df['Births'])
    return df

def create_sequences(data, window_size=12, horizon=1):
    X, y = [], []
    for i in range(window_size, len(data) - horizon + 1):
        X.append(data[i - window_size:i])
        y.append(data[i + horizon - 1])
    return np.array(X), np.array(y)

@st.cache_resource
def get_scaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data.reshape(-1, 1))
    return scaler

@st.cache_resource
def load_models():
    lstm = load_model("lstm_model.h5")
    bilstm = load_model("bilstm_model.h5")
    transformer = load_model("transformer_model.keras")
    return lstm, bilstm, transformer

def forecast_plot(model, X_val, y_val, scaler, title):
    pred = model.predict(X_val, verbose=0)
    pred_inv = np.expm1(scaler.inverse_transform(pred))
    y_val_inv = np.expm1(scaler.inverse_transform(y_val.reshape(-1, 1)))

    st.subheader(f"{title}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_val_inv, label="Actual")
    ax.plot(pred_inv, label="Forecast")
    ax.set_title(f"{title} - Scotland Births")
    ax.set_ylabel("Predicted Births")
    ax.legend()
    st.pyplot(fig)

    mae = np.mean(np.abs(y_val_inv - pred_inv))
    rmse = np.sqrt(np.mean((y_val_inv - pred_inv) ** 2))
    smape = 100 * np.mean(2 * np.abs(y_val_inv - pred_inv) / (np.abs(y_val_inv) + np.abs(pred_inv)))
    st.markdown(f"**MAE:** {mae:.2f} &nbsp;&nbsp;&nbsp; **RMSE:** {rmse:.2f} &nbsp;&nbsp;&nbsp; **SMAPE:** {smape:.2f}%")

def forecast_horizon_plot(model, data, scaler, window_size=12, horizon=1, title="Horizon Forecast"):
    input_seq = data[-window_size:].reshape(1, window_size, 1)
    predictions = []

    for _ in range(horizon):
        pred = model.predict(input_seq, verbose=0)
        predictions.append(pred[0, 0])
        input_seq = np.concatenate([input_seq[:, 1:, :], pred.reshape(1, 1, 1)], axis=1)

    pred_inv = np.expm1(scaler.inverse_transform(np.array(predictions).reshape(-1, 1)))
    future_dates = pd.date_range(df['Date'].iloc[-1], periods=horizon + 1, freq='MS')[1:]

    st.subheader(f"{title} - {horizon}-Month Ahead")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(future_dates, pred_inv, marker='o', label="Forecast")
    ax.set_title(f"{horizon}-Month Ahead Forecast")
    ax.set_ylabel("Predicted Births")
    ax.legend()
    st.pyplot(fig)
    st.markdown("Forecasts are generated sequentially from the last available month.")

def forecast_horizon_metrics_plot(model, data, scaler, window_size=12, horizon=1, title="Horizon Evaluation"):
    X_h, y_h = create_sequences(data, window_size=window_size, horizon=horizon)
    X_h = X_h.reshape((X_h.shape[0], X_h.shape[1], 1))
    pred = model.predict(X_h, verbose=0)

    pred_inv = np.expm1(scaler.inverse_transform(pred))
    y_h_inv = np.expm1(scaler.inverse_transform(y_h.reshape(-1, 1)))

    mae = np.mean(np.abs(y_h_inv - pred_inv))
    rmse = np.sqrt(np.mean((y_h_inv - pred_inv) ** 2))
    smape = 100 * np.mean(2 * np.abs(y_h_inv - pred_inv) / (np.abs(y_h_inv) + np.abs(pred_inv)))

    st.markdown(f"### {title} - {horizon}-Month Horizon")
    st.markdown(f"**MAE:** {mae:.2f} &nbsp;&nbsp;&nbsp; **RMSE:** {rmse:.2f} &nbsp;&nbsp;&nbsp; **SMAPE:** {smape:.2f}%")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_h_inv, label="Actual")
    ax.plot(pred_inv, label="Predicted")
    ax.set_title(f"{title} - {horizon}-Step Ahead")
    ax.set_ylabel("Predicted Births")
    ax.legend()
    st.pyplot(fig)

# ---------- MAIN APP ----------
with st.spinner("Loading data and models..."):
    df = load_data()
    log_values = df['log_births'].values
    scaler = get_scaler(log_values)
    scaled = scaler.transform(log_values.reshape(-1, 1)).flatten()

    lstm, bilstm, transformer = load_models()

# ---------- SIDEBAR ----------
model_choice = st.sidebar.selectbox("Choose Model", ["LSTM", "Bidirectional LSTM", "Transformer"])
horizon_choice = st.sidebar.selectbox("Forecast Horizon (months ahead)", [1, 3, 6, 12])

# ---------- VALIDATION FORECAST ----------
X_val, y_val = create_sequences(scaled, window_size=12)
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
split_idx = int(len(X_val) * 0.7)
X_val_eval, y_val_eval = X_val[split_idx:], y_val[split_idx:]

if model_choice == "LSTM":
    forecast_plot(lstm, X_val_eval, y_val_eval, scaler, "LSTM Forecast")
elif model_choice == "Bidirectional LSTM":
    forecast_plot(bilstm, X_val_eval, y_val_eval, scaler, "Bidirectional LSTM Forecast")
elif model_choice == "Transformer":
    forecast_plot(transformer, X_val_eval, y_val_eval, scaler, "Transformer Forecast")

# ---------- HORIZON FORECAST ----------
st.markdown("---")
st.subheader("Forecast Horizon Analysis")

if model_choice == "LSTM":
    forecast_horizon_plot(lstm, scaled, scaler, window_size=12, horizon=horizon_choice, title="LSTM Future Forecast")
    forecast_horizon_metrics_plot(lstm, scaled, scaler, window_size=12, horizon=horizon_choice, title="LSTM Historical Evaluation")
elif model_choice == "Bidirectional LSTM":
    forecast_horizon_plot(bilstm, scaled, scaler, window_size=12, horizon=horizon_choice, title="BiLSTM Future Forecast")
    forecast_horizon_metrics_plot(bilstm, scaled, scaler, window_size=12, horizon=horizon_choice, title="BiLSTM Historical Evaluation")
elif model_choice == "Transformer":
    forecast_horizon_plot(transformer, scaled, scaler, window_size=12, horizon=horizon_choice, title="Transformer Future Forecast")
    forecast_horizon_metrics_plot(transformer, scaled, scaler, window_size=12, horizon=horizon_choice, title="Transformer Historical Evaluation")
