import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tcn import TCN
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# CONFIGURASI DASAR APLIKASI
# -----------------------------
st.set_page_config(page_title="Prediksi Harga Bitcoin", layout="wide")
st.title("📊 Prediksi Harga Bitcoin Otomatis (TCN–BiLSTM–GRU)")

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
@st.cache_resource
def load_prediction_model():
    model = load_model("model_finetuned_btc_2025.h5",compile=False, custom_objects={"TCN": TCN})
    scaler = joblib.load("scaler_finetuned.save")
    return model, scaler

model, scaler = load_prediction_model()

model, scaler = load_prediction_model()

# DEBUG SCALER
print("Min:", scaler.data_min_)
print("Max:", scaler.data_max_)


# -----------------------------
# AMBIL DATA TERBARU DARI YFINANCE
# -----------------------------
@st.cache_data
def fetch_btc_data(window_size=60):
    df = yf.download("BTC-USD", period="90d", interval="1d")
    df = df[["Close"]].dropna()
    return df.tail(window_size)

df = fetch_btc_data()

# -----------------------------
# PREPROCESS DAN PREDIKSI
# -----------------------------
def prepare_input(df, window_size=60):
    scaled_data = scaler.transform(df[['Close']])
    X = [scaled_data[-window_size:]]
    return np.array(X)

X_input = prepare_input(df)
pred_scaled = model.predict(X_input)
predicted_price = scaler.inverse_transform(pred_scaled)[0][0]

# -----------------------------
# TAMPILAN UI
# -----------------------------
st.subheader("📅 Tabel Harga Bitcoin Terbaru")
st.dataframe(df.tail(10), use_container_width=True)

st.subheader("🔮 Hasil Prediksi Harga Berikutnya")
st.metric("Harga Prediksi (USD)", f"${predicted_price:,.2f}")

print("Predicted scaled shape:", pred_scaled.shape)
print("Predicted scaled value:", pred_scaled)
