import streamlit as st
import numpy as np
import pandas as pd
import requests
import joblib
from tensorflow.keras.models import load_model

# === Load model dan scaler ===
model = load_model("model_tcn_bilstm_gru.h5")
scaler = joblib.load("scaler_btc.save")

# === Konfigurasi Streamlit ===
st.set_page_config(page_title="Prediksi Harga Bitcoin", layout="centered")
st.title("🪙 Prediksi Harga Bitcoin Otomatis")
st.markdown("Prediksi otomatis berdasarkan 60 hari terakhir via CoinGecko Pro API")

# === Masukkan API Key (bisa disimpan di secrets atau langsung di code — lebih aman pakai secrets) ===
API_KEY = st.secrets["COINGECKO_API_KEY"] if "COINGECKO_API_KEY" in st.secrets else "YOUR_API_KEY_HERE"

@st.cache_data(ttl=3600)
def load_btc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "60",
        "interval": "daily"
    }
    headers = {
        "x-cg-pro-api-key": API_KEY
    }

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    prices = data['prices']
    df = pd.DataFrame(prices, columns=["Timestamp", "Close"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit='ms')
    df = df[["Date", "Close"]]
    return df

try:
    df = load_btc_data()
    st.subheader("📊 Harga Penutupan Bitcoin (60 Hari Terakhir)")
    st.dataframe(df.tail(), use_container_width=True)

    # === Preprocessing ===
    last_60 = df['Close'].values.reshape(-1, 1)
    last_60_scaled = scaler.transform(last_60)
    X_input = last_60_scaled.reshape(1, 60, 1)

    # === Prediksi ===
    y_pred_scaled = model.predict(X_input)
    y_pred = scaler.inverse_transform(y_pred_scaled)[0][0]

    st.subheader("📈 Hasil Prediksi")
    pred_date = (df['Date'].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    st.success(f"💰 Prediksi Harga Bitcoin untuk {pred_date}: **${y_pred:,.2f}**")

    st.line_chart(df.set_index("Date")["Close"], use_container_width=True)

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data atau memproses prediksi: {e}")
