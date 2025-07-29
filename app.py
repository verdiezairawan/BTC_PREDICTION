import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model # <-- Perubahan di sini
import joblib
import plotly.graph_objects as go

# Memuat model dan scaler
try:
    model = load_model('model_tcn_bilstm_gru.h5')
    scaler = joblib.load('scaler_btc.save')
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    st.stop()

# Judul Aplikasi
st.title("Prediksi Harga Bitcoin (BTC)")

# Mengambil data historis Bitcoin dari Yahoo Finance
try:
    btc_data = yf.download(tickers='BTC-USD', period='5y', interval='1d')
    if btc_data.empty:
        st.warning("Tidak ada data yang diambil dari Yahoo Finance.")
        st.stop()
except Exception as e:
    st.error(f"Gagal mengambil data dari yfinance: {e}")
    st.stop()

# Menampilkan data mentah
st.subheader("Data Historis Harga Penutupan Bitcoin (5 Tahun Terakhir)")
st.write(btc_data.tail())

# Mempersiapkan data untuk prediksi
try:
    close_prices = btc_data['Close'].values.reshape(-1, 1)
    scaled_close_prices = scaler.transform(close_prices)

    X_test = []
    y_test = close_prices[60:, 0]
    for i in range(60, len(scaled_close_prices)):
        X_test.append(scaled_close_prices[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
except Exception as e:
    st.error(f"Gagal dalam pra-pemrosesan data: {e}")
    st.stop()

# Melakukan prediksi
try:
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
except Exception as e:
    st.error(f"Gagal melakukan prediksi: {e}")
    st.stop()

# Menampilkan hasil prediksi
st.subheader('Prediksi vs Harga Aktual')

fig = go.Figure()
fig.add_trace(go.Scatter(x=btc_data.index[60:], y=y_test, mode='lines', name='Harga Aktual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=btc_data.index[60:], y=predictions.flatten(), mode='lines', name='Harga Prediksi', line=dict(color='red')))

fig.update_layout(
    title='Perbandingan Harga Aktual dan Prediksi Bitcoin',
    xaxis_title='Tanggal',
    yaxis_title='Harga (USD)',
    legend_title='Keterangan'
)

st.plotly_chart(fig)

# Prediksi untuk hari berikutnya
try:
    last_60_days = scaled_close_prices[-60:]
    last_60_days = np.reshape(last_60_days, (1, 60, 1))
    next_day_prediction_scaled = model.predict(last_60_days)
    next_day_prediction = scaler.inverse_transform(next_day_prediction_scaled)

    st.subheader('Prediksi Harga untuk Besok')
    st.write(f"Prediksi harga penutupan Bitcoin untuk hari berikutnya adalah: **${next_day_prediction[0][0]:.2f}**")
except Exception as e:
    st.error(f"Gagal memprediksi harga untuk hari berikutnya: {e}")
