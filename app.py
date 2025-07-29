import streamlit as st
import numpy as np
import pandas as pd
import requests  # <-- Impor library baru
from datetime import datetime  # <-- Impor library baru
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go

# --- Fungsi baru untuk mengambil data dari CoinGecko ---
def get_coingecko_data(coin_id='bitcoin', vs_currency='usd', days='1825', interval='daily'):
    """Mengambil data historis dari CoinGecko API dan mengembalikannya sebagai DataFrame."""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': interval
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Ini akan menampilkan error jika request gagal (misal: 404, 500)
        
        data = response.json()['prices']
        
        if not data:
            st.warning("API CoinGecko tidak mengembalikan data.")
            return pd.DataFrame()
        
        # Konversi data ke format DataFrame yang sesuai
        df = pd.DataFrame(data, columns=['Timestamp', 'Close'])
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        
        # Hanya gunakan kolom 'Close' agar formatnya mirip dengan yfinance
        return df[['Close']]
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal terhubung ke API CoinGecko: {e}")
        return None
    except Exception as e:
        st.error(f"Terjadi error saat memproses data dari CoinGecko: {e}")
        return None

# Memuat model dan scaler
try:
    model = load_model('model_tcn_bilstm_gru.h5')
    scaler = joblib.load('scaler_btc.save')
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    st.stop()

# Judul Aplikasi
st.title("Prediksi Harga Bitcoin (BTC)")

# --- Mengganti blok yfinance dengan fungsi CoinGecko ---
btc_data = get_coingecko_data(days='1825') # 1825 hari = 5 tahun

if btc_data is None or btc_data.empty:
    st.warning("Gagal mengambil data harga Bitcoin. Aplikasi tidak dapat melanjutkan.")
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
