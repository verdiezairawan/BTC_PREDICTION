import streamlit as st
import numpy as np
import pandas as pd
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go

# --- Fungsi baru untuk mengambil data dari Alpha Vantage ---
@st.cache_data(ttl=3600) # Menambah cache agar tidak mengambil data setiap kali reload
def get_alphavantage_data(symbol, market, api_key):
    """Mengambil data historis dari Alpha Vantage API dan mengembalikannya sebagai DataFrame."""
    try:
        cc = CryptoCurrencies(key=api_key, output_format='pandas')
        data, _ = cc.get_digital_currency_daily(symbol=symbol, market=market)
        
        if data is None or data.empty:
            st.warning("API Alpha Vantage tidak mengembalikan data. Pastikan API key valid.")
            return pd.DataFrame()
            
        # Format ulang DataFrame agar sesuai dengan kebutuhan model
        data.rename(columns={
            '4b. close (USD)': 'Close'
        }, inplace=True)
        
        # Mengubah index menjadi datetime dan mengurutkannya
        data.index = pd.to_datetime(data.index)
        data = data.sort_index(ascending=True)
        
        # Hanya ambil data 'Close' yang dibutuhkan
        return data[['Close']]
        
    except Exception as e:
        st.error(f"Gagal mengambil atau memproses data dari Alpha Vantage. Error: {e}")
        st.error("Pastikan API key Anda sudah benar dan ditambahkan di Secrets Streamlit.")
        return None

# --- Konfigurasi Awal dan Judul ---
st.set_page_config(page_title="Prediksi Harga Bitcoin", layout="wide")
st.title("Prediksi Harga Bitcoin (BTC)")

# --- Memuat model dan scaler ---
try:
    model = load_model('model_tcn_bilstm_gru.h5')
    scaler = joblib.load('scaler_btc.save')
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    st.stop()

# --- Mengambil API Key dari Streamlit Secrets ---
# Pastikan kamu sudah menambahkan ini di pengaturan aplikasi Streamlit!
try:
    alpha_vantage_api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
except KeyError:
    st.error("API Key Alpha Vantage tidak ditemukan. Silakan tambahkan di 'Settings > Secrets' pada aplikasi Streamlit Anda.")
    st.stop()
    
# --- Mengambil dan menampilkan data ---
btc_data = get_alphavantage_data(symbol='BTC', market='USD', api_key=alpha_vantage_api_key)

if btc_data is None or btc_data.empty:
    st.warning("Aplikasi tidak dapat melanjutkan karena gagal mengambil data.")
    st.stop()

st.subheader("Data Historis Harga Penutupan Bitcoin")
st.write(btc_data.tail())

# --- Pra-pemrosesan dan Prediksi (kode ini tetap sama) ---
try:
    close_prices = btc_data['Close'].values.reshape(-1, 1)
    scaled_close_prices = scaler.transform(close_prices)

    X_test = []
    # Mengambil data aktual yang sesuai dengan jumlah prediksi
    y_test_start_index = len(btc_data) - len(scaled_close_prices[60:])
    y_test = close_prices[y_test_start_index:, 0]
    
    for i in range(60, len(scaled_close_prices)):
        X_test.append(scaled_close_prices[i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
except Exception as e:
    st.error(f"Gagal dalam pra-pemrosesan data: {e}")
    st.stop()
    
# ... (sisa kode prediksi dan plot sama persis)
try:
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
except Exception as e:
    st.error(f"Gagal melakukan prediksi: {e}")
    st.stop()

# Menampilkan hasil prediksi
st.subheader('Prediksi vs Harga Aktual')

# Pastikan panjang data untuk plot sama
actual_dates = btc_data.index[y_test_start_index:]
if len(actual_dates) != len(predictions):
    st.warning("Terjadi ketidakcocokan panjang data antara prediksi dan data aktual.")
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_dates, y=y_test, mode='lines', name='Harga Aktual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=actual_dates, y=predictions.flatten(), mode='lines', name='Harga Prediksi', line=dict(color='red')))

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
