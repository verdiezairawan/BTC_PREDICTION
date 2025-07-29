import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import joblib

# Load model awal
model = load_model('model_tcn_bilstm_gru.h5', compile=False)

# Load data terbaru
df = yf.download("BTC-USD", start="2021-01-01")[["Close"]].dropna()

# Buat ulang scaler dari data baru
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[['Close']])

def create_dataset(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_close, time_steps=60)

# Reshape input sesuai model
X = X.reshape((X.shape[0], X.shape[1], 1))

from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.fit(X, y, epochs=30, batch_size=32, verbose=1)

# Prediksi harga berikutnya
last_60 = scaled_close[-60:]
input_pred = last_60.reshape((1, 60, 1))
pred_scaled = model.predict(input_pred)
pred_price = scaler.inverse_transform(pred_scaled)[0][0]

print(f"Hasil Prediksi: ${pred_price:,.2f}")

model.save("model_finetuned_btc_2025.h5")
joblib.dump(scaler, "scaler_finetuned.save")
