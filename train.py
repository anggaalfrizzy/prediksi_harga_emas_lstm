# %% [markdown]
# # TA-04: Peramalan Harga Emas Harian dengan LSTM
# Alur lengkap: EDA, pembersihan, scaling, windowing, model LSTM (TensorFlow/Keras), evaluasi, lalu simpan model dan scaler.
# Bisa dijalankan sebagai notebook (`analisis_lstm.ipynb`) atau skrip (`python train.py`).

# %%
import json, random
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use("Agg")  # dijalankan sebagai skrip: jangan buka jendela plot
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

SEED, START, TIMESTEPS, TRAIN_RATIO, EPOCHS, HORIZON = 42, "2000-01-01", 60, 0.8, 80, 30
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
DATA, OUT = Path("data/gold_price.csv"), Path("model")
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Muat data dan eksplorasi
# Kolom waktu (`date`) dijadikan index. Data asli mulai 1968, tetapi harga emas sebelum 2000 berada di rezim yang sangat berbeda, jadi analisis dimulai dari 2000.

# %%
raw = pd.read_csv(DATA, parse_dates=["date"], index_col="date").sort_index()
print("Bentuk data  :", raw.shape, "|", raw.index.min().date(), "sampai", raw.index.max().date())
print("Missing price:", int(raw["price"].isna().sum()))

# %% [markdown]
# ## 2. Pembersihan
# Hari tanpa harga (umumnya hari libur) diisi dengan harga hari sebelumnya (`ffill`), lalu data difilter mulai 2000.

# %%
price = raw["price"].ffill().loc[START:].dropna()
df = price.to_frame()
print("Baris setelah dibersihkan:", len(df), "| missing:", int(df["price"].isna().sum()))
print(price.describe().round(2))

# %%
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
price.plot(ax=ax[0], lw=1, label="Harga harian")
price.rolling(252).mean().plot(ax=ax[0], lw=2, label="Rata-rata 252 hari")
ax[0].set_title("Harga emas harian: tren jangka panjang"); ax[0].legend(); ax[0].grid(True)
price.resample("YE").mean().plot(kind="bar", ax=ax[1], color="#C99A2E")
ax[1].set_title("Rata-rata harga per tahun"); ax[1].set_xlabel("")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. Scaling dan windowing
# Scaler hanya di-`fit` pada **data latih** supaya tidak ada kebocoran informasi dari masa depan. Setiap sampel berisi 60 hari terakhir (X) untuk memprediksi hari ke-61 (y), dalam bentuk tensor 3D `(samples, timesteps, features)`.

# %%
n = len(df)
split = int(n * TRAIN_RATIO)                 # indeks baris pertama data uji (urutan waktu dijaga)
prices = df[["price"]].values
scaler = MinMaxScaler(feature_range=(0, 1)).fit(prices[:split])
scaled = scaler.transform(prices)

X, y = [], []
for i in range(TIMESTEPS, n):
    X.append(scaled[i - TIMESTEPS:i, 0]); y.append(scaled[i, 0])
X, y = np.array(X).reshape(-1, TIMESTEPS, 1), np.array(y)

target_idx = np.arange(TIMESTEPS, n)
is_train = target_idx < split
X_train, y_train, X_test, y_test = X[is_train], y[is_train], X[~is_train], y[~is_train]
test_dates = df.index[TIMESTEPS:][~is_train]
print("X_train:", X_train.shape, "| X_test:", X_test.shape)
print("Data uji dimulai:", test_dates[0].date())

# %% [markdown]
# ## 4. Arsitektur dan pelatihan model LSTM
# 2 layer LSTM (50 unit), Dropout 0.2 untuk mengurangi overfitting, lalu Dense. Compile dengan `adam` dan `mean_squared_error`. Validasi memakai 10% bagian akhir data latih, sehingga data uji tetap murni untuk evaluasi akhir.

# %%
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(TIMESTEPS, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1),
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# %%
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32,
                    validation_split=0.1, callbacks=[es], verbose=2)

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="loss"); plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Kurva loss"); plt.xlabel("Epoch"); plt.legend(); plt.grid(True); plt.show()

# %% [markdown]
# ## 5. Evaluasi pada data uji
# Prediksi dan nilai aktual dikembalikan ke skala harga asli (*inverse transform*). Sebagai pembanding ada tebakan sederhana: harga hari ini sama dengan harga kemarin.

# %%
pred = scaler.inverse_transform(model.predict(X_test, verbose=0)).ravel()
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
naive = prices[target_idx[~is_train] - 1, 0]

rmse = float(np.sqrt(mean_squared_error(actual, pred)))
mae = float(mean_absolute_error(actual, pred))
mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
naive_rmse = float(np.sqrt(mean_squared_error(actual, naive)))
naive_mae = float(mean_absolute_error(actual, naive))
print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")
print(f"Pembanding (harga kemarin): RMSE {naive_rmse:.2f} | MAE {naive_mae:.2f}")

plt.figure(figsize=(14, 6))
plt.plot(test_dates, actual, color="blue", label="Harga aktual")
plt.plot(test_dates, pred, color="orange", label="Harga prediksi")
plt.title("Prediksi harga emas dengan LSTM (data uji)"); plt.xlabel("Tanggal"); plt.ylabel("Harga")
plt.legend(); plt.grid(True); plt.show()

# %% [markdown]
# ## 6. Peramalan ke depan (rekursif)
# Prediksi satu hari dimasukkan kembali ke jendela 60 hari untuk meramal hari berikutnya. Semakin jauh horizonnya, semakin besar error yang menumpuk.

# %%
def recursive_forecast(model, scaler, series, steps, timesteps=TIMESTEPS):
    window = scaler.transform(series.values[-timesteps:].reshape(-1, 1)).ravel().tolist()
    out = []
    for _ in range(steps):
        x = np.array(window[-timesteps:]).reshape(1, timesteps, 1)
        p = float(model.predict(x, verbose=0)[0, 0])
        out.append(p); window.append(p)
    return scaler.inverse_transform(np.array(out).reshape(-1, 1)).ravel()

fc_values = recursive_forecast(model, scaler, price, HORIZON)
fc_dates = pd.bdate_range(price.index[-1] + pd.Timedelta(days=1), periods=HORIZON)
print(pd.Series(fc_values, index=fc_dates.date).round(2).head(10))

# %% [markdown]
# ## 7. Simpan model, scaler, dan hasil evaluasi
# File di folder `model/` dibaca oleh aplikasi Flask (`app.py`).

# %%
model.save(str(OUT / "gold_lstm.h5"))
joblib.dump(scaler, OUT / "scaler.joblib")

r = lambda a, d=2: [round(float(v), d) for v in a]
(OUT / "config.json").write_text(json.dumps({
    "start": START, "timesteps": TIMESTEPS, "rows": n, "horizon": HORIZON,
    "last_date": str(price.index[-1].date()), "split_date": str(df.index[split].date())}))
(OUT / "metrics.json").write_text(json.dumps({
    "rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 2),
    "naive_rmse": round(naive_rmse, 2), "naive_mae": round(naive_mae, 2),
    "train_samples": int(len(X_train)), "test_samples": int(len(X_test)),
    "epochs_run": len(history.history["loss"]), "params": int(model.count_params())}))
(OUT / "eval.json").write_text(json.dumps({
    "dates": [str(d.date()) for d in test_dates], "actual": r(actual), "predicted": r(pred)}))
(OUT / "history.json").write_text(json.dumps({
    "loss": r(history.history["loss"], 6), "val_loss": r(history.history["val_loss"], 6)}))
(OUT / "forecast.json").write_text(json.dumps({
    "dates": [str(d.date()) for d in fc_dates], "values": r(fc_values)}))
print("Tersimpan di folder model/:", sorted(p.name for p in OUT.iterdir()))
