"""Aplikasi Flask: peramalan harga emas harian dengan LSTM (TA-04)."""
import json, os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE / "model"))
UNIT = os.environ.get("PRICE_UNIT", "USD/oz")  # sesuaikan dengan satuan pada sumber dataset
app = Flask(__name__)


def read_json(name):
    p = MODEL_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


CONFIG = read_json("config.json") or {"start": "2000-01-01", "timesteps": 60, "horizon": 30}
raw = pd.read_csv(BASE / "data" / "gold_price.csv", parse_dates=["date"], index_col="date").sort_index()
PRICES = raw["price"].ffill().loc[CONFIG["start"]:].dropna()

_cache = {}


def get_model():
    """Muat model Keras dan scaler sekali saja. None jika belum tersedia."""
    if "m" not in _cache:
        try:
            import joblib
            from tensorflow.keras.models import load_model
            _cache["m"] = (load_model(MODEL_DIR / "gold_lstm.h5", compile=False),
                           joblib.load(MODEL_DIR / "scaler.joblib"))
        except Exception as exc:  # TensorFlow atau file model tidak tersedia
            app.logger.warning("Model tidak dimuat: %s", exc)
            _cache["m"] = None
    return _cache["m"]


def live_forecast(days):
    model, scaler = get_model()
    ts = CONFIG["timesteps"]
    window = scaler.transform(PRICES.values[-ts:].reshape(-1, 1)).ravel().tolist()
    out = []
    for _ in range(days):
        x = np.array(window[-ts:]).reshape(1, ts, 1)
        p = float(model.predict(x, verbose=0)[0, 0])
        out.append(p)
        window.append(p)
    return scaler.inverse_transform(np.array(out).reshape(-1, 1)).ravel().tolist()


def err(message, code=503):
    return jsonify(message=message), code


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return jsonify(status="ok")


@app.get("/api/summary")
def summary():
    last = float(PRICES.iloc[-1])
    year_ago = float(PRICES.iloc[-253]) if len(PRICES) > 253 else float(PRICES.iloc[0])
    return jsonify(unit=UNIT, last_price=round(last, 2), last_date=str(PRICES.index[-1].date()),
                   change_1y=round((last / year_ago - 1) * 100, 2), first_date=str(PRICES.index[0].date()),
                   rows=len(PRICES), model_ready=read_json("metrics.json") is not None)


@app.get("/api/history")
def history():
    return jsonify(dates=[str(d.date()) for d in PRICES.index], prices=[round(float(v), 2) for v in PRICES.values])


@app.get("/api/evaluation")
def evaluation():
    metrics, ev, hist = read_json("metrics.json"), read_json("eval.json"), read_json("history.json")
    if not (metrics and ev):
        return err("Model belum dilatih. Jalankan: python train.py")
    return jsonify(metrics=metrics, eval=ev, history=hist, config=CONFIG)


@app.get("/api/forecast")
def forecast():
    horizon = CONFIG.get("horizon", 30)
    days = max(1, min(request.args.get("days", 10, type=int), horizon))
    base = {"last_date": str(PRICES.index[-1].date()), "last_price": round(float(PRICES.iloc[-1]), 2)}
    if get_model():
        dates = pd.bdate_range(PRICES.index[-1] + pd.Timedelta(days=1), periods=days)
        vals = [round(v, 2) for v in live_forecast(days)]
        return jsonify(mode="live", dates=[str(d.date()) for d in dates], values=vals, **base)
    pre = read_json("forecast.json")
    if pre:
        return jsonify(mode="precomputed", dates=pre["dates"][:days], values=pre["values"][:days], **base)
    return err("Model belum dilatih. Jalankan: python train.py")


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
