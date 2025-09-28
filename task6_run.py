import os
import time
import random
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from itertools import product
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ===================== Paths & Constants =====================
DATA_DIR = "data"

TICKER = "AAPL"
START = "2018-01-01"
N_STEPS = 60
HORIZON = 1          # single-step horizon
K_STEPS = 5          # multistep forecast horizon

# MODE: choose between "uni_multistep", "multi_single", "multi_multistep"
MODE = "multi_single"

EPOCHS = 40
BATCH = 32

SEARCH_SPACE = {
    "layer_type": ["LSTM", "GRU", "RNN"],
    "num_layers": [1, 2],
    "hidden_size": [64, 128],
    "dropout": [0.2],
    "lr": [1e-3],
    "batch_size": [BATCH],
    "epochs": [EPOCHS],
}

def _cache_path(ticker: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{ticker.upper()}.csv")

# ===================== Column normalization helpers =====================
def _strip_ticker_from_col(col: str, ticker: str) -> str:
    s = str(col).strip().replace(" ", "_").lower()
    tl = ticker.lower()
    for sep in ("_", "-", ".", " "):
        if s.endswith(sep + tl):
            s = s[: -(len(tl) + 1)]
        if s.startswith(tl + sep):
            s = s[len(tl) + 1 :]
    while "__" in s:
        s = s.replace("__", "_")
    return s

def _normalize_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels[0]) == 1:
            df.columns = df.columns.droplevel(0)
        else:
            df.columns = ["_".join([str(x) for x in tup if x is not None]).strip() for tup in df.columns]
    return df.rename(columns={c: _strip_ticker_from_col(c, ticker) for c in df.columns})

def _sanitize_ohlc(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError("Empty DataFrame in _sanitize_ohlc.")
    df = _normalize_columns(df, ticker)

    if not isinstance(df.index, pd.DatetimeIndex):
        date_candidates = [c for c in df.columns if c in ("date", "timestamp")]
        if date_candidates:
            dcol = date_candidates[0]
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
            df = df.set_index(dcol)
        else:
            df = df.reset_index()
            first = df.columns[0]
            df[first] = pd.to_datetime(df[first], errors="coerce")
            df = df.set_index(first)
    df = df[~df.index.isna()].sort_index()

    cols_lower = {c: c.lower() for c in df.columns}
    def pick(token):
        token = token.replace(" ", "_")
        for c, low in cols_lower.items():
            if low == token or low.endswith("_" + token) or low.startswith(token + "_") or token in low:
                return c
        return None

    open_c, high_c, low_c, close_c, vol_c = pick("open"), pick("high"), pick("low"), pick("close") or pick("adj_close"), pick("volume")
    selected = [c for c in (open_c, high_c, low_c, close_c, vol_c) if c]
    if not selected or close_c is None:
        raise RuntimeError(f"No OHLC columns found. Got columns: {list(df.columns)}")

    df = df[selected].copy().apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[close_c])
    rename_map = {open_c:"Open", high_c:"High", low_c:"Low", close_c:"Close"}
    if vol_c: rename_map[vol_c] = "Volume"
    return df.rename(columns=rename_map)

# ===================== Fetch prices =====================
def fetch_prices(ticker="AAPL", start="2018-01-01", end=None, use_cache=True, retries=5):
    cache_file = _cache_path(ticker)
    if use_cache and os.path.exists(cache_file):
        for opts in ({"parse_dates": ["Date"], "index_col": "Date"}, {"parse_dates": [0], "index_col": 0}, {}):
            try:
                df = pd.read_csv(cache_file, **opts, on_bad_lines="skip")
                return _sanitize_ohlc(df, ticker)
            except Exception:
                pass
    last_err = None
    for i in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False, threads=False)
            if df is not None and not df.empty:
                df = _sanitize_ohlc(df, ticker)
                df.to_csv(cache_file, index_label="Date")
                return df
            raise RuntimeError("Empty DataFrame from yfinance (download).")
        except Exception as e:
            last_err = e
            time.sleep([2,4,8,12,16][min(i,4)] * random.uniform(0.7, 1.3))
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, end=end, auto_adjust=True)
        if hist is not None and not hist.empty:
            df = _sanitize_ohlc(hist, ticker)
            df.to_csv(cache_file, index_label="Date")
            return df
        raise RuntimeError("Empty history.")
    except Exception as e_hist:
        last_err = f"{last_err} | history(): {e_hist}"
    try:
        stq = pd.read_csv(f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d")
        if stq is None or stq.empty or stq.shape[1] < 2:
            raise RuntimeError("Unexpected Stooq response.")
        stq = _sanitize_ohlc(stq, ticker)
        stq.to_csv(cache_file, index_label="Date")
        return stq
    except Exception as e2:
        raise RuntimeError(f"Failed to fetch {ticker}. Yahoo error: {last_err}; Stooq error: {e2}")

# ===================== Scaling =====================
def minmax_scale(series: pd.Series):
    s = pd.Series(series).squeeze().astype("float32")
    smin, smax = float(s.min()), float(s.max())
    denom = smax - smin if smax != smin else 1.0
    scaled = (s - smin) / denom
    return scaled.astype("float32"), smin, smax

def minmax_scale_matrix(X: np.ndarray):
    X = np.asarray(X, dtype="float32")
    mins, maxs = X.min(axis=0), X.max(axis=0)
    denom = np.where((maxs - mins) == 0.0, 1.0, (maxs - mins))
    return ((X - mins) / denom).astype("float32"), mins.astype("float32"), maxs.astype("float32")

# ===================== Windows =====================
def make_sequences(series: pd.Series, n_steps=60, horizon=1):
    vals = series.values.astype("float32")
    X, y = [], []
    for i in range(n_steps, len(vals) - horizon + 1):
        X.append(vals[i-n_steps:i]); y.append(vals[i+horizon-1])
    return np.array(X)[:, :, None], np.array(y)[:, None]

def make_windows_multistep_univariate(close_scaled, n_steps=60, k=5):
    vals = np.asarray(close_scaled, dtype="float32").reshape(-1)
    Xs, ys = [], []
    for i in range(n_steps, len(vals)-k+1):
        Xs.append(vals[i-n_steps:i]); ys.append(vals[i:i+k])
    return np.array(Xs, dtype="float32").reshape(-1, n_steps, 1), np.array(ys, dtype="float32")

def make_windows_multivariate_single_step(X_scaled, y_scaled, n_steps=60, horizon=1):
    # Ensure NumPy arrays and correct shapes (fix for tuple indexing on Series)
    X_scaled = np.asarray(X_scaled, dtype="float32")
    y_scaled = np.asarray(y_scaled, dtype="float32").reshape(-1, 1)
    Xs, ys = [], []
    N = len(X_scaled)
    for i in range(n_steps, N - horizon + 1):
        Xs.append(X_scaled[i-n_steps:i, :]); ys.append(y_scaled[i+horizon-1, 0])
    return np.array(Xs, dtype="float32"), np.array(ys, dtype="float32").reshape(-1, 1)

def make_windows_multivariate_multistep(X_scaled, y_scaled, n_steps=60, k=5):
    # Ensure NumPy arrays and correct shapes
    X_scaled = np.asarray(X_scaled, dtype="float32")
    y_scaled = np.asarray(y_scaled, dtype="float32").reshape(-1, 1)
    Xs, ys = [], []
    N = len(X_scaled)
    for i in range(n_steps, N - k + 1):
        Xs.append(X_scaled[i-n_steps:i, :]); ys.append(y_scaled[i:i+k, 0])
    return np.array(Xs, dtype="float32"), np.array(ys, dtype="float32")

# ===================== Model =====================
def build_sequence_model(input_shape, layer_type="LSTM", num_layers=2, hidden_size=128, dropout=0.2, dense_units=1, lr=1e-3):
    Layer = {"LSTM": layers.LSTM, "GRU": layers.GRU, "RNN": layers.SimpleRNN}[layer_type.upper()]
    model = models.Sequential(name=f"{layer_type}_L{num_layers}_H{hidden_size}")
    for i in range(num_layers):
        return_seq = i < num_layers-1
        cell = Layer(hidden_size, return_sequences=return_seq)
        if i == 0:
            model.add(cell.__class__(**cell.get_config(), input_shape=input_shape))
        else:
            model.add(cell)
        if dropout and return_seq:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dense_units))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae","mape"])
    return model

# ===================== Train/Eval =====================
def split_time_series(X, y, train_ratio=0.7, val_ratio=0.15):
    N = len(X); ntr = int(N*train_ratio); nv = int(N*val_ratio)
    return (X[:ntr], y[:ntr]), (X[ntr:ntr+nv], y[ntr:ntr+nv]), (X[ntr+nv:], y[ntr+nv:])

def train_one(model, Xtr, ytr, Xv, yv, outdir, epochs=40, batch_size=64, verbose=1):
    os.makedirs(outdir, exist_ok=True)
    return model.fit(
        Xtr, ytr,
        validation_data=(Xv, yv),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[
            EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6),
            ModelCheckpoint(os.path.join(outdir, "best.h5"), save_best_only=True, monitor="val_loss")
        ]
    )

def evaluate(model, Xte, yte):
    loss, mae, mape = model.evaluate(Xte, yte, verbose=0)
    return {"loss": float(loss), "mae": float(mae), "mape": float(mape)}

# ===================== Plots =====================
def plot_history(hist, path):
    plt.figure()
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def plot_pred_vs_actual(model, Xte, yte, path, step_index=0):
    pred = model.predict(Xte, verbose=0); y_true = yte
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        j = min(step_index, y_true.shape[1]-1)
        y_true = y_true[:, j]; pred = pred[:, j]
    else:
        y_true = y_true.flatten(); pred = pred.flatten()
    plt.figure()
    plt.plot(y_true, label="Actual")
    plt.plot(pred, label="Predicted")
    plt.xlabel("Time Step"); plt.ylabel("Scaled Close"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# ===================== Runner =====================
def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = fetch_prices(TICKER, start=START)
    rows = []

    if MODE == "uni_multistep":
        scaled_close, _, _ = minmax_scale(df["Close"].squeeze())
        X, y = make_windows_multistep_univariate(scaled_close, N_STEPS, K_STEPS)
        input_shape = (N_STEPS, 1); dense_units = K_STEPS

    elif MODE == "multi_single":
        feats = ["Open","High","Low","Close","Volume"]
        Xs, _, _ = minmax_scale_matrix(df[feats].values.astype("float32"))
        ys, _, _ = minmax_scale(df["Close"].squeeze())
        X, y = make_windows_multivariate_single_step(Xs, ys, N_STEPS, HORIZON)
        input_shape = (N_STEPS, X.shape[2]); dense_units = 1

    elif MODE == "multi_multistep":
        feats = ["Open","High","Low","Close","Volume"]
        Xs, _, _ = minmax_scale_matrix(df[feats].values.astype("float32"))
        ys, _, _ = minmax_scale(df["Close"].squeeze())
        X, y = make_windows_multivariate_multistep(Xs, ys, N_STEPS, K_STEPS)
        input_shape = (N_STEPS, X.shape[2]); dense_units = K_STEPS

    else:
        raise ValueError(f"Unknown MODE {MODE}")

    (Xtr, ytr), (Xv, yv), (Xte, yte) = split_time_series(X, y)

    for layer_type, num_layers, hidden_size, dropout, lr, batch_size, epochs in product(
        SEARCH_SPACE["layer_type"], SEARCH_SPACE["num_layers"], SEARCH_SPACE["hidden_size"],
        SEARCH_SPACE["dropout"], SEARCH_SPACE["lr"], SEARCH_SPACE["batch_size"], SEARCH_SPACE["epochs"]
    ):
        run_name = f"{MODE}_{layer_type}_L{num_layers}_H{hidden_size}_D{dropout}_B{batch_size}"
        outdir = os.path.join("models", run_name)

        model = build_sequence_model(
            input_shape=input_shape,
            layer_type=layer_type,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout,
            dense_units=dense_units,
            lr=lr
        )
        hist = train_one(model, Xtr, ytr, Xv, yv, outdir, epochs, batch_size)
        plot_history(hist, os.path.join("results", f"{run_name}_history.png"))
        plot_pred_vs_actual(model, Xte, yte, os.path.join("results", f"{run_name}_pred.png"), step_index=0)
        metrics = evaluate(model, Xte, yte)
        rows.append({
            "mode": MODE, "run": run_name, "layer_type": layer_type,
            "num_layers": num_layers, "hidden_size": hidden_size,
            "dropout": dropout, "batch_size": batch_size, "lr": lr,
            "epochs": epochs, "dense_units": dense_units, **metrics
        })

    res_df = pd.DataFrame(rows).sort_values(["mae", "loss"])
    res_path = os.path.join("results", f"task6_{MODE}_results.csv")
    res_df.to_csv(res_path, index=False)
    print(f"\nSaved results to: {res_path}\nTop configs by MAE:\n",
          res_df[["mode","run","layer_type","num_layers","hidden_size","mae","loss","mape"]].head(5))

if __name__ == "__main__":
    main()
