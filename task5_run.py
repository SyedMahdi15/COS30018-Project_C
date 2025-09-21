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
HORIZON = 1          # used for single-step with horizon offset
K_STEPS = 5          # used for multistep (predict next K days)

# Choose one: "uni_multistep", "multi_single", "multi_multistep"
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
    """Remove ticker prefix/suffix with common separators; lowercase & underscore."""
    s = str(col).strip()
    s = s.replace(" ", "_")
    cl = s.lower()
    tl = ticker.lower()
    for sep in ("_", "-", ".", " "):
        if cl.endswith(sep + tl):
            cl = cl[: -(len(tl) + 1)]
        if cl.startswith(tl + sep):
            cl = cl[len(tl) + 1 :]
    while "__" in cl:
        cl = cl.replace("__", "_")
    return cl

def _normalize_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels[0]) == 1:
            df.columns = df.columns.droplevel(0)
        else:
            df.columns = ["_".join([str(x) for x in tup if x is not None]).strip() for tup in df.columns]
    new_cols = {}
    for c in df.columns:
        base = _strip_ticker_from_col(c, ticker)
        new_cols[c] = base
    df = df.rename(columns=new_cols)
    return df

def _sanitize_ohlc(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Clean & standardize OHLC data (case-insensitive, tolerant of ticker-suffixed names)."""
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

    open_c  = pick("open")
    high_c  = pick("high")
    low_c   = pick("low")
    close_c = pick("close") or pick("adj_close")
    vol_c   = pick("volume")

    selected = [c for c in (open_c, high_c, low_c, close_c, vol_c) if c]
    if not selected or close_c is None:
        raise RuntimeError(f"No OHLC columns found. Got columns: {list(df.columns)}")

    df = df[selected].copy()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[close_c])
    if df.empty:
        raise RuntimeError("All rows dropped while sanitizing OHLC (no valid Close).")

    rename_map = {}
    if open_c:  rename_map[open_c]  = "Open"
    if high_c:  rename_map[high_c]  = "High"
    if low_c:   rename_map[low_c]   = "Low"
    if close_c: rename_map[close_c] = "Close"
    if vol_c:   rename_map[vol_c]   = "Volume"
    df = df.rename(columns=rename_map)

    return df

# ===================== Fetch prices =====================
def fetch_prices(ticker="AAPL", start="2018-01-01", end=None, use_cache=True, retries=5):
    """Fetch prices from cache, then Yahoo (download), then Yahoo (Ticker.history), then Stooq."""
    cache_file = _cache_path(ticker)

    if use_cache and os.path.exists(cache_file):
        for opts in ({"parse_dates": ["Date"], "index_col": "Date"},
                     {"parse_dates": [0], "index_col": 0},
                     {}):
            try:
                df = pd.read_csv(cache_file, **opts, on_bad_lines="skip")
                df = _sanitize_ohlc(df, ticker)
                return df
            except Exception:
                pass

    last_err = None
    for i in range(retries):
        try:
            df = yf.download(
                ticker, start=start, end=end,
                interval="1d", auto_adjust=True,
                progress=False, threads=False
            )
            if df is not None and not df.empty:
                df = _sanitize_ohlc(df, ticker)
                df.to_csv(cache_file, index_label="Date")
                return df
            raise RuntimeError("Empty DataFrame from yfinance (download).")
        except Exception as e:
            last_err = e
            base = [2, 4, 8, 12, 16][min(i, 4)]
            time.sleep(base * random.uniform(0.7, 1.3))

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, end=end, auto_adjust=True)
        if hist is not None and not hist.empty:
            df = _sanitize_ohlc(hist, ticker)
            df.to_csv(cache_file, index_label="Date")
            return df
        raise RuntimeError("Empty DataFrame from yfinance (Ticker.history).")
    except Exception as e_hist:
        last_err = last_err or e_hist
        last_err = f"{last_err} | history(): {e_hist}"

    try:
        stq = pd.read_csv(f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d")
        if stq is None or stq.empty or stq.shape[1] < 2:
            raise RuntimeError("Unexpected Stooq response (empty or 1 column).")
        stq = _sanitize_ohlc(stq, ticker)
        stq.to_csv(cache_file, index_label="Date")
        return stq
    except Exception as e2:
        raise RuntimeError(f"Failed to fetch {ticker}. Yahoo error: {last_err}; Stooq error: {e2}")

# ===================== Scaling helpers =====================
def minmax_scale(series: pd.Series):
    """Scale a single series to [0,1]; returns scaled series and (min,max)."""
    s = pd.Series(series).squeeze()
    s = pd.to_numeric(s, errors="coerce").dropna().astype("float32")
    if s.empty:
        raise ValueError("minmax_scale: no numeric values found.")
    smin, smax = float(s.min()), float(s.max())
    denom = smax - smin if smax != smin else 1.0
    scaled = (pd.to_numeric(pd.Series(series).squeeze(), errors="coerce").astype("float32") - smin) / denom
    return scaled.astype("float32"), smin, smax

def minmax_scale_matrix(X: np.ndarray):
    """Scale each column of X independently to [0,1]. Returns scaled, mins, maxs."""
    X = np.asarray(X, dtype="float32")
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    denom = np.where((maxs - mins) == 0.0, 1.0, (maxs - mins))
    Xs = (X - mins) / denom
    return Xs.astype("float32"), mins.astype("float32"), maxs.astype("float32")

# ===================== Windowing (Sequences) =====================
def make_sequences(series: pd.Series, n_steps=60, horizon=1):
    """Univariate single-output: past n_steps Close -> predict Close at t+horizon."""
    vals = pd.Series(series).squeeze().values.astype("float32")
    if len(vals) < n_steps + horizon:
        raise ValueError(f"Not enough data to build sequences (have {len(vals)}, need {n_steps+horizon}).")
    X, y = [], []
    for i in range(n_steps, len(vals) - horizon + 1):
        X.append(vals[i - n_steps:i])
        y.append(vals[i + horizon - 1])
    return np.array(X)[:, :, None], np.array(y)[:, None]

def make_windows_multistep_univariate(close_scaled: np.ndarray, n_steps=60, k=5):
    """Univariate multistep: past n_steps Close -> next k Closes (vector)."""
    close_scaled = np.asarray(close_scaled).reshape(-1, 1)
    vals = close_scaled[:, 0]
    if len(vals) < n_steps + k:
        raise ValueError(f"Not enough data: have {len(vals)}, need {n_steps+k}.")
    Xs, ys = [], []
    for i in range(n_steps, len(vals) - k + 1):
        Xs.append(vals[i - n_steps:i])
        ys.append(vals[i:i + k])
    X = np.array(Xs).reshape(-1, n_steps, 1)
    y = np.array(ys)
    return X, y

def make_windows_multivariate_single_step(X_scaled: np.ndarray, y_scaled: np.ndarray, n_steps=60, horizon=1):
    """Multivariate single-step: past n_steps features -> Close at t+horizon."""
    X_scaled = np.asarray(X_scaled, dtype="float32")
    y_scaled = np.asarray(y_scaled, dtype="float32").reshape(-1, 1)
    N = len(X_scaled)
    if N < n_steps + horizon:
        raise ValueError(f"Not enough data: have {N}, need {n_steps+horizon}.")
    Xs, ys = [], []
    for i in range(n_steps, N - horizon + 1):
        Xs.append(X_scaled[i - n_steps:i, :])
        ys.append(y_scaled[i + horizon - 1, 0])
    X = np.array(Xs)
    y = np.array(ys).reshape(-1, 1)
    return X, y

def make_windows_multivariate_multistep(X_scaled: np.ndarray, y_scaled: np.ndarray, n_steps=60, k=5):
    """Multivariate multistep: past n_steps features -> next k Closes (vector)."""
    X_scaled = np.asarray(X_scaled, dtype="float32")
    y_scaled = np.asarray(y_scaled, dtype="float32").reshape(-1, 1)
    N = len(X_scaled)
    if N < n_steps + k:
        raise ValueError(f"Not enough data: have {N}, need {n_steps+k}.")
    Xs, ys = [], []
    for i in range(n_steps, N - k + 1):
        Xs.append(X_scaled[i - n_steps:i, :])
        ys.append(y_scaled[i:i + k, 0])
    X = np.array(Xs)
    y = np.array(ys)
    return X, y

# ===================== Model =====================
def build_sequence_model(input_shape,
                         layer_type="LSTM",
                         num_layers=2,
                         hidden_size=128,
                         dropout=0.2,
                         bidirectional=False,
                         dense_units=1,
                         lr=1e-3):
    Layer = {"LSTM": layers.LSTM,
             "GRU": layers.GRU,
             "RNN": layers.SimpleRNN}[layer_type.upper()]
    model = models.Sequential(name=f"{layer_type}_L{num_layers}_H{hidden_size}")
    for i in range(num_layers):
        return_seq = i < num_layers - 1
        cell = Layer(hidden_size, return_sequences=return_seq)
        if bidirectional:
            cell = layers.Bidirectional(cell)
        if i == 0:
            model.add(cell.__class__(**cell.get_config(), input_shape=input_shape))
        else:
            model.add(cell)
        if dropout and return_seq:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dense_units))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="mse", metrics=["mae", "mape"])
    return model

# ===================== Train & Eval =====================
def split_time_series(X, y, train_ratio=0.7, val_ratio=0.15):
    N = len(X)
    n_tr = int(N * train_ratio)
    n_v = int(N * val_ratio)
    return (X[:n_tr], y[:n_tr]), (X[n_tr:n_tr+n_v], y[n_tr:n_tr+n_v]), (X[n_tr+n_v:], y[n_tr+n_v:])

def train_one(model, Xtr, ytr, Xv, yv, outdir, epochs=40, batch_size=64, verbose=1):
    os.makedirs(outdir, exist_ok=True)
    cbs = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6),
        ModelCheckpoint(os.path.join(outdir, "best.h5"),
                        save_best_only=True, monitor="val_loss"),
    ]
    return model.fit(Xtr, ytr,
                     validation_data=(Xv, yv),
                     epochs=epochs,
                     batch_size=batch_size,
                     verbose=verbose,
                     callbacks=cbs)

def evaluate(model, Xte, yte):
    loss, mae, mape = model.evaluate(Xte, yte, verbose=0)
    return {"loss": float(loss), "mae": float(mae), "mape": float(mape)}

# ===================== Plots =====================
def plot_history(hist, path):
    plt.figure()
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_pred_vs_actual(model, Xte, yte, path, step_index=0):
    """Handles single-output (shape (N,1)) and multistep (shape (N,k))."""
    pred = model.predict(Xte, verbose=0)
    y_true = yte
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        j = max(0, min(step_index, y_true.shape[1]-1))
        y_true = y_true[:, j]
        pred = pred[:, j]
    else:
        y_true = y_true.flatten()
        pred = pred.flatten()
    plt.figure()
    plt.plot(y_true, label="Actual")
    plt.plot(pred, label="Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Scaled Close")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ===================== Runner =====================
def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = fetch_prices(TICKER, start=START)

    rows = []

    if MODE == "uni_multistep":
        scaled_close, smin, smax = minmax_scale(df["Close"].squeeze())
        X, y = make_windows_multistep_univariate(scaled_close, n_steps=N_STEPS, k=K_STEPS)
        input_shape = (N_STEPS, 1)
        dense_units = K_STEPS

    elif MODE == "multi_single":
        feats = ["Open", "High", "Low", "Close", "Volume"]
        X_raw = df[feats].values.astype("float32")
        y_raw = df["Close"].values.astype("float32").reshape(-1, 1)
        X_scaled, Xmins, Xmaxs = minmax_scale_matrix(X_raw)
        y_scaled, ymin, ymax = minmax_scale(df["Close"].squeeze())
        X, y = make_windows_multivariate_single_step(X_scaled, y_scaled, n_steps=N_STEPS, horizon=HORIZON)
        input_shape = (N_STEPS, X.shape[2])
        dense_units = 1

    elif MODE == "multi_multistep":
        feats = ["Open", "High", "Low", "Close", "Volume"]
        X_raw = df[feats].values.astype("float32")
        y_raw = df["Close"].values.astype("float32").reshape(-1, 1)
        X_scaled, Xmins, Xmaxs = minmax_scale_matrix(X_raw)
        y_scaled, ymin, ymax = minmax_scale(df["Close"].squeeze())
        X, y = make_windows_multivariate_multistep(X_scaled, y_scaled, n_steps=N_STEPS, k=K_STEPS)
        input_shape = (N_STEPS, X.shape[2])
        dense_units = K_STEPS

    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    (Xtr, ytr), (Xv, yv), (Xte, yte) = split_time_series(X, y)

    for layer_type, num_layers, hidden_size, dropout, lr, batch_size, epochs in product(
        SEARCH_SPACE["layer_type"],
        SEARCH_SPACE["num_layers"],
        SEARCH_SPACE["hidden_size"],
        SEARCH_SPACE["dropout"],
        SEARCH_SPACE["lr"],
        SEARCH_SPACE["batch_size"],
        SEARCH_SPACE["epochs"]
    ):
        run_name = f"{MODE}_{layer_type}_L{num_layers}_H{hidden_size}_D{dropout}_B{batch_size}"
        outdir = os.path.join("models", run_name)

        model = build_sequence_model(
            input_shape=input_shape,
            layer_type=layer_type,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout,
            lr=lr,
            dense_units=dense_units
        )
        hist = train_one(model, Xtr, ytr, Xv, yv, outdir,
                         epochs=epochs, batch_size=batch_size)
        plot_history(hist, os.path.join("results", f"{run_name}_history.png"))
        plot_pred_vs_actual(model, Xte, yte, os.path.join("results", f"{run_name}_pred.png"), step_index=0)
        metrics = evaluate(model, Xte, yte)
        rows.append({
            "mode": MODE,
            "run": run_name,
            "layer_type": layer_type,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "dense_units": dense_units,
            **metrics
        })

    res_df = pd.DataFrame(rows).sort_values(["mae", "loss"])
    os.makedirs("results", exist_ok=True)
    res_path = os.path.join("results", f"task4_{MODE}_results.csv")
    res_df.to_csv(res_path, index=False)

    print(f"\nSaved results to: {res_path}")
    print("\nTop configs by MAE:")
    print(res_df[["mode","run","layer_type","num_layers","hidden_size","mae","loss","mape"]].head(5))

if __name__ == "__main__":
    main()
