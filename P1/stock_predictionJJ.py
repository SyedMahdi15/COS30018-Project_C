import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
import yfinance as yf
import matplotlib.pyplot as plt

np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

# Data / features
N_STEPS = 50                  # sequence/window length
LOOKUP_STEP = 15              # predict N days into the future
SCALE = True
SPLIT_BY_DATE = False         # True: split by time, False: random split
SHUFFLE = True
TEST_SIZE = 0.2
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

# Model
N_LAYERS = 2
CELL = LSTM
UNITS = 256
DROPOUT = 0.4
BIDIRECTIONAL = False

# Training
LOSS = tf.keras.losses.Huber()     # Keras 3-compatible loss object
LOSS_NAME = "huber"                # for file naming only
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 500
ES_PATIENCE = 10
LR_PATIENCE = 5
LR_FACTOR = 0.5

# Data range & cache
TICKER = "AAPL"
START_DATE = "2018-01-01"          # or None
END_DATE = None                    # e.g., "2025-08-01" or None for today
PERIOD = None                      # alternative to dates, e.g. "10y"
NA_STRATEGY = "ffill_bfill"        # "drop" or "ffill_bfill"
INTERVAL = "1d"

# Paths
DATE_NOW = time.strftime("%Y-%m-%d")
RESULTS_DIR = "results"
LOGS_DIR = "logs"
DATA_DIR = "data"
CSV_RESULTS_DIR = "csv-results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CSV_RESULTS_DIR, exist_ok=True)

if START_DATE or END_DATE:
    CACHE_NAME = f"{TICKER}_{START_DATE or 'NA'}_{END_DATE or 'NA'}.csv"
else:
    CACHE_NAME = f"{TICKER}_{PERIOD or '10y'}.csv"
CACHE_PATH = os.path.join(DATA_DIR, CACHE_NAME)

scale_str = f"sc-{int(SCALE)}"
shuffle_str = f"sh-{int(SHUFFLE)}"
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"

MODEL_NAME = (
    f"{DATE_NOW}_{TICKER}-{shuffle_str}-{scale_str}-{split_by_date_str}-"
    f"{LOSS_NAME}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-"
    f"layers-{N_LAYERS}-units-{UNITS}" + ("-b" if BIDIRECTIONAL else "")
)

def shuffle_in_unison(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def _download_prices_yf(
    ticker: str,
    start_date: str = None,   # "YYYY-MM-DD"
    end_date: str = None,     # "YYYY-MM-DD"
    period: str = None,       # e.g. "10y"
    interval: str = "1d",
    cache_path: str = None,
    use_local: bool = True
) -> pd.DataFrame:
    # 1) Try local cache
    if use_local and cache_path and os.path.isfile(cache_path):
        df = pd.read_csv(cache_path, index_col=0)
        # ensure datetime index
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
    else:
        # 2) Download with yfinance using either start/end or period
        if start_date or end_date:
            df = yf.download(
                ticker, start=start_date, end=end_date,
                interval=interval, auto_adjust=False, progress=False
            )
        else:
            df = yf.download(
                ticker, period=(period or "10y"),
                interval=interval, auto_adjust=False, progress=False
            )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned by yfinance for {ticker}. Check ticker/date/period.")

    # If yfinance returned a MultiIndex (e.g., ('Adj Close','AAPL')), collapse it
    if isinstance(df.columns, pd.MultiIndex):
        fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        lev0 = [c[0] for c in df.columns]
        lev1 = [c[1] for c in df.columns]
        score0 = sum(x in fields for x in lev0)
        score1 = sum(x in fields for x in lev1)
        df.columns = lev0 if score0 >= score1 else lev1

    # Normalize column names
    df = df.copy()
    df.columns = [str(c).lower().replace(" ", "") for c in df.columns]

    # Ensure required columns exist
    if "adjclose" not in df.columns:
        if "close" in df.columns:
            df["adjclose"] = df["close"]
        else:
            raise RuntimeError("Neither 'Adj Close' nor 'Close' present in data.")

    needed = {"open", "high", "low", "adjclose", "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Save cache (post-normalization) if requested
    if cache_path:
        df.to_csv(cache_path)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]

    return df

def load_data(
    ticker,
    n_steps=50,
    scale=True,
    shuffle=True,
    lookup_step=1,
    split_by_date=True,
    test_size=0.2,
    feature_columns=("adjclose", "volume", "open", "high", "low"),
    start_date: str = None,
    end_date: str = None,
    period: str = None,
    cache_path: str = None,
    use_local: bool = True,
    na_strategy: str = "drop"  # "drop" | "ffill_bfill"
):
    # 1) Load
    if isinstance(ticker, str):
        df = _download_prices_yf(
            ticker,
            start_date=start_date,
            end_date=end_date,
            period=period,
            interval=INTERVAL,
            cache_path=cache_path,
            use_local=use_local
        )
    elif isinstance(ticker, pd.DataFrame):
        df = ticker.copy()
        df.columns = [c.lower().replace(" ", "") for c in df.columns]
        if "adjclose" not in df.columns and "close" in df.columns:
            df["adjclose"] = df["close"]
    else:
        raise TypeError("ticker must be str or pd.DataFrame")

    # 2) Feature sanity
    for col in feature_columns:
        assert col in df.columns, f"'{col}' not in dataframe."

    # 3) NaN handling
    if na_strategy.lower() == "ffill_bfill":
        df = df.sort_index()
        df[list(feature_columns)] = df[list(feature_columns)].ffill().bfill()
    elif na_strategy.lower() == "drop":
        df = df.dropna(subset=list(feature_columns))
    else:
        raise ValueError("na_strategy must be 'drop' or 'ffill_bfill'")

    # 4) Date column
    if "date" not in df.columns:
        df["date"] = df.index

    # 5) Scaling
    result = {"df": df.copy()}
    column_scaler = {}
    if scale:
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler

    # 6) Label (future)
    df["future"] = df["adjclose"].shift(-lookup_step)
    last_sequence_raw = np.array(df[list(feature_columns)].tail(lookup_step))
    df = df.dropna(subset=["future"])

    # 7) Build sequences
    sequence_data = []
    window = deque(maxlen=n_steps)
    for row, target in zip(df[list(feature_columns) + ["date"]].values, df["future"].values):
        window.append(row)
        if len(window) == n_steps:
            sequence_data.append([np.array(window), target])

    if not sequence_data:
        raise ValueError("Not enough data after processing! Lower n_steps/lookup_step or extend date range.")

    # Last sequence for future inference
    last_feats_only = [r[:len(feature_columns)] for r in list(window)]
    last_sequence = np.array(last_feats_only + list(last_sequence_raw)).astype(np.float32)
    result["last_sequence"] = last_sequence

    # 8) Split X, y
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    X = np.array(X)
    y = np.array(y)

    if split_by_date:
        cut = int((1 - test_size) * len(X))
        X_train, y_train = X[:cut], y[:cut]
        X_test, y_test = X[cut:], y[cut:]
        if shuffle:
            shuffle_in_unison(X_train, y_train)
            shuffle_in_unison(X_test, y_test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    # 9) Build test_df from dates in last column
    dates = X_test[:, -1, -1]
    test_df = result["df"].loc[dates]
    test_df = test_df[~test_df.index.duplicated(keep="first")]

    # Strip the date column from tensors
    X_train = X_train[:, :, :len(feature_columns)].astype(np.float32)
    X_test = X_test[:, :, :len(feature_columns)].astype(np.float32)

    result.update(dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, test_df=test_df))
    return result

def create_model(
    sequence_length,
    n_features,
    units=256,
    cell=LSTM,
    n_layers=2,
    dropout=0.3,
    loss="mean_absolute_error",
    optimizer="rmsprop",
    bidirectional=False,
):
    model = Sequential()
    model.add(Input(shape=(sequence_length, n_features)))  # explicit Input

    for i in range(n_layers):
        return_sequences = i < n_layers - 1
        if bidirectional:
            model.add(Bidirectional(cell(units, return_sequences=return_sequences)))
        else:
            model.add(cell(units, return_sequences=return_sequences))
        model.add(Dropout(dropout))

    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

def plot_graph(test_df, lookup_step):
    plt.plot(test_df[f"true_adjclose_{lookup_step}"])
    plt.plot(test_df[f"adjclose_{lookup_step}"])
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

def get_final_df(model, data, scale=True, lookup_step=1):
    buy_profit = lambda current, pred_future, true_future: (true_future - current) if pred_future > current else 0
    sell_profit = lambda current, pred_future, true_future: (current - true_future) if pred_future < current else 0

    X_test = data["X_test"]
    y_test = data["y_test"]

    # Predict
    y_pred = model.predict(X_test, verbose=0)

    if scale:
        inv = data["column_scaler"]["adjclose"].inverse_transform
        y_test = np.squeeze(inv(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(inv(y_pred))

    test_df = data["test_df"].copy()
    test_df[f"adjclose_{lookup_step}"] = y_pred
    test_df[f"true_adjclose_{lookup_step}"] = y_test
    test_df.sort_index(inplace=True)

    final_df = test_df
    final_df["buy_profit"] = list(
        map(buy_profit, final_df["adjclose"], final_df[f"adjclose_{lookup_step}"], final_df[f"true_adjclose_{lookup_step}"])
    )
    final_df["sell_profit"] = list(
        map(sell_profit, final_df["adjclose"], final_df[f"adjclose_{lookup_step}"], final_df[f"true_adjclose_{lookup_step}"])
    )
    return final_df

def predict_future(model, data, n_steps, scale=True):
    last_sequence = data["last_sequence"][-n_steps:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    prediction = model.predict(last_sequence, verbose=0)
    if scale:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print(f"Loading data for ticker: {TICKER}")

    data = load_data(
        TICKER,
        N_STEPS,
        scale=SCALE,
        split_by_date=SPLIT_BY_DATE,
        shuffle=SHUFFLE,
        lookup_step=LOOKUP_STEP,
        test_size=TEST_SIZE,
        feature_columns=FEATURE_COLUMNS,
        start_date=START_DATE,
        end_date=END_DATE,
        period=PERIOD,
        cache_path=CACHE_PATH,
        use_local=True,
        na_strategy=NA_STRATEGY
    )

    # Save raw df snapshot for reference
    raw_csv_path = os.path.join(DATA_DIR, f"{TICKER}_{DATE_NOW}.csv")
    data["df"].to_csv(raw_csv_path)
    print(f"Data snapshot saved to {raw_csv_path}")

    # Build model
    model = create_model(
        N_STEPS,
        len(FEATURE_COLUMNS),
        loss=LOSS,
        units=UNITS,
        cell=CELL,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        optimizer=OPTIMIZER,
        bidirectional=BIDIRECTIONAL,
    )

    # Callbacks (Keras 3: .weights.h5 for weights-only)
    checkpointer = ModelCheckpoint(
        os.path.join(RESULTS_DIR, MODEL_NAME + ".weights.h5"),
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    )
    tensorboard = TensorBoard(log_dir=os.path.join(LOGS_DIR, MODEL_NAME))
    early_stopping = EarlyStopping(monitor="val_loss", patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=LR_FACTOR, patience=LR_PATIENCE, verbose=1)

    # Train
    history = model.fit(
        data["X_train"],
        data["y_train"],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(data["X_test"], data["y_test"]),
        callbacks=[checkpointer, tensorboard, early_stopping, reduce_lr],
        verbose=1,
    )

    # Load best weights (if not already restored by EarlyStopping)
    best_weights = os.path.join(RESULTS_DIR, MODEL_NAME + ".weights.h5")
    if os.path.exists(best_weights):
        model.load_weights(best_weights)

    # Evaluate
    loss_val, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    if SCALE:
        mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae

    # Final df + metrics
    final_df = get_final_df(model, data, scale=SCALE, lookup_step=LOOKUP_STEP)
    future_price = predict_future(model, data, N_STEPS, scale=SCALE)

    accuracy_score = (len(final_df[final_df["sell_profit"] > 0]) + len(final_df[final_df["buy_profit"] > 0])) / len(final_df)
    total_buy_profit = final_df["buy_profit"].sum()
    total_sell_profit = final_df["sell_profit"].sum()
    total_profit = total_buy_profit + total_sell_profit
    profit_per_trade = total_profit / len(final_df)

    print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
    print(f"{LOSS_NAME} loss:", loss_val)
    print("Mean Absolute Error:", mean_absolute_error)
    print("Accuracy score:", accuracy_score)
    print("Total buy profit:", total_buy_profit)
    print("Total sell profit:", total_sell_profit)
    print("Total profit:", total_profit)
    print("Profit per trade:", profit_per_trade)

    # Plot Actual vs Predicted
    plot_graph(final_df, LOOKUP_STEP)

    # Save final CSV
    csv_filename = os.path.join(CSV_RESULTS_DIR, MODEL_NAME + ".csv")
    final_df.to_csv(csv_filename, index=True)
    print(f"Saved final results to {csv_filename}")
