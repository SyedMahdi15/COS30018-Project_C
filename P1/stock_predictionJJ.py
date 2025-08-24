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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import yfinance as yf
import matplotlib.pyplot as plt

# -----------------------------
# Reproducibility
# -----------------------------
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

# -----------------------------
# Hyperparams / Config
# -----------------------------
N_STEPS = 50                 # sequence length
LOOKUP_STEP = 15             # predict N days into the future
SCALE = True
SPLIT_BY_DATE = False
SHUFFLE = True
TEST_SIZE = 0.2
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

N_LAYERS = 2
CELL = LSTM
UNITS = 256
DROPOUT = 0.4
BIDIRECTIONAL = False

LOSS = "huber_loss"          # or "mae"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 500

ticker = "AMZN"
date_now = time.strftime("%Y-%m-%d")

scale_str = f"sc-{int(SCALE)}"
shuffle_str = f"sh-{int(SHUFFLE)}"
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"

model_name = (
    f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-"
    f"{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-"
    f"layers-{N_LAYERS}-units-{UNITS}"
)
if BIDIRECTIONAL:
    model_name += "-b"

# Paths
RESULTS_DIR = "results"
LOGS_DIR = "logs"
DATA_DIR = "data"
CSV_RESULTS_DIR = "csv-results"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CSV_RESULTS_DIR, exist_ok=True)

ticker_data_filename = os.path.join(DATA_DIR, f"{ticker}_{date_now}.csv")


# -----------------------------
# Utils
# -----------------------------
def shuffle_in_unison(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def _download_prices_yf(ticker: str, period="10y", interval="1d") -> pd.DataFrame:
    """
    Download OHLCV from Yahoo via yfinance and return a single-level,
    lowercase/space-free column set: open, high, low, close, adjclose, volume
    """
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned by yfinance for {ticker}. Try another ticker or longer period.")

    # If yfinance returned a MultiIndex (e.g., ('Adj Close','AAPL')), collapse it.
    if isinstance(df.columns, pd.MultiIndex):
        # choose the level that looks like field names (Open/High/Low/Close/Adj Close/Volume)
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
            raise RuntimeError("Neither 'Adj Close' nor 'Close' available in yfinance data.")

    needed = {"open", "high", "low", "adjclose", "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in downloaded data: {missing}")

    # Drop NaNs on required columns
    df = df.dropna(subset=list(needed))

    # Make sure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna(axis=0, subset=[df.index.name])

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
):
    """
    Loads data via yfinance, normalizes, scales (optional), builds sequences,
    and returns dict with train/test splits, scalers, original df, etc.
    """
    # 1) Load
    if isinstance(ticker, str):
        df = _download_prices_yf(ticker)
    elif isinstance(ticker, pd.DataFrame):
        df = ticker.copy()
        # Ensure normalized column names
        df.columns = [c.lower().replace(" ", "") for c in df.columns]
        if "adjclose" not in df.columns and "close" in df.columns:
            df["adjclose"] = df["close"]
    else:
        raise TypeError("ticker can be a str or a pd.DataFrame.")

    # 2) Sanity on features
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    # 3) Add a date column from index if missing
    if "date" not in df.columns:
        df["date"] = df.index

    # 4) Optional scaling
    result = {}
    result["df"] = df.copy()
    column_scaler = {}

    if scale:
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler

    # 5) Future/label column
    df["future"] = df["adjclose"].shift(-lookup_step)

    # Keep the last `lookup_step` rows' features (before dropping NaNs)
    last_sequence_raw = np.array(df[list(feature_columns)].tail(lookup_step))

    # Drop NaNs created by shift
    df.dropna(inplace=True)

    # 6) Build sequences
    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[list(feature_columns) + ["date"]].values, df["future"].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # If not enough data after processing:
    if len(sequence_data) == 0:
        raise ValueError("Not enough data after processing! Try lowering n_steps or lookup_step.")

    last_seq_features_only = [row[: len(feature_columns)] for row in list(sequences)]
    last_sequence = np.array(last_seq_features_only + list(last_sequence_raw)).astype(np.float32)
    result["last_sequence"] = last_sequence

    # 7) Split into X, y
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    X = np.array(X)
    y = np.array(y)

    # 8) Train/test split
    if split_by_date:
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        if shuffle:
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle
        )
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = (
            X_train,
            X_test,
            y_train,
            y_test,
        )

    # 9) Build test_df from dates located in the last column of X_test
    dates = result["X_test"][:, -1, -1]
    test_df = result["df"].loc[dates]
    test_df = test_df[~test_df.index.duplicated(keep="first")]
    # Keep only features in X arrays (strip the date column)
    result["X_train"] = result["X_train"][:, :, : len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, : len(feature_columns)].astype(np.float32)
    result["test_df"] = test_df

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
    for i in range(n_layers):
        return_sequences = i < n_layers - 1
        if i == 0:
            if bidirectional:
                model.add(
                    Bidirectional(
                        cell(units, return_sequences=return_sequences),
                        input_shape=(sequence_length, n_features),
                    )
                )
            else:
                model.add(cell(units, return_sequences=return_sequences, input_shape=(sequence_length, n_features)))
        else:
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
    # Load data
    data = load_data(
        ticker,
        N_STEPS,
        scale=SCALE,
        split_by_date=SPLIT_BY_DATE,
        shuffle=SHUFFLE,
        lookup_step=LOOKUP_STEP,
        test_size=TEST_SIZE,
        feature_columns=FEATURE_COLUMNS,
    )

    # Save raw df snapshot
    data["df"].to_csv(ticker_data_filename)

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

    # Callbacks
    checkpointer = ModelCheckpoint(
        os.path.join(RESULTS_DIR, model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1
    )
    tensorboard = TensorBoard(log_dir=os.path.join(LOGS_DIR, model_name))

    # Train
    history = model.fit(
        data["X_train"],
        data["y_train"],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(data["X_test"], data["y_test"]),
        callbacks=[checkpointer, tensorboard],
        verbose=1,
    )

    # Load best weights
    model_path = os.path.join(RESULTS_DIR, model_name) + ".h5"
    if os.path.exists(model_path):
        model.load_weights(model_path)

    # Evaluate
    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
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
    print(f"{LOSS} loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    print("Accuracy score:", accuracy_score)
    print("Total buy profit:", total_buy_profit)
    print("Total sell profit:", total_sell_profit)
    print("Total profit:", total_profit)
    print("Profit per trade:", profit_per_trade)

    # Plot
    plot_graph(final_df, LOOKUP_STEP)

    # Save final csv
    csv_filename = os.path.join(CSV_RESULTS_DIR, model_name + ".csv")
    final_df.to_csv(csv_filename, index=True)
    print(f"Saved final results to {csv_filename}")
