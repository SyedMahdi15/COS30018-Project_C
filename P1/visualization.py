import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf


# -----------------------------
# Column normalization helper
# -----------------------------
def _normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # If MultiIndex (e.g., ('Adj Close','AAPL')), flatten to first level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # Ensure string, lowercase, remove spaces
    df = df.rename(columns=str.lower)
    df.columns = [str(c).replace(" ", "") for c in df.columns]
    return df


# -----------------------------
# Helpers
# -----------------------------
def _validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_price_columns(df)

    need = {"open", "high", "low", "close"}
    if not set(df.columns).issuperset(need):
        raise ValueError(f"OHLC columns missing. Need: {need}. Found: {list(df.columns)}")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.sort_index()
    return df


def _resample_n_days(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 1:
        return df

    grp_id = np.floor(np.arange(len(df)) / n).astype(int)
    df_ = df.copy()
    df_["__grp"] = grp_id

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df_.columns:
        agg["volume"] = "sum"

    out = df_.groupby("__grp", as_index=False).agg(agg)
    # Use the last timestamp of each block as its index
    idx = df_.groupby("__grp").apply(lambda x: x.index[-1])
    out.index = pd.to_datetime(idx.values)
    out.drop(columns=["__grp"], errors="ignore", inplace=True)
    out = out.sort_index()
    return out
# -----------------------------
# Public APIs
# -----------------------------
def plot_candlestick(
    df: pd.DataFrame,
    n: int = 1,
    mav=(20, 50),
    volume: bool = True,
    style: str = "yahoo",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    df = _validate_ohlc(df)
    df_c = _resample_n_days(df, n)

    if title is None:
        title = f"Candlestick ({n}-day candles)" if n > 1 else "Candlestick"

    df_plot = df_c.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    fig, _axes = mpf.plot(
        df_plot,
        type="candle",
        style=style,
        mav=mav,
        volume=volume and ("Volume" in df_plot.columns),
        title=title,
        returnfig=True,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_boxplot(
    df: pd.DataFrame,
    column: str = "adjclose",
    window: int = 20,
    stride: int = 20,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    df = _normalize_price_columns(df)
    if column not in df.columns:
        raise ValueError(f"'{column}' not in DataFrame columns: {list(df.columns)}")

    s = df[column].copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna().sort_index()

    # collect window slices
    boxes, labels = [], []
    i = 0
    vals = s.values
    idx = s.index
    while i + window <= len(vals):
        boxes.append(vals[i : i + window])
        labels.append(idx[i + window - 1].strftime("%Y-%m-%d"))
        i += stride

    if not boxes:
        raise ValueError("Not enough data for the chosen window/stride.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(boxes, showfliers=False)

    ax.set_xticks(range(1, len(labels) + 1))
    # show a subset of labels to avoid clutter
    step = max(1, len(labels) // 8)
    ax.set_xticklabels(
        [labels[j] if (j % step == 0 or j == len(labels) - 1) else "" for j in range(len(labels))],
        rotation=45,
    )
    ax.set_ylabel(column.capitalize())
    ax.set_xlabel(f"Windows (window={window}, stride={stride})")
    ax.set_title(title or f"Boxplot of {column} over rolling windows")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------
# Demo / Manual run
# -----------------------------
if __name__ == "__main__":
    import yfinance as yf

    os.makedirs("csv-results", exist_ok=True)

    # Download AAPL for demo
    df = yf.download(
        "AAPL",
        start="2023-01-01",
        end="2025-08-01",
        auto_adjust=False,
        progress=False
    )
    if df is None or df.empty:
        raise RuntimeError("No data downloaded. Check internet, ticker, or dates.")

    # Normalize columns for our functions (handles MultiIndex safely)
    df = _normalize_price_columns(df)

    # 1) Candlestick chart with 3-day aggregated candles
    plot_candlestick(
        df[["open", "high", "low", "close", "volume"]],
        n=3,
        mav=(20, 50),
        volume=True,
        style="yahoo",
        title="AAPL Candles (n=3)",
        save_path="csv-results/candles_n3.png",
        show=True,   # set False to only save the image
    )
    # 2) Boxplot over rolling windows (20-day, non-overlapping)
    plot_boxplot(
        df,
        column="adjclose",
        window=20,
        stride=20,
        title="AAPL Adj Close Boxplot (20-day windows)",
        save_path="csv-results/box_adjclose_w20.png",
        show=True,   # set False to only save the image
    )

    print("Saved: csv-results/candles_n3.png and csv-results/box_adjclose_w20.png")
