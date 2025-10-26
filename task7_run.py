# task7_sentiment_model.py
# ------------------------------------------------------------
# COS30018 - Task 7 Extension
# Sentiment-Based Stock Price Movement Prediction (Classification)
# ------------------------------------------------------------

import os
import time
import math
import random
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from datetime import datetime
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# Silence warnings for cleaner logs
warnings.filterwarnings("ignore")

# ===================== Paths & Constants =====================
DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TICKER = "AAPL"
START = "2018-01-01"
END = None                   # use latest
SEED = 2195                  # deterministic
TEST_SIZE = 0.15             # 15% test (chronological split)
VAL_SIZE = 0.15              # 15% validation (chronological split)
USE_FINBERT = False          # Optional: requires transformers + internet for first download
NEWS_CSV = os.path.join(DATA_DIR, "news_headlines.csv")  # optional input file: date,headline

# Small hyperparam search over models
SEARCH_SPACE = [
    {"model": "logreg", "C": 1.0, "penalty": "l2"},
    {"model": "logreg", "C": 0.5, "penalty": "l2"},
    {"model": "rf",     "n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
    {"model": "rf",     "n_estimators": 500, "max_depth": 8,    "min_samples_leaf": 1},
]

random.seed(SEED)
np.random.seed(SEED)

# ===================== Utilities =====================
def _cache_path_prices(ticker: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker.upper()}.csv")

def _normalize_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels[0]) == 1:
            df.columns = df.columns.droplevel(0)
        else:
            df.columns = ["_".join([str(x) for x in tup if x is not None]).strip() for tup in df.columns]
    # strip ticker-like suffix/prefix
    def _strip(c):
        s = str(c).strip().replace(" ", "_").lower()
        tl = ticker.lower()
        for sep in ("_", "-", ".", " "):
            if s.endswith(sep + tl):
                s = s[:-(len(tl) + 1)]
            if s.startswith(tl + sep):
                s = s[len(tl) + 1:]
        while "__" in s:
            s = s.replace("__", "_")
        return s
    return df.rename(columns={c: _strip(c) for c in df.columns})

def _sanitize_ohlc(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError("Empty DF in _sanitize_ohlc")
    df = _normalize_columns(df, ticker)

    if not isinstance(df.index, pd.DatetimeIndex):
        # try to infer a date column
        candidates = [c for c in df.columns if c in ("date", "timestamp")]
        if candidates:
            dcol = candidates[0]
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
            df = df.set_index(dcol)
        else:
            df = df.reset_index()
            first = df.columns[0]
            df[first] = pd.to_datetime(df[first], errors="coerce")
            df = df.set_index(first)

    df = df[~df.index.isna()].sort_index()

    def pick(token):
        token = token.replace(" ", "_").lower()
        for c in df.columns:
            low = c.lower()
            if low == token or low.endswith("_"+token) or low.startswith(token+"_") or token in low:
                return c
        return None

    open_c  = pick("open")
    high_c  = pick("high")
    low_c   = pick("low")
    close_c = pick("close") or pick("adj_close")
    vol_c   = pick("volume")

    need = [open_c, high_c, low_c, close_c]
    if any([c is None for c in need]):
        raise RuntimeError(f"Missing essential OHLC columns. Got {list(df.columns)}")

    keep = [c for c in [open_c, high_c, low_c, close_c, vol_c] if c]
    df = df[keep].copy().apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[close_c])

    rename = {open_c: "Open", high_c: "High", low_c: "Low", close_c: "Close"}
    if vol_c: rename[vol_c] = "Volume"
    return df.rename(columns=rename)

def fetch_prices(ticker="AAPL", start="2018-01-01", end=None,
                 use_cache=True, retries=4) -> pd.DataFrame:
    """
    Fetch OHLCV daily prices with caching and fallbacks.
    Returns DataFrame with Date index and columns: Open, High, Low, Close, Volume
    """
    cache_file = _cache_path_prices(ticker)
    # 0) Cache
    if use_cache and os.path.exists(cache_file):
        for opts in ({"parse_dates": ["Date"], "index_col": "Date"},
                     {"parse_dates": [0], "index_col": 0}, {}):
            try:
                df = pd.read_csv(cache_file, **opts, on_bad_lines="skip")
                df = _sanitize_ohlc(df, ticker)
                return df
            except Exception:
                pass

    # 1) yfinance download
    last_err = None
    for i in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end,
                             interval="1d", auto_adjust=True,
                             progress=False, threads=False)
            if df is not None and not df.empty:
                df = _sanitize_ohlc(df, ticker)
                df.to_csv(cache_file, index_label="Date")
                return df
            raise RuntimeError("Empty DataFrame from yfinance.")
        except Exception as e:
            last_err = e
            time.sleep([2, 4, 8, 12][min(i, 3)] * random.uniform(0.7, 1.3))

    # 2) Ticker.history fallback
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, end=end, auto_adjust=True)
        if hist is not None and not hist.empty:
            df = _sanitize_ohlc(hist, ticker)
            df.to_csv(cache_file, index_label="Date")
            return df
        raise RuntimeError("Empty history via yfinance.Ticker().history().")
    except Exception as e_hist:
        last_err = last_err or e_hist

    # 3) Stooq fallback
    try:
        stq = pd.read_csv(f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d")
        if stq is None or stq.empty or stq.shape[1] < 2:
            raise RuntimeError("Unexpected Stooq response.")
        stq = _sanitize_ohlc(stq, ticker)
        stq.to_csv(cache_file, index_label="Date")
        return stq
    except Exception as e2:
        raise RuntimeError(f"Failed to fetch {ticker}. Last error: {last_err}; Stooq: {e2}")

# ===================== Sentiment =====================
def load_or_create_news_csv(path: str) -> pd.DataFrame:
    """
    Expects a CSV with columns: date, headline
    If not present, creates a tiny synthetic demo file to keep pipeline runnable.
    """
    if not os.path.exists(path):
        demo = pd.DataFrame({
            "date": [
                "2021-10-18","2021-10-19","2021-10-20","2021-10-21","2021-10-22",
                "2021-10-25","2021-10-26","2021-10-27","2021-10-28","2021-10-29"
            ],
            "headline": [
                "Apple unveils strong product line; analysts optimistic",
                "Concerns rise over supply chain; iPhone delays expected",
                "Services revenue growth beats expectations",
                "Antitrust lawsuit chatter intensifies in EU market",
                "Mixed signals from China demand, investors cautious",
                "New MacBook Pro reviews are overwhelmingly positive",
                "Reports suggest chip shortages easing this quarter",
                "Apple faces patent dispute, potential legal risk",
                "Holiday season outlook revised upward by bank analysts",
                "Production targets adjusted; long-term outlook steady"
            ]
        })
        demo.to_csv(path, index=False)
    news = pd.read_csv(path)
    news["date"] = pd.to_datetime(news["date"], errors="coerce").dt.date
    news = news.dropna(subset=["date", "headline"])
    return news

def compute_daily_sentiment(news_df: pd.DataFrame, use_finbert: bool = False) -> pd.DataFrame:
    """
    Returns DataFrame with index as Date and column 'sentiment' (daily mean compound score).
    Uses VADER by default; optional FinBERT (requires transformers).
    Includes robust handling for missing NLTK 'vader_lexicon'.
    """
    if use_finbert:
        try:
            from transformers import pipeline
            model_name = "ProsusAI/finbert"
            clf = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, truncation=True)
            def score_fn(text: str) -> float:
                out = clf(str(text)[:512])[0]
                label = out["label"].lower()
                if "pos" in label: return 1.0
                if "neg" in label: return -1.0
                return 0.0
        except Exception:
            # Fallback to VADER if FinBERT unavailable
            use_finbert = False

    if not use_finbert:
        import os
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        # Ensure a user-writable nltk_data path to avoid permission issues
        user_nltk_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
        if user_nltk_dir not in nltk.data.path:
            nltk.data.path.append(user_nltk_dir)

        # Try to instantiate; if lexicon missing, download and retry
        try:
            sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", download_dir=user_nltk_dir, quiet=True)
            if user_nltk_dir not in nltk.data.path:
                nltk.data.path.append(user_nltk_dir)
            sia = SentimentIntensityAnalyzer()

        def score_fn(text: str) -> float:
            return float(sia.polarity_scores(str(text))["compound"])

    news_df = news_df.copy()
    news_df["sentiment"] = news_df["headline"].astype(str).map(score_fn)

    # Aggregate by day (mean compound)
    daily = news_df.groupby("date", as_index=False)["sentiment"].mean()
    daily["Date"] = pd.to_datetime(daily["date"])
    daily = daily.drop(columns=["date"]).set_index("Date").sort_index()
    return daily

# ===================== Technical Indicators =====================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds common technical indicators to DF (assumes columns: Open, High, Low, Close, Volume).
    """
    X = df.copy()
    # Returns
    X["ret1"] = X["Close"].pct_change()
    X["logret1"] = np.log1p(X["ret1"])
    # Moving averages
    X["sma5"] = X["Close"].rolling(5).mean()
    X["sma10"] = X["Close"].rolling(10).mean()
    X["ema12"] = X["Close"].ewm(span=12, adjust=False).mean()
    X["ema26"] = X["Close"].ewm(span=26, adjust=False).mean()
    # MACD & signal
    X["macd"] = X["ema12"] - X["ema26"]
    X["macd_signal"] = X["macd"].ewm(span=9, adjust=False).mean()
    # RSI
    X["rsi14"] = rsi(X["Close"], 14)
    # Volume transforms
    X["vol_chg"] = X["Volume"].pct_change() if "Volume" in X.columns else 0.0
    return X

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = (gain / (loss + 1e-8)).replace([np.inf, -np.inf], np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

# ===================== Datasets & Splits =====================
def build_dataset_with_sentiment(ticker: str, start: str, end=None,
                                 news_csv: str = NEWS_CSV,
                                 use_finbert: bool = False) -> pd.DataFrame:
    """
    Merge prices + technicals + sentiment into one daily dataframe.
    Creates binary target: next_close_up (1 if Close_{t+1} > Close_t else 0).
    """
    prices = fetch_prices(ticker=ticker, start=start, end=end)
    tech = add_indicators(prices)
    news = load_or_create_news_csv(news_csv)
    sent = compute_daily_sentiment(news, use_finbert)

    # Align by date (market days). Forward-fill sentiment to market days if missing.
    df = tech.copy()
    df = df.join(sent.reindex(df.index), how="left")
    df["sentiment"] = df["sentiment"].fillna(method="ffill").fillna(0.0)

    # Target: next-day movement
    df["close_next"] = df["Close"].shift(-1)
    df["target_up"] = (df["close_next"] > df["Close"]).astype(int)

    # Drop warmup NaNs from indicators and last row (no next close)
    df = df.dropna().iloc[:-1, :]
    return df

def chronological_split(df: pd.DataFrame,
                        test_size: float = TEST_SIZE,
                        val_size: float = VAL_SIZE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: train -> val -> test by ratios.
    """
    N = len(df)
    n_test = int(N * test_size)
    n_val = int(N * val_size)
    n_train = N - n_val - n_test
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train+n_val].copy()
    test = df.iloc[n_train+n_val:].copy()
    return train, val, test

# ===================== Modeling =====================
def build_feature_sets(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Returns (features_baseline, features_with_sentiment).
    Baseline excludes 'sentiment'; augmented includes it.
    """
    base_feats = [
        "Open","High","Low","Close","Volume",
        "ret1","logret1","sma5","sma10","ema12","ema26",
        "macd","macd_signal","rsi14","vol_chg"
    ]
    base_feats = [f for f in base_feats if f in df.columns]
    sent_feats = base_feats + (["sentiment"] if "sentiment" in df.columns else [])
    return base_feats, sent_feats

def scale_fit_transform(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, features: list):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train[features].values)
    Xv  = scaler.transform(val[features].values)
    Xte = scaler.transform(test[features].values)
    ytr = train["target_up"].values
    yv  = val["target_up"].values
    yte = test["target_up"].values
    return Xtr, Xv, Xte, ytr, yv, yte, scaler

def train_eval_config(Xtr, ytr, Xv, yv, Xte, yte, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a single config and evaluate on validation and test. Returns metrics dict.
    """
    if cfg["model"] == "logreg":
        clf = LogisticRegression(
            C=cfg.get("C", 1.0),
            penalty=cfg.get("penalty", "l2"),
            solver="liblinear",
            max_iter=1000,
            random_state=SEED
        )
    elif cfg["model"] == "rf":
        clf = RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", None),
            min_samples_leaf=cfg.get("min_samples_leaf", 1),
            n_jobs=-1,
            random_state=SEED
        )
    else:
        raise ValueError(f"Unknown model: {cfg['model']}")

    clf.fit(Xtr, ytr)

    def _metrics(X, y, split):
        proba = None
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[:, 1]
        pred = clf.predict(X)
        m = {
            f"{split}_acc": accuracy_score(y, pred),
            f"{split}_prec": precision_score(y, pred, zero_division=0),
            f"{split}_rec": recall_score(y, pred, zero_division=0),
            f"{split}_f1": f1_score(y, pred, zero_division=0),
        }
        if proba is not None and len(np.unique(y)) == 2:
            try:
                m[f"{split}_auc"] = roc_auc_score(y, proba)
            except Exception:
                m[f"{split}_auc"] = np.nan
        return m, pred, proba

    m_tr, _, _  = _metrics(Xtr, ytr, "train")
    m_v,  _, _  = _metrics(Xv,  yv,  "val")
    m_te, pte, pproba = _metrics(Xte, yte, "test")

    out = {"config": cfg, **m_tr, **m_v, **m_te, "pred_test": pte, "proba_test": pproba, "clf": clf}
    return out

def plot_confusion(y_true, y_pred, path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Down/Equal", "Up"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Down/Equal", "Up"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ===================== Runner =====================
def main():
    print("Building dataset with sentiment…")
    df = build_dataset_with_sentiment(TICKER, START, END, NEWS_CSV, USE_FINBERT)

    # Feature sets
    base_feats, sent_feats = build_feature_sets(df)

    # Chronological split
    train, val, test = chronological_split(df, TEST_SIZE, VAL_SIZE)

    # Baseline (no sentiment)
    Xtr_b, Xv_b, Xte_b, ytr, yv, yte, scaler_b = scale_fit_transform(train, val, test, base_feats)
    # With sentiment
    Xtr_s, Xv_s, Xte_s, _, _, _, scaler_s = scale_fit_transform(train, val, test, sent_feats)

    rows = []
    artifacts = {}

    # Search over configs for both feature sets
    for cfg in SEARCH_SPACE:
        # Baseline
        res_b = train_eval_config(Xtr_b, ytr, Xv_b, yv, Xte_b, yte, cfg)
        rec_b = {
            "feature_set": "baseline_no_sent",
            "model": cfg["model"],
            **{k: v for k, v in res_b.items() if isinstance(v, (int, float)) or k.endswith("_auc")}
        }
        rows.append(rec_b)
        key_b = f"baseline_{cfg['model']}"
        artifacts[key_b] = res_b

        # With sentiment
        res_s = train_eval_config(Xtr_s, ytr, Xv_s, yv, Xte_s, yte, cfg)
        rec_s = {
            "feature_set": "with_sentiment",
            "model": cfg["model"],
            **{k: v for k, v in res_s.items() if isinstance(v, (int, float)) or k.endswith("_auc")}
        }
        rows.append(rec_s)
        key_s = f"sent_{cfg['model']}"
        artifacts[key_s] = res_s

    # Results table
    res_df = pd.DataFrame(rows)
    res_cols = ["feature_set", "model",
                "val_acc", "val_f1", "val_prec", "val_rec", "val_auc",
                "test_acc", "test_f1", "test_prec", "test_rec", "test_auc",
                "train_acc", "train_f1"]
    
    for c in res_cols:
        if c not in res_df.columns:
            res_df[c] = np.nan

    res_df = res_df[["feature_set", "model",
                     "train_acc", "train_f1",
                     "val_acc", "val_f1", "val_prec", "val_rec", "val_auc",
                     "test_acc", "test_f1", "test_prec", "test_rec", "test_auc"]]

    # Choose best by validation F1
    best_idx = res_df["val_f1"].astype(float).idxmax()
    best_row = res_df.loc[best_idx]
    best_key = ("sent_" if best_row["feature_set"] == "with_sentiment" else "baseline_") + str(best_row["model"])
    best_art = artifacts[best_key]

    # Save results
    csv_path = os.path.join(RESULTS_DIR, "task7_results.csv")
    res_df.to_csv(csv_path, index=False)

    # Save confusion matrices
    if "baseline_logreg" in artifacts:
        plot_confusion(
            yte, artifacts["baseline_logreg"]["pred_test"],
            os.path.join(RESULTS_DIR, "task7_confusion_baseline_logreg.png"),
            "Baseline (No Sentiment) – Logistic Regression"
        )
    if "sent_logreg" in artifacts:
        plot_confusion(
            yte, artifacts["sent_logreg"]["pred_test"],
            os.path.join(RESULTS_DIR, "task7_confusion_sent_logreg.png"),
            "With Sentiment – Logistic Regression"
        )

    # Save predictions CSV for best model
    pred_df = test.copy()
    pred_df["pred_up"] = best_art["pred_test"]
    if best_art["proba_test"] is not None:
        pred_df["proba_up"] = best_art["proba_test"]
    out_pred_csv = os.path.join(RESULTS_DIR, "task7_best_predictions.csv")

    # Include sentiment column only if present (it is, but safe)
    cols = ["Close"]
    if "sentiment" in pred_df.columns:
        cols.append("sentiment")
    if "proba_up" in pred_df.columns:
        cols.append("proba_up")
    cols += ["pred_up", "target_up"]
    pred_df[cols].to_csv(out_pred_csv)

    # Console summary
    print("\nSaved results to:", csv_path)
    print("\nTop models (by val F1):")
    print(res_df.sort_values("val_f1", ascending=False).head(6))

    print("\nBest model summary:")
    print(best_row.to_string())

    # Classification report on test for best model
    print("\nTest Classification Report (Best):")
    print(classification_report(yte, best_art["pred_test"], digits=4))

    print("\nArtifacts saved in:", RESULTS_DIR)
    print("• task7_results.csv")
    if "baseline_logreg" in artifacts:
        print("• task7_confusion_baseline_logreg.png")
    if "sent_logreg" in artifacts:
        print("• task7_confusion_sent_logreg.png")
    print("• task7_best_predictions.csv")

# ===================== Main =====================
if __name__ == "__main__":
    main()
