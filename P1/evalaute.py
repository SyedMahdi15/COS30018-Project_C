import os
import glob
import numpy as np
import tensorflow as tf

from stock_predictionJJ import (
    TICKER, START_DATE, END_DATE, PERIOD, NA_STRATEGY, INTERVAL,
    N_STEPS, LOOKUP_STEP, SCALE, SPLIT_BY_DATE, SHUFFLE, TEST_SIZE,
    FEATURE_COLUMNS, N_LAYERS, CELL, UNITS, DROPOUT, BIDIRECTIONAL,
    LOSS, LOSS_NAME, OPTIMIZER,
    RESULTS_DIR, LOGS_DIR, DATA_DIR, CSV_RESULTS_DIR, DATE_NOW,
    MODEL_NAME, load_data, create_model, get_final_df, predict_future, plot_graph
)

def _load_best_weights(model):
    exact = os.path.join(RESULTS_DIR, MODEL_NAME + ".weights.h5")
    if os.path.exists(exact):
        model.load_weights(exact)
        print(f"[evaluate] Loaded weights: {exact}")
        return

    # 2) latest_<TICKER>.weights.h5 (if training saved a copy)
    latest_by_ticker = os.path.join(RESULTS_DIR, f"latest_{TICKER}.weights.h5")
    if os.path.exists(latest_by_ticker):
        model.load_weights(latest_by_ticker)
        print(f"[evaluate] Loaded weights: {latest_by_ticker}")
        return

    # 3) LATEST_RUN.txt → contains last MODEL_NAME used
    latest_run_txt = os.path.join(RESULTS_DIR, "LATEST_RUN.txt")
    if os.path.exists(latest_run_txt):
        with open(latest_run_txt, "r") as f:
            last_model_name = f.read().strip()
        candidate = os.path.join(RESULTS_DIR, last_model_name + ".weights.h5")
        if os.path.exists(candidate):
            model.load_weights(candidate)
            print(f"[evaluate] Loaded weights: {candidate}")
            return

    # 4) Fallback: pick most recent weights for this ticker
    pattern = os.path.join(RESULTS_DIR, f"*_{TICKER}-*.weights.h5")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if matches:
        model.load_weights(matches[0])
        print(f"[evaluate] Loaded weights: {matches[0]}")
        return

    raise FileNotFoundError(
        f"Could not find any weights for ticker {TICKER}. "
        f"Tried: {exact}, {latest_by_ticker}, LATEST_RUN.txt, and pattern {pattern}."
    )

def main():
    print(f"[evaluate] Loading data for ticker: {TICKER}")

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
        cache_path=None,         # or point to a specific cached CSV if you prefer
        use_local=True,
        na_strategy=NA_STRATEGY
    )

    model = create_model(
        N_STEPS,
        len(FEATURE_COLUMNS),
        loss=LOSS,               # same loss object as training
        units=UNITS,
        cell=CELL,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        optimizer=OPTIMIZER,
        bidirectional=BIDIRECTIONAL,
    )

    _load_best_weights(model)

    # Evaluate
    loss_val, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    if SCALE:
        mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae

    # Final df + metrics
    final_df = get_final_df(model, data, scale=SCALE, lookup_step=LOOKUP_STEP)
    future_price = predict_future(model, data, N_STEPS, scale=SCALE)

    accuracy_score = (
        (final_df["sell_profit"] > 0).sum() + (final_df["buy_profit"] > 0).sum()
    ) / len(final_df)
    total_buy_profit = final_df["buy_profit"].sum()
    total_sell_profit = final_df["sell_profit"].sum()
    total_profit = total_buy_profit + total_sell_profit
    profit_per_trade = total_profit / len(final_df)

    print("\n=== Evaluation Summary ===")
    print(f"Ticker: {TICKER}")
    print(f"Predicted price after {LOOKUP_STEP} days: {future_price:.2f}")
    print(f"{LOSS_NAME} loss (val): {loss_val:.6f}")
    print(f"Mean Absolute Error (inverse scaled): {mean_absolute_error:.4f}")
    print(f"Accuracy score: {accuracy_score:.4f}")
    print(f"Total buy profit: {total_buy_profit:.2f}")
    print(f"Total sell profit: {total_sell_profit:.2f}")
    print(f"Total profit: {total_profit:.2f}")
    print(f"Profit per trade: {profit_per_trade:.4f}")

    # Plot Actual vs Predicted
    plot_graph(final_df, LOOKUP_STEP)

    csv_out = os.path.join(CSV_RESULTS_DIR, MODEL_NAME + "_eval.csv")
    final_df.to_csv(csv_out, index=True)
    print(f"[evaluate] Saved evaluation CSV to {csv_out}")

if __name__ == "__main__":
    main()
