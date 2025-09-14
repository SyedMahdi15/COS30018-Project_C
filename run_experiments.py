# import os
# import pandas as pd
# from itertools import product

# from src.data import fetch_prices, minmax_scale, make_sequences
# from src.model_factory import build_sequence_model
# from src.train_eval import split_time_series, train_one, evaluate
# from src.plots import plot_history, plot_pred_vs_actualconda 

# # ---------- Config ----------
# TICKER   = "AAPL"
# START    = "2018-01-01"
# N_STEPS  = 60
# HORIZON  = 1
# EPOCHS   = 40
# BATCH    = 32

# SEARCH_SPACE = {
#     "layer_type":  ["LSTM", "GRU", "RNN"],
#     "num_layers":  [1, 2],
#     "hidden_size": [64, 128],
#     "dropout":     [0.2],
#     "lr":          [1e-3],
#     "batch_size":  [BATCH],
#     "epochs":      [EPOCHS],
# }
# # ----------------------------

# def main():
#     os.makedirs("results", exist_ok=True)
#     os.makedirs("models", exist_ok=True)

#     # 1) Data
#     df = fetch_prices(TICKER, start=START)
#     scaled_close, smin, smax = minmax_scale(df["Close"])
#     X, y = make_sequences(scaled_close, n_steps=N_STEPS, horizon=HORIZON)
#     (Xtr, ytr), (Xv, yv), (Xte, yte) = split_time_series(X, y)

#     rows = []
#     for layer_type, num_layers, hidden_size, dropout, lr, batch_size, epochs in product(
#         SEARCH_SPACE["layer_type"],
#         SEARCH_SPACE["num_layers"],
#         SEARCH_SPACE["hidden_size"],
#         SEARCH_SPACE["dropout"],
#         SEARCH_SPACE["lr"],
#         SEARCH_SPACE["batch_size"],
#         SEARCH_SPACE["epochs"],
#     ):
#         run_name = f"{layer_type}_L{num_layers}_H{hidden_size}_D{dropout}_B{batch_size}"
#         outdir = os.path.join("models", run_name)

#         # 2) Model
#         model = build_sequence_model(
#             input_shape=(N_STEPS, 1),
#             layer_type=layer_type,
#             num_layers=num_layers,
#             hidden_size=hidden_size,
#             dropout=dropout,
#             lr=lr
#         )

#         # 3) Train
#         hist = train_one(
#             model, Xtr, ytr, Xv, yv,
#             outdir=outdir, epochs=epochs, batch_size=batch_size, verbose=1
#         )

#         # 4) Plots
#         plot_history(hist, os.path.join("results", f"{run_name}_history.png"))
#         plot_pred_vs_actual(model, Xte, yte, os.path.join("results", f"{run_name}_pred.png"))

#         # 5) Evaluate
#         metrics = evaluate(model, Xte, yte)
#         rows.append({
#             "run": run_name,
#             "layer_type": layer_type,
#             "num_layers": num_layers,
#             "hidden_size": hidden_size,
#             "dropout": dropout,
#             "batch_size": batch_size,
#             "lr": lr,
#             "epochs": epochs,
#             **metrics
#         })

#     # 6) Save summary
#     df_res = pd.DataFrame(rows).sort_values(["mae", "loss"])
#     df_res.to_csv("results/task4_results.csv", index=False)
#     print("\nTop configs by MAE:")
#     print(df_res.head(5))

# if __name__ == "__main__":
#     main()
