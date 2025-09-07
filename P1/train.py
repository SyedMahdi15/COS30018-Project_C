import os
import shutil
import tensorflow as tf

from stock_predictionJJ import load_data, create_model
from parameters import * 

# ---------- Ensure directories ----------
for folder in ["results", "logs", "data", "csv-results"]:
    os.makedirs(folder, exist_ok=True)

ticker_local = "AAPL"

print(f"Loading data for ticker: {ticker_local}")
data = load_data(
    ticker=ticker_local,
    n_steps=N_STEPS,
    scale=SCALE,
    shuffle=SHUFFLE,
    lookup_step=LOOKUP_STEP,
    split_by_date=SPLIT_BY_DATE,
    test_size=TEST_SIZE,
    feature_columns=FEATURE_COLUMNS,

    # NEW: controlled date range, caching, and NaN policy
    start_date="2018-01-01",
    end_date="2025-08-01",
    cache_path=os.path.join("data", f"{ticker_local}_2018-01-01_2025-08-01.csv"),
    use_local=True,
    na_strategy="ffill_bfill"   # <- lowercase
)

# ---------- Save raw dataset snapshot ----------
ticker_data_filename = os.path.join("data", f"{ticker_local}.csv")
data["df"].to_csv(ticker_data_filename, index=True)
print(f"Data saved to {ticker_data_filename}")

# ---------- Build model ----------
model = create_model(
    sequence_length=N_STEPS,
    n_features=len(FEATURE_COLUMNS),
    units=UNITS,
    cell=CELL,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    loss=tf.keras.losses.Huber(),   # Keras 3 safe
    optimizer=OPTIMIZER,
    bidirectional=BIDIRECTIONAL
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

# IMPORTANT: model_name comes from parameters.py — keep it in sync with evaluate.py OR we’ll also save a “latest” copy
checkpointer = ModelCheckpoint(
    os.path.join("results", model_name + ".weights.h5"),
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
    verbose=1
)

tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)

# ---------- Train ----------
history = model.fit(
    data["X_train"], data["y_train"],
    validation_data=(data["X_test"], data["y_test"]),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpointer, tensorboard, early_stopping, reduce_lr],
    verbose=1
)

# ---------- Save model structure ----------
with open(os.path.join("results", "model.json"), "w") as json_file:
    json_file.write(model.to_json())

# ---------- Make evaluation robust: write pointers to latest weights ----------
best_weights = os.path.join("results", model_name + ".weights.h5")
if os.path.exists(best_weights):
    # Ensure the in-memory model also has the very best weights
    model.load_weights(best_weights)

    # 1) Save a copy with a stable filename for this ticker
    latest_weights = os.path.join("results", f"latest_{ticker_local}.weights.h5")
    shutil.copyfile(best_weights, latest_weights)

    # 2) Save the exact model_name used (so evaluate.py can read it)
    with open(os.path.join("results", "LATEST_RUN.txt"), "w") as f:
        f.write(model_name)

print("Training complete !. Model saved in results/ folder.")
