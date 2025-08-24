# train.py
import os
import tensorflow as tf  

from stock_predictionJJ import load_data, create_model
from parameters import *

# ✅ Ensure directories exist
for folder in ["results", "logs", "data"]:
    os.makedirs(folder, exist_ok=True)

# ✅ Load dataset
print(f"Loading data for ticker: {ticker}")
data = load_data(
    ticker,
    n_steps=N_STEPS,
    scale=SCALE,
    shuffle=SHUFFLE,
    lookup_step=LOOKUP_STEP,
    split_by_date=SPLIT_BY_DATE,
    test_size=TEST_SIZE,
    feature_columns=FEATURE_COLUMNS
)

# ✅ Save raw dataset to CSV
ticker_data_filename = os.path.join("data", f"{ticker}.csv")
data["df"].to_csv(ticker_data_filename, index=False)
print(f"Data saved to {ticker_data_filename}")

# ✅ Build model
model = create_model(
    sequence_length=N_STEPS,
    n_features=len(FEATURE_COLUMNS),
    units=UNITS,
    cell=CELL,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    loss=tf.keras.losses.Huber(),   
    optimizer=OPTIMIZER,
    bidirectional=BIDIRECTIONAL
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

checkpointer = ModelCheckpoint(
    os.path.join("results", model_name + ".weights.h5"),
    save_weights_only=True,
    save_best_only=True,
    verbose=1
)
from tensorflow.keras.models import model_from_json

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

history = model.fit(
    data["X_train"], data["y_train"],
    validation_data=(data["X_test"], data["y_test"]),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpointer, tensorboard, early_stopping, reduce_lr],
    verbose=1
)
model_json = model.to_json()
with open(os.path.join("results", "model.json"), "w") as json_file:
    json_file.write(model_json)

print("Training complete ✅. Model saved in results/ folder.")
