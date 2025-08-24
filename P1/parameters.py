import os
import time
from tensorflow.keras.layers import LSTM

# ==============================
# Data parameters
# ==============================

# Window size (sequence length)
N_STEPS = 50
# Lookup step, 1 means next day
LOOKUP_STEP = 15

# Whether to scale feature columns & output price
SCALE = True
scale_str = f"sc-{int(SCALE)}"

# Whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"

# Whether to split the training/testing set by date
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"

# Test ratio size, 0.2 = 20%
TEST_SIZE = 0.2

# Features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

# Current date
date_now = time.strftime("%Y-%m-%d")

# ==============================
# Model parameters
# ==============================

N_LAYERS = 2           # Number of LSTM layers
CELL = LSTM            # LSTM cell type
UNITS = 256            # Neurons per layer
DROPOUT = 0.4          # Dropout rate
BIDIRECTIONAL = False  # Use bidirectional LSTM?

# ==============================
# Training parameters
# ==============================

LOSS = "huber_loss"    # Loss function
OPTIMIZER = "adam"     # Optimizer
BATCH_SIZE = 64        # Batch size
EPOCHS = 500           # Number of epochs

# ==============================
# Ticker & file paths
# ==============================

# Stock ticker symbol
ticker = "AAPL"  # Change to any valid ticker (e.g., "MSFT", "TSLA")

# Data folder
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Dataset file path
ticker_data_filename = os.path.join(DATA_DIR, f"{ticker}_{date_now}.csv")

# Model name (unique based on parameters)
model_name = (
    f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-"
    f"{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-"
    f"layers-{N_LAYERS}-units-{UNITS}"
)
if BIDIRECTIONAL:
    model_name += "-b"
