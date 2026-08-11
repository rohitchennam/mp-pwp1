import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================
# Dataset Paths
# ==========================

RAW_DATA = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "CreditCarddataset_Final_370100.csv"
)

PROCESSED_DATA = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models"
)

LOG_DIR = os.path.join(
    BASE_DIR,
    "logs"
)

# ==========================
# Dataset Split
# ==========================

RANDOM_STATE = 42

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.10