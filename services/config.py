from pathlib import Path


# =========================================================
# PROJECT DIRECTORIES
# =========================================================

# Root folder: sistem-backend-ai-lightgbm-mlp/
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directory
DATA_DIR = BASE_DIR / "data"

# Trained models directory
MODEL_DIR = BASE_DIR / "model"

# Evaluation results directory
RESULTS_DIR = BASE_DIR / "results"


# =========================================================
# DATASET FILES
# =========================================================

TRAIN_TRANSACTION_PATH = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_PATH = DATA_DIR / "train_identity.csv"

TEST_TRANSACTION_PATH = DATA_DIR / "test_transaction.csv"
TEST_IDENTITY_PATH = DATA_DIR / "test_identity.csv"


# =========================================================
# MODEL OUTPUT FILES
# =========================================================

LIGHTGBM_MODEL_PATH = MODEL_DIR / "lightgbm_model.pkl"
MLP_MODEL_PATH = MODEL_DIR / "mlp_model.keras"

SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
LABEL_ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"


# =========================================================
# EVALUATION OUTPUT FILES
# =========================================================

CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = RESULTS_DIR / "roc_curve.png"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.png"
METRICS_PATH = RESULTS_DIR / "metrics.txt"


# =========================================================
# DATA CONFIGURATION
# =========================================================

TARGET_COLUMN = "isFraud"
ID_COLUMN = "TransactionID"
TIME_COLUMN = "TransactionDT"

RANDOM_STATE = 42

# IEEE-CIS contains highly imbalanced fraud labels.
# Class balancing will be handled in the training pipeline.
USE_CLASS_WEIGHT = True


# =========================================================
# LIGHTGBM CONFIGURATION
# =========================================================

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 1200,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


# =========================================================
# MLP CONFIGURATION
# =========================================================

MLP_HIDDEN_UNITS = (256, 128, 64, 32)

MLP_DROPOUT_RATE = 0.25
MLP_LEARNING_RATE = 0.0005
MLP_BATCH_SIZE = 512
MLP_EPOCHS = 60
MLP_VALIDATION_SPLIT = 0.15

EARLY_STOPPING_PATIENCE = 8


# =========================================================
# HYBRID MODEL CONFIGURATION
# =========================================================

# Initial weights for probability averaging.
# These are starting values, not assumed optimal weights.
LIGHTGBM_WEIGHT = 0.5
MLP_WEIGHT = 0.5

# The final classification threshold will be selected
# using validation data during the training/evaluation workflow.
DEFAULT_THRESHOLD = 0.5


# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================

for directory in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
