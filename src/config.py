"""
Configuration file for the AuroraGate Expense Categorization Challenge.

This module centralizes all paths, hyperparameters, and settings used
throughout the project to ensure consistency and ease of maintenance.
All paths are constructed using pathlib.Path for cross-platform compatibility.
"""

from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
import logging

# =============================================================================
# PROJECT PATHS
# =============================================================================
# BASE_DIR is the root of the project. Since this file is located in src/,
# we go one level up (parent.parent) to reach the project root.
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
PRETRAINED_DIR = BASE_DIR / "pretrained"

# =============================================================================
# GENERAL SETTINGS
# =============================================================================
RANDOM_STATE = 42
TARGET_COLUMN = "category"
TEST_SIZE = 0.2

TARGET_CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Transportation",
    "Entertainment",
    "Shopping",
    "Bills & Utilities",
    "Health & Wellness",
    "Travel",
    "Education",
    "Other",
]

# =============================================================================
# LOGGING SETTINGS
# =============================================================================
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================
N_SPLITS = 5
TIME_SERIES_SPLIT = TimeSeriesSplit(n_splits=N_SPLITS)

# =============================================================================
# TEXT PROCESSING SETTINGS
# =============================================================================
TFIDF_PARAMS = {
    "max_features": 50000,
    "ngram_range": (3, 5),  # Character n-grams (3 to 5 chars)
    "analyzer": "char_wb",
    "sublinear_tf": True,
}

FASTTEXT_PARAMS = {
    "vector_size": 100,
    "window": 5,
    "min_count": 2,
    "epochs": 10,
}

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.1,
    "depth": 6,
    "random_seed": RANDOM_STATE,
    "verbose": 100,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

# =============================================================================
# THRESHOLD TUNING SETTINGS
# =============================================================================
THRESHOLD_TUNING = {
    "default_threshold": 0.5,
    "min_threshold": 0.1,
    "max_threshold": 0.9,
    "n_trials": 50,  # For Optuna
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def setup_directories() -> None:
    """
    Create all required project directories if they do not already exist.

    This function ensures that the data, cache, models, submissions, experiments,
    and pretrained directories are available before any processing begins.
    Using parents=True allows nested directory creation, and exist_ok=True
    prevents errors if the directory already exists.
    """
    directories = [
        DATA_DIR,
        CACHE_DIR,
        MODELS_DIR,
        SUBMISSIONS_DIR,
        EXPERIMENTS_DIR,
        PRETRAINED_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MAIN (for testing purposes)
# =============================================================================
if __name__ == "__main__":
    # Setup directories and verify configuration
    setup_directories()

    print("=" * 60)
    print("AuroraGate Project Configuration")
    print("=" * 60)

    print("\n[Project Paths]")
    print(f"  BASE_DIR:          {BASE_DIR}")
    print(f"  DATA_DIR:          {DATA_DIR}")
    print(f"  CACHE_DIR:         {CACHE_DIR}")
    print(f"  MODELS_DIR:        {MODELS_DIR}")
    print(f"  SUBMISSIONS_DIR:   {SUBMISSIONS_DIR}")
    print(f"  EXPERIMENTS_DIR:   {EXPERIMENTS_DIR}")
    print(f"  PRETRAINED_DIR:    {PRETRAINED_DIR}")

    print("\n[General Settings]")
    print(f"  RANDOM_STATE:      {RANDOM_STATE}")
    print(f"  TARGET_COLUMN:     {TARGET_COLUMN}")
    print(f"  TEST_SIZE:         {TEST_SIZE}")
    print(f"  TARGET_CATEGORIES: {len(TARGET_CATEGORIES)} categories")

    print("\n[Validation Settings]")
    print(f"  N_SPLITS:          {N_SPLITS}")
    print(f"  TIME_SERIES_SPLIT: {TIME_SERIES_SPLIT}")

    print("\n[CatBoost Hyperparameters]")
    for key, value in CATBOOST_PARAMS.items():
        print(f"  {key}: {value}")

    print("\n[LightGBM Hyperparameters]")
    for key, value in LIGHTGBM_PARAMS.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("All directories have been created successfully.")
    print("=" * 60)