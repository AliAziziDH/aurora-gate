"""Final submission pipeline for AuroraGate using clean CatBoost native text_features and tuned thresholds."""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.config import MODELS_DIR, SUBMISSIONS_DIR, TARGET_COLUMN
from src.data_loader import DataLoader
from src.feature_engineering import engineer_features
from src.logger import get_logger
from src.train_model import apply_class_thresholds, _prepare_model_frame

logger = get_logger(__name__)

MODEL_PATH = Path(MODELS_DIR) / "catboost_model.pkl"
THRESHOLDS_PATH = Path(MODELS_DIR) / "thresholds.json"
SUBMISSION_PATH = Path(SUBMISSIONS_DIR) / "submission_final.csv"


def run_final_pipeline() -> Dict[str, Any]:
    """Generate final submission using trained CatBoost model and tuned class thresholds."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file missing: {MODEL_PATH}. Run `python -m src.train_model` first.")
    if not THRESHOLDS_PATH.is_file():
        raise FileNotFoundError(f"Thresholds file missing: {THRESHOLDS_PATH}. Run `python -m src.train_model` first.")
        
    logger.info("Loading CatBoost model and tuned class thresholds...")
    model: CatBoostClassifier = joblib.load(MODEL_PATH)
    
    with THRESHOLDS_PATH.open("r", encoding="utf-8") as f:
        thresholds: Dict[str, float] = json.load(f)
        
    classes = np.array(model.classes_)
    
    # Load raw test data
    loader = DataLoader(use_cache=False)
    raw_test = loader.load_test_data(force_reload=True)
    raw_test = raw_test.sort_values("transaction_id").reset_index(drop=True)
    
    # Feature Engineering (with store_name_target_enc using train stats)
    engineered_test = engineer_features(raw_test, is_train=False)
    test_frame, _, _ = _prepare_model_frame(engineered_test)
    
    # Predict probabilities
    logger.info("Predicting test probabilities...")
    test_probas = model.predict_proba(test_frame)
    
    # Apply tuned thresholds
    test_predictions = apply_class_thresholds(test_probas, classes, thresholds)
    
    submission = pd.DataFrame({
        "transaction_id": raw_test["transaction_id"],
        TARGET_COLUMN: test_predictions
    })
    
    submission.to_csv(SUBMISSION_PATH, index=False)
    logger.info("Saved final submission to %s", SUBMISSION_PATH)
    
    result = {
        "submission_path": str(SUBMISSION_PATH),
        "shape": list(submission.shape),
        "distribution": submission[TARGET_COLUMN].value_counts().to_dict(),
    }
    
    print("\n" + "=" * 70)
    print("AURORAGATE FINAL SUBMISSION GENERATED")
    print("=" * 70)
    print(f"File Path: {SUBMISSION_PATH}")
    print(f"Shape: {submission.shape}")
    print("\nPrediction Distribution:")
    print(submission[TARGET_COLUMN].value_counts().to_string())
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    run_final_pipeline()