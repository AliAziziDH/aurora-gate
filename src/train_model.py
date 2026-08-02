"""Train and evaluate an AuroraGate CatBoost transaction classifier using native text_features."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import (
    CATBOOST_PARAMS,
    EXPERIMENTS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.data_loader import DataLoader
from src.feature_engineering import categorical_feature_names, engineer_features
from src.logger import get_logger
from src.training_utils import (
    compute_class_weights,
    save_model_artifacts,
    save_training_summary,
)

logger = get_logger(__name__)

MODEL_NAME = "catboost"
MODEL_PATH = Path(MODELS_DIR) / f"{MODEL_NAME}_model.pkl"
THRESHOLDS_PATH = Path(MODELS_DIR) / "thresholds.json"


def _prepare_model_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Prepare feature DataFrame for CatBoost using native text_features.
    
    Returns:
        Tuple of (frame, categorical_columns, text_columns)
    """
    categorical = [col for col in categorical_feature_names() if col in df.columns]
    text_columns = ["description"]
    
    excluded = {"date", "transaction_id", TARGET_COLUMN}
    feature_columns = [
        col for col in df.columns
        if col not in excluded
    ]
    
    frame = df[feature_columns].copy()
    
    # Ensure text column is clean string
    for col in text_columns:
        frame[col] = frame[col].fillna("").astype(str)
        
    # Ensure categorical columns are string
    for col in categorical:
        frame[col] = frame[col].fillna("unknown").astype(str)
        
    return frame, categorical, text_columns


def _build_catboost_model(
    class_weights: Dict[str, float],
    text_features: List[str],
    cat_features: List[str]
) -> CatBoostClassifier:
    """Build CatBoost classifier configured for native text features."""
    params = dict(CATBOOST_PARAMS)
    params.update(
        {
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1:average=Macro",
            "class_weights": class_weights,
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "allow_writing_files": False,
            "text_features": text_features,
            "cat_features": cat_features,
            "od_type": "Iter",
            "od_wait": 50,
        }
    )
    return CatBoostClassifier(**params)


def optimize_class_thresholds(
    y_true: np.ndarray,
    oof_probabilities: np.ndarray,
    classes: np.ndarray
) -> Dict[str, float]:
    """
    Unified probability threshold optimization directly maximizing Macro F1 on OOF.
    
    Returns a dictionary mapping class names to probability multiplier weights.
    """
    num_classes = len(classes)
    
    def objective(multipliers: np.ndarray) -> float:
        # Scale probabilities by multipliers and take argmax
        scaled_probas = oof_probabilities * multipliers
        preds = classes[np.argmax(scaled_probas, axis=1)]
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        return -score  # Minimize negative F1
        
    initial_multipliers = np.ones(num_classes)
    bounds = [(0.1, 10.0)] * num_classes
    
    res = minimize(
        objective,
        x0=initial_multipliers,
        bounds=bounds,
        method="Powell",
        options={"maxiter": 200}
    )
    
    best_multipliers = res.x
    best_thresholds = {str(cls): float(best_multipliers[i]) for i, cls in enumerate(classes)}
    return best_thresholds


def apply_class_thresholds(
    probabilities: np.ndarray,
    classes: np.ndarray,
    thresholds: Dict[str, float]
) -> np.ndarray:
    """Apply probability multipliers and return class predictions."""
    multipliers = np.array([thresholds.get(str(cls), 1.0) for cls in classes])
    scaled_probas = probabilities * multipliers
    return classes[np.argmax(scaled_probas, axis=1)]


def train_model() -> Dict[str, object]:
    """Train CatBoost using native text_features and OOF threshold optimization."""
    logger.info("Starting CatBoost training pipeline with native text_features...")
    
    # 1. Load Data
    loader = DataLoader(use_cache=False)
    raw_train = loader.load_train_data(force_reload=True)
    raw_train = raw_train.sort_values("transaction_id").reset_index(drop=True)
    
    # 2. Engineer Features (with smooth K-Fold target encoding enabled)
    engineered = engineer_features(raw_train, is_train=True)
    model_frame, categorical_cols, text_cols = _prepare_model_frame(engineered)
    
    target = engineered[TARGET_COLUMN].astype(str)
    classes = np.array(sorted(target.unique()))
    class_weights = compute_class_weights(target)
    
    # 3. Cross-Validation
    splitter = TimeSeriesSplit(n_splits=5)
    
    oof_indices = []
    oof_predictions_list = []
    oof_probabilities_list = []
    fold_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(model_frame), start=1):
        X_tr, X_val = model_frame.iloc[train_idx], model_frame.iloc[valid_idx]
        y_tr, y_val = target.iloc[train_idx], target.iloc[valid_idx]
        
        model = _build_catboost_model(class_weights, text_cols, categorical_cols)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=50,
            verbose=False
        )
        
        probas = model.predict_proba(X_val)
        model_classes = np.array(model.classes_)
        
        # Align probabilities to sorted global classes
        aligned_probas = np.zeros((len(valid_idx), len(classes)))
        for i, cls in enumerate(model_classes):
            idx_in_global = np.where(classes == cls)[0][0]
            aligned_probas[:, idx_in_global] = probas[:, i]
            
        preds = classes[np.argmax(aligned_probas, axis=1)]
        fold_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        fold_scores.append(float(fold_f1))
        
        logger.info("Fold %d raw CatBoost Macro F1: %.4f", fold, fold_f1)
        
        oof_indices.extend(valid_idx)
        oof_predictions_list.append(preds)
        oof_probabilities_list.append(aligned_probas)
        
    oof_y_true = target.iloc[oof_indices].to_numpy()
    oof_probabilities = np.vstack(oof_probabilities_list)
    raw_oof_preds = np.concatenate(oof_predictions_list)
    
    raw_oof_macro_f1 = f1_score(oof_y_true, raw_oof_preds, average="macro", zero_division=0)
    logger.info("Overall Raw OOF Macro F1: %.4f", raw_oof_macro_f1)
    
    # 4. Threshold Optimization on OOF
    logger.info("Optimizing per-class probability thresholds on OOF...")
    threshold_multipliers = optimize_class_thresholds(oof_y_true, oof_probabilities, classes)
    
    tuned_oof_preds = apply_class_thresholds(oof_probabilities, classes, threshold_multipliers)
    tuned_oof_macro_f1 = f1_score(oof_y_true, tuned_oof_preds, average="macro", zero_division=0)
    logger.info("Overall Tuned OOF Macro F1: %.4f (Gain: +%.4f)", tuned_oof_macro_f1, tuned_oof_macro_f1 - raw_oof_macro_f1)
    
    # 5. Train Final Model on Full Training Dataset
    logger.info("Training final CatBoost model on complete dataset...")
    final_model = _build_catboost_model(class_weights, text_cols, categorical_cols)
    final_model.fit(model_frame, target, verbose=False)
    
    # Save artifacts
    saved_paths = save_model_artifacts(
        model=final_model,
        artifacts={
            "categorical_columns": categorical_cols,
            "text_columns": text_cols,
            "feature_columns": model_frame.columns.tolist(),
        },
        model_name=MODEL_NAME,
    )
    
    with THRESHOLDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(threshold_multipliers, f, indent=2)
        
    result = {
        "fold_scores": [{"fold": i + 1, "macro_f1": score} for i, score in enumerate(fold_scores)],
        "raw_oof_macro_f1": float(raw_oof_macro_f1),
        "tuned_oof_macro_f1": float(tuned_oof_macro_f1),
        "f1_improvement": float(tuned_oof_macro_f1 - raw_oof_macro_f1),
        "classes": classes.tolist(),
        "threshold_multipliers": threshold_multipliers,
        "model_path": str(saved_paths["model"]),
    }
    
    save_training_summary(result, f"{MODEL_NAME}_training_summary")
    
    print("\n" + "=" * 70)
    print("AURORAGATE CATBOOST CV SUMMARY (Native text_features)")
    print("=" * 70)
    for f in result["fold_scores"]:
        print(f"Fold {f['fold']} Macro F1: {f['macro_f1']:.4f}")
    print(f"\nRaw OOF Macro F1:   {raw_oof_macro_f1:.4f}")
    print(f"Tuned OOF Macro F1: {tuned_oof_macro_f1:.4f}")
    print(f"CV Macro F1 Gain:   +{tuned_oof_macro_f1 - raw_oof_macro_f1:.4f}")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    train_model()