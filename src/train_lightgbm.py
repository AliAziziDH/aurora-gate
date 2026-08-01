"""Train a LightGBM model and evaluate a CatBoost-LightGBM ensemble."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import EXPERIMENTS_DIR, MODELS_DIR, RANDOM_STATE, TARGET_COLUMN
from src.data_loader import DataLoader
from src.feature_engineering import engineer_features
from src.train_model import (
    MODEL_PATH as CATBOOST_MODEL_PATH,
    VECTORIZER_PATH,
    _fit_text_features,
    _build_model,
    _class_weights,
    _model_frame,
    _prepare_text,
    _transform_text_features,
)
from src.logger import get_logger
from src.training_utils import (
    run_cv_training,
    save_model_artifacts,
    save_training_summary,
)


logger = get_logger(__name__)

LIGHTGBM_MODEL_PATH = Path(MODELS_DIR) / "lightgbm_model.pkl"
THRESHOLDS_PATH = Path(MODELS_DIR) / "thresholds.json"
ENSEMBLE_WEIGHTS_PATH = Path(MODELS_DIR) / "ensemble_weights.json"
ENSEMBLE_SUMMARY_PATH = Path(EXPERIMENTS_DIR) / "ensemble_summary.json"


def _load_text_artifacts(text: pd.Series) -> Dict[str, Any]:
    """Load the fitted text bundle or fit and persist it when absent."""
    if VECTORIZER_PATH.is_file():
        bundle = joblib.load(VECTORIZER_PATH)
        required = {"char_vectorizer", "word_vectorizer", "svd"}
        if required.issubset(bundle):
            logger.info("Loaded TF-IDF artifacts from %s", VECTORIZER_PATH)
            return bundle

    logger.info("TF-IDF artifacts not found; fitting them from training data")
    char_vectorizer, word_vectorizer, svd, _ = _fit_text_features(text)
    bundle = {
        "char_vectorizer": char_vectorizer,
        "word_vectorizer": word_vectorizer,
        "svd": svd,
    }
    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, VECTORIZER_PATH)
    return bundle


def _load_thresholds() -> Dict[str, float]:
    """Load class thresholds required by the ensemble post-processing."""
    if not THRESHOLDS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {THRESHOLDS_PATH}; run `python -m src.train_model` first."
        )
    with THRESHOLDS_PATH.open("r", encoding="utf-8") as threshold_file:
        return {str(key): float(value) for key, value in json.load(threshold_file).items()}


def _numeric_frame(frame: pd.DataFrame, categorical: List[str]) -> pd.DataFrame:
    """Encode categorical columns numerically for LightGBM."""
    result = frame.copy()
    for column in categorical:
        result[column] = (
        result[column]
        .fillna("unknown")
        .astype("category")
        .cat.codes
        .astype("int32")
    )
    return result.astype(float)


def _build_lightgbm() -> lgb.LGBMClassifier:
    """Build the requested balanced LightGBM classifier."""
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.1,
        num_leaves=31,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        verbose=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass",
        n_jobs=-1,
    )


def _ordered_probabilities(
    probabilities: np.ndarray, model_classes: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """Reorder model probabilities into the shared sorted class order."""
    ordered = np.zeros((len(probabilities), len(classes)), dtype=float)
    for model_index, label in enumerate(model_classes):
        target_index = np.where(classes == str(label))[0]
        if len(target_index):
            ordered[:, target_index[0]] = probabilities[:, model_index]
    return ordered


def _apply_thresholds(
    probabilities: np.ndarray, classes: np.ndarray, thresholds: Dict[str, float]
) -> np.ndarray:
    """Apply the saved one-vs-rest thresholds and return class labels."""
    adjusted = probabilities.copy()
    for index, label in enumerate(classes):
        adjusted[:, index] /= thresholds.get(str(label), 1.0)
    return classes[np.argmax(adjusted, axis=1)]


def _build_lightgbm_builder():
    """Create a model builder function for LightGBM."""
    def model_builder():
        """Build and return a LightGBM classifier."""
        return _build_lightgbm()
    return model_builder


def train_lightgbm() -> Dict[str, Any]:
    """Train LightGBM, evaluate folds, save artifacts, and report ensemble gains."""
    # Load data and prepare features
    if not CATBOOST_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {CATBOOST_MODEL_PATH}; run `python -m src.train_model` first."
        )
    catboost_model = joblib.load(CATBOOST_MODEL_PATH)
    thresholds = _load_thresholds()

    loader = DataLoader(use_cache=False)
    raw_train = loader.load_train_data(force_reload=True)
    raw_train = raw_train.sort_values("transaction_id").reset_index(drop=True)
    engineered = engineer_features(raw_train, is_train=True)
    text_bundle = _load_text_artifacts(_prepare_text(engineered))
    text_features = _transform_text_features(
        _prepare_text(engineered),
        text_bundle["char_vectorizer"],
        text_bundle["word_vectorizer"],
        text_bundle["svd"],
    )
    cat_frame, categorical = _model_frame(engineered, text_features)
    lgb_frame = _numeric_frame(cat_frame, categorical)
    labels = engineered[TARGET_COLUMN].astype(str).to_numpy()
    classes = np.array(sorted(np.unique(labels)))
    encoded_labels = np.array([np.where(classes == label)[0][0] for label in labels])
    catboost_weights = _class_weights(pd.Series(labels))
    
    # Train LightGBM using CV utilities
    lgb_builder = _build_lightgbm_builder()
    lgb_fit_params = {
        "eval_set": None,  # Will be set by run_cv_training
        "callbacks": [lgb.early_stopping(50, verbose=False)],
    }
    
    lgb_fold_scores, lgb_oof_preds, lgb_oof_probas = run_cv_training(
        model_builder=lgb_builder,
        train_data=lgb_frame,
        target=pd.Series(encoded_labels),
        fit_params=lgb_fit_params,
        eval_metric="macro_f1",
    )
    
    # Reorder LightGBM probabilities to match classes
    lgb_oof_probas = _ordered_probabilities(lgb_oof_probas, classes, classes)
    
    # Train CatBoost models and evaluate ensemble
    splitter = TimeSeriesSplit(n_splits=5)
    fold_results = []
    all_cat_probabilities = []
    
    for fold, (train_indices, valid_indices) in enumerate(splitter.split(lgb_frame), start=1):
        # Get LightGBM probabilities for this fold
        lgb_probabilities = lgb_oof_probas[valid_indices]
        
        # Train and evaluate CatBoost for this fold
        fold_catboost = _build_model(catboost_weights)
        categorical_indices = [cat_frame.columns.get_loc(column) for column in categorical]
        fold_catboost.fit(
            cat_frame.iloc[train_indices],
            labels[train_indices],
            cat_features=categorical_indices,
            eval_set=(cat_frame.iloc[valid_indices], labels[valid_indices]),
            use_best_model=False,
        )
        cat_probabilities = _ordered_probabilities(
            fold_catboost.predict_proba(cat_frame.iloc[valid_indices]),
            np.asarray(fold_catboost.classes_),
            classes,
        )
        
        # Evaluate all approaches
        fold_labels = labels[valid_indices]
        lgb_predictions = _apply_thresholds(lgb_probabilities, classes, thresholds)
        cat_predictions = _apply_thresholds(cat_probabilities, classes, thresholds)
        ensemble_probabilities = (lgb_probabilities + cat_probabilities) / 2.0
        ensemble_predictions = _apply_thresholds(ensemble_probabilities, classes, thresholds)
        
        lgb_score = f1_score(fold_labels, lgb_predictions, average="macro", zero_division=0)
        cat_score = f1_score(fold_labels, cat_predictions, average="macro", zero_division=0)
        ensemble_score = f1_score(
            fold_labels, ensemble_predictions, 
            average="macro", zero_division=0
        )
        
        fold_results.append({
            "fold": fold,
            "lightgbm_macro_f1": float(lgb_score),
            "catboost_macro_f1": float(cat_score),
            "ensemble_macro_f1": float(ensemble_score),
            "ensemble_improvement_over_catboost": float(ensemble_score - cat_score),
        })
        
        logger.info(
            "Fold %d: LightGBM F1=%.4f, CatBoost F1=%.4f, Ensemble F1=%.4f",
            fold, lgb_score, cat_score, ensemble_score
        )
        all_cat_probabilities.append(cat_probabilities)

    # Train final LightGBM model
    final_lightgbm = _build_lightgbm()
    final_lightgbm.fit(lgb_frame, encoded_labels)
    
    # Save LightGBM artifacts
    save_model_artifacts(
        model=final_lightgbm,
        artifacts={
            "feature_columns": lgb_frame.columns.tolist(),
            "categorical_columns": categorical,
            "classes": classes.tolist(),
        },
        model_name="lightgbm",
    )
    
    # Save ensemble weights
    ensemble_weights = {"catboost": 0.5, "lightgbm": 0.5}
    with ENSEMBLE_WEIGHTS_PATH.open("w", encoding="utf-8") as weights_file:
        json.dump(ensemble_weights, weights_file, indent=2)

    # Calculate overall metrics
    cat_validation = np.vstack(all_cat_probabilities)
    validation_labels = labels  # Using all labels since we're using OOF predictions
    
    lgb_overall = f1_score(
        validation_labels,
        _apply_thresholds(lgb_oof_probas, classes, thresholds),
        average="macro",
        zero_division=0,
    )
    cat_overall = f1_score(
        validation_labels,
        _apply_thresholds(cat_validation, classes, thresholds),
        average="macro",
        zero_division=0,
    )
    ensemble_overall = f1_score(
        validation_labels,
        _apply_thresholds((lgb_oof_probas + cat_validation) / 2.0, classes, thresholds),
        average="macro",
        zero_division=0,
    )
    
    # Prepare summary
    summary = {
        "fold_results": fold_results,
        "overall_lightgbm_macro_f1": float(lgb_overall),
        "overall_catboost_macro_f1": float(cat_overall),
        "overall_ensemble_macro_f1": float(ensemble_overall),
        "overall_improvement_over_catboost": float(ensemble_overall - cat_overall),
        "ensemble_weights": ensemble_weights,
        "classes": classes.tolist(),
        "model_path": str(LIGHTGBM_MODEL_PATH),
        "ensemble_weights_path": str(ENSEMBLE_WEIGHTS_PATH),
    }
    
    # Save summary
    save_training_summary(summary, "ensemble_summary")

    print("=" * 70)
    print("AuroraGate LightGBM and Ensemble Training")
    print("=" * 70)
    for result in fold_results:
        print(
            f"Fold {result['fold']}: LightGBM={result['lightgbm_macro_f1']:.4f}, "
            f"Ensemble={result['ensemble_macro_f1']:.4f}"
        )
    print(f"Overall LightGBM macro F1: {lgb_overall:.4f}")
    print(f"Overall CatBoost macro F1: {cat_overall:.4f}")
    print(f"Overall Ensemble macro F1: {ensemble_overall:.4f}")
    print(f"Improvement over CatBoost: {ensemble_overall - cat_overall:+.4f}")
    print(f"Saved model: {LIGHTGBM_MODEL_PATH}")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    train_lightgbm()