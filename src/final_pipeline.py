"""Select the best AuroraGate pipeline, apply rules, and create a submission."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import EXPERIMENTS_DIR, MODELS_DIR, SUBMISSIONS_DIR, TARGET_COLUMN
from src.data_loader import DataLoader, logger as data_loader_logger
from src.feature_engineering import engineer_features
from src.train_lightgbm import (
    LIGHTGBM_MODEL_PATH,
    _apply_thresholds,
    _load_text_artifacts,
    _numeric_frame,
    _ordered_probabilities,
    _build_lightgbm,
)
from src.train_model import (
    MODEL_PATH as CATBOOST_MODEL_PATH,
    VECTORIZER_PATH,
    _build_model,
    _class_weights,
    _model_frame,
    _prepare_text,
    _transform_text_features,
)


logger = logging.getLogger(__name__)
logger.setLevel(data_loader_logger.level)

THRESHOLDS_PATH = Path(MODELS_DIR) / "thresholds.json"
OPTUNA_SUMMARY_PATH = Path(EXPERIMENTS_DIR) / "optuna_summary.json"
ENSEMBLE_SUMMARY_PATH = Path(EXPERIMENTS_DIR) / "ensemble_summary.json"
RULES_PATH = Path(EXPERIMENTS_DIR) / "error_analysis.json"
SUBMISSION_PATH = Path(SUBMISSIONS_DIR) / "submission_final.csv"


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object or raise an actionable missing-file error."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _load_thresholds() -> Dict[str, float]:
    """Load class-specific probability thresholds."""
    return {str(key): float(value) for key, value in _load_json(THRESHOLDS_PATH).items()}


def _select_pipeline() -> Dict[str, Any]:
    """Select Optuna only when its recorded score beats the baseline ensemble."""
    baseline_summary = _load_json(ENSEMBLE_SUMMARY_PATH)
    optuna_summary = _load_json(OPTUNA_SUMMARY_PATH)
    baseline_score = float(
        baseline_summary.get("overall_ensemble_macro_f1", 0.9108)
    )
    tuned_score = float(optuna_summary.get("best_value_macro_f1", 0.0))
    use_tuned = tuned_score > baseline_score
    selected = "optuna" if use_tuned else "baseline"
    logger.info(
        "Pipeline selection: %s (baseline=%.4f, Optuna=%.4f)",
        selected,
        baseline_score,
        tuned_score,
    )
    return {
        "selected": selected,
        "baseline_score": baseline_score,
        "optuna_score": tuned_score,
        "used_tuned_parameters": use_tuned,
    }


def _rule_items() -> List[Tuple[re.Pattern[str], str, str]]:
    """Return explicit keyword rules plus useful rules from error analysis."""
    rules: List[Tuple[re.Pattern[str], str, str]] = [
        (re.compile(r"\b(?:IRS|TAX)\b", re.I), "Bills & Utilities", "tax keyword"),
        (re.compile(r"\b(?:UBER|LYFT)\b", re.I), "Transportation", "ride-share keyword"),
    ]
    report = _load_json(RULES_PATH)
    for item in report.get("suggested_rules", []):
        rule_text = str(item.get("rule", ""))
        category = item.get("suggested_category")
        if not category:
            continue
        keywords = re.findall(r"[A-Za-z][A-Za-z &]+", rule_text)
        for keyword in keywords:
            keyword = keyword.strip()
            if keyword.upper() in {"IN DESCRIPTION", "DESCRIPTION"} or len(keyword) < 3:
                continue
            pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.I)
            rules.append((pattern, str(category), f"error-analysis rule: {rule_text}"))
    unique_rules = []
    seen = set()
    for pattern, category, reason in rules:
        key = (pattern.pattern, category)
        if key not in seen:
            seen.add(key)
            unique_rules.append((pattern, category, reason))
    return unique_rules


def apply_rules(df: pd.DataFrame, predictions: Iterable[str]) -> np.ndarray:
    """Apply deterministic keyword rules to predicted transaction categories."""
    if "description" not in df.columns:
        raise ValueError("Rule post-processing requires a description column")
    corrected = np.asarray(list(predictions), dtype=object).copy()
    if len(corrected) != len(df):
        raise ValueError("The number of predictions must equal the number of rows")
    descriptions = df["description"].fillna("").astype(str).to_numpy()
    for pattern, category, _ in _rule_items():
        mask = np.array([bool(pattern.search(text)) for text in descriptions])
        corrected[mask] = category
    return corrected


def _prepare_frames(
    raw_df: pd.DataFrame, text_bundle: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Engineer features and create CatBoost and LightGBM model frames."""
    engineered = engineer_features(raw_df, is_train=TARGET_COLUMN in raw_df.columns)
    text = _prepare_text(engineered)
    text_features = _transform_text_features(
        text,
        text_bundle["char_vectorizer"],
        text_bundle["word_vectorizer"],
        text_bundle["svd"],
    )
    cat_frame, categorical = _model_frame(engineered, text_features)
    lgb_frame = _numeric_frame(cat_frame, categorical)
    return engineered, cat_frame, lgb_frame, categorical


def _validation_predictions(
    engineered: pd.DataFrame,
    cat_frame: pd.DataFrame,
    lgb_frame: pd.DataFrame,
    categorical: List[str],
    thresholds: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate leakage-free baseline OOF predictions for the validation summary."""
    labels = engineered[TARGET_COLUMN].astype(str).to_numpy()
    classes = np.array(sorted(np.unique(labels)))
    encoded = np.array([np.where(classes == label)[0][0] for label in labels])
    weights = _class_weights(pd.Series(labels))
    cat_indices = [cat_frame.columns.get_loc(column) for column in categorical]
    splitter = TimeSeriesSplit(n_splits=5)
    actual_parts, cat_parts, ensemble_parts = [], [], []
    for fold, (train_indices, valid_indices) in enumerate(splitter.split(cat_frame), start=1):
        cat_model = _build_model(weights)
        cat_model.fit(
            cat_frame.iloc[train_indices],
            labels[train_indices],
            cat_features=cat_indices,
            eval_set=(cat_frame.iloc[valid_indices], labels[valid_indices]),
            use_best_model=False,
        )
        cat_probabilities = _ordered_probabilities(
            cat_model.predict_proba(cat_frame.iloc[valid_indices]),
            np.asarray(cat_model.classes_),
            classes,
        )
        lgb_model = _build_lightgbm()
        lgb_model.fit(
            lgb_frame.iloc[train_indices],
            encoded[train_indices],
            eval_X=[lgb_frame.iloc[valid_indices].to_numpy()],
            eval_y=encoded[valid_indices],
            callbacks=[],
        )
        lgb_probabilities = _ordered_probabilities(
            lgb_model.predict_proba(lgb_frame.iloc[valid_indices]),
            np.asarray(lgb_model.classes_),
            classes,
        )
        ensemble_predictions = _apply_thresholds(
            (cat_probabilities + lgb_probabilities) / 2.0, classes, thresholds
        )
        actual_parts.append(labels[valid_indices])
        cat_parts.append(_apply_thresholds(cat_probabilities, classes, thresholds))
        ensemble_parts.append(ensemble_predictions)
        logger.info("Validation fold %d complete", fold)
    return (
        np.concatenate(actual_parts),
        np.concatenate(cat_parts),
        np.concatenate(ensemble_parts),
    )


def _predict_test(
    engineered_test: pd.DataFrame,
    cat_frame: pd.DataFrame,
    lgb_frame: pd.DataFrame,
    classes: np.ndarray,
    thresholds: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict test categories using the persisted full-data baseline models."""
    if not CATBOOST_MODEL_PATH.is_file() or not LIGHTGBM_MODEL_PATH.is_file():
        raise FileNotFoundError("Both persisted CatBoost and LightGBM models are required")
    cat_model = joblib.load(CATBOOST_MODEL_PATH)
    lgb_model = joblib.load(LIGHTGBM_MODEL_PATH)
    cat_probabilities = _ordered_probabilities(
        cat_model.predict_proba(cat_frame), np.asarray(cat_model.classes_), classes
    )
    lgb_probabilities = _ordered_probabilities(
        lgb_model.predict_proba(lgb_frame), np.asarray(lgb_model.classes_), classes
    )
    cat_predictions = _apply_thresholds(cat_probabilities, classes, thresholds)
    ensemble_predictions = _apply_thresholds(
        (cat_probabilities + lgb_probabilities) / 2.0, classes, thresholds
    )
    return cat_predictions, ensemble_predictions


def run_final_pipeline() -> Dict[str, Any]:
    """Evaluate the selected pipeline, post-process rules, and write submission."""
    for directory in (Path(MODELS_DIR), Path(EXPERIMENTS_DIR), Path(SUBMISSIONS_DIR)):
        directory.mkdir(parents=True, exist_ok=True)
    selection = _select_pipeline()
    thresholds = _load_thresholds()

    loader = DataLoader(use_cache=False)
    raw_train = loader.load_train_data(force_reload=True).sort_values("transaction_id").reset_index(drop=True)
    train_engineered_for_text = engineer_features(raw_train, is_train=True)
    text_bundle = _load_text_artifacts(_prepare_text(train_engineered_for_text))
    engineered_train, train_cat, train_lgb, categorical = _prepare_frames(raw_train, text_bundle)
    actual, cat_predictions, ensemble_predictions = _validation_predictions(
        engineered_train, train_cat, train_lgb, categorical, thresholds
    )
    ensemble_after_rules = apply_rules(
        engineered_train.iloc[-len(ensemble_predictions):], ensemble_predictions
    )
    cat_score = f1_score(actual, cat_predictions, average="macro", zero_division=0)
    ensemble_score = f1_score(actual, ensemble_predictions, average="macro", zero_division=0)
    rules_score = f1_score(actual, ensemble_after_rules, average="macro", zero_division=0)

    raw_test = loader.load_test_data(force_reload=True).sort_values("transaction_id").reset_index(drop=True)
    engineered_test, test_cat, test_lgb, _ = _prepare_frames(raw_test, text_bundle)
    classes = np.array(sorted(raw_train[TARGET_COLUMN].astype(str).unique()))
    cat_test_predictions, ensemble_test_predictions = _predict_test(
        engineered_test, test_cat, test_lgb, classes, thresholds
    )
    rule_test_predictions = apply_rules(engineered_test, ensemble_test_predictions)
    if rules_score > max(cat_score, ensemble_score):
        final_strategy = "ensemble_with_rules"
        test_predictions = rule_test_predictions
    elif cat_score >= ensemble_score:
        final_strategy = "catboost"
        test_predictions = cat_test_predictions
    else:
        final_strategy = "ensemble"
        test_predictions = ensemble_test_predictions
    submission = pd.DataFrame({"transaction_id": raw_test["transaction_id"], TARGET_COLUMN: test_predictions})
    submission.to_csv(SUBMISSION_PATH, index=False)

    result = {
        "selection": selection,
        "validation": {
            "catboost_macro_f1": float(cat_score),
            "ensemble_macro_f1_before_rules": float(ensemble_score),
            "ensemble_macro_f1_after_rules": float(rules_score),
            "rule_impact": float(rules_score - ensemble_score),
            "final_strategy": final_strategy,
        },
        "submission": {
            "path": str(SUBMISSION_PATH),
            "shape": list(submission.shape),
            "distribution": submission[TARGET_COLUMN].value_counts().to_dict(),
        },
    }
    print("=" * 70)
    print("AuroraGate Final Pipeline")
    print("=" * 70)
    print(f"Selected pipeline: {selection['selected']}")
    print(f"CatBoost validation macro F1: {cat_score:.4f}")
    print(f"Ensemble before rules: {ensemble_score:.4f}")
    print(f"Ensemble after rules: {rules_score:.4f}")
    print(f"Final submission strategy: {final_strategy}")
    print(f"Submission shape: {submission.shape}")
    print("Submission distribution:")
    print(submission[TARGET_COLUMN].value_counts().to_string())
    print(f"Saved submission: {SUBMISSION_PATH}")
    print("=" * 70)
    return result


if __name__ == "__main__":
    run_final_pipeline()