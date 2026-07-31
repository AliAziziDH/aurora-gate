"""Select the best AuroraGate pipeline, apply rules, and create a submission."""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import EXPERIMENTS_DIR, MODELS_DIR, SUBMISSIONS_DIR, TARGET_COLUMN
from src.data_loader import DataLoader
from src.feature_engineering import engineer_features
from src.logger import get_logger
from src.train_lightgbm import (
    ENSEMBLE_WEIGHTS_PATH,
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


logger = get_logger(__name__)

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


def optimize_weights(
    cat_proba: np.ndarray, lgb_proba: np.ndarray, y_true: np.ndarray
) -> float:
    """Find the CatBoost probability weight that maximizes validation macro F1."""
    def objective(weight: np.ndarray) -> float:
        w1 = float(weight[0])
        blended = w1 * cat_proba + (1.0 - w1) * lgb_proba
        predictions = np.argmax(blended, axis=1)
        return -f1_score(y_true, predictions, average="macro", zero_division=0)

    result = minimize(objective, x0=[0.5], bounds=[(0.0, 1.0)], method="Nelder-Mead")
    return float(np.clip(result.x[0], 0.0, 1.0))


def _prepare_frames(
    raw_df: pd.DataFrame, text_bundle: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Engineer features and create CatBoost and LightGBM model frames."""
    ordered = raw_df.sort_values("transaction_id").reset_index(drop=True).copy()
    transaction_ids = pd.to_numeric(ordered["transaction_id"], errors="coerce")
    dates = pd.to_datetime(ordered["date"], errors="coerce")
    ordered["time_since_last_transaction"] = transaction_ids.diff().fillna(0)
    ordered["transaction_count_per_day"] = (
        ordered.groupby(dates.dt.normalize(), dropna=False).cumcount() + 1
    )
    engineered = engineer_features(ordered, is_train=TARGET_COLUMN in ordered.columns)
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


def _align_persisted_frame(
    frame: pd.DataFrame, model: Any, engineered: pd.DataFrame
) -> pd.DataFrame:
    """Align inference columns with current or legacy persisted model artifacts."""
    expected = getattr(model, "feature_names_", None) or getattr(
        model, "feature_name_", None
    )
    if not expected:
        return frame
    aligned = frame.copy()
    if "transaction_id" in expected and "transaction_id" not in aligned.columns:
        aligned["transaction_id"] = engineered["transaction_id"].to_numpy()
    missing = set(expected) - set(aligned.columns)
    if missing:
        raise ValueError(f"Persisted model requires missing features: {sorted(missing)}")
    return aligned.loc[:, expected]


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
    actual_parts, cat_parts, lgb_parts = [], [], []
    for fold, (train_indices, valid_indices) in enumerate(splitter.split(cat_frame), start=1):
        cat_model = _build_model(weights)
        try:
            cat_model.fit(
                cat_frame.iloc[train_indices],
                labels[train_indices],
                cat_features=cat_indices,
                eval_set=(cat_frame.iloc[valid_indices], labels[valid_indices]),
                use_best_model=False,
            )
        except Exception as error:
            logger.error("Training failed: %s", error)
            logger.info("Falling back to default parameters...")
            cat_model = _build_model(weights)
            cat_model.fit(
                cat_frame.iloc[train_indices],
                labels[train_indices],
                cat_features=cat_indices,
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
        actual_parts.append(labels[valid_indices])
        cat_parts.append(cat_probabilities)
        lgb_parts.append(lgb_probabilities)
        logger.info("Validation fold %d complete", fold)
    return (
        np.concatenate(actual_parts),
        np.concatenate(cat_parts),
        np.concatenate(lgb_parts),
    )


def _predict_test(
    engineered_test: pd.DataFrame,
    cat_frame: pd.DataFrame,
    lgb_frame: pd.DataFrame,
    classes: np.ndarray,
    thresholds: Dict[str, float],
    catboost_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict test categories using the persisted full-data baseline models."""
    if not CATBOOST_MODEL_PATH.is_file() or not LIGHTGBM_MODEL_PATH.is_file():
        raise FileNotFoundError("Both persisted CatBoost and LightGBM models are required")
    cat_model = joblib.load(CATBOOST_MODEL_PATH)
    lgb_model = joblib.load(LIGHTGBM_MODEL_PATH)
    cat_frame = _align_persisted_frame(cat_frame, cat_model, engineered_test)
    lgb_frame = _align_persisted_frame(lgb_frame, lgb_model, engineered_test)
    cat_probabilities = _ordered_probabilities(
        cat_model.predict_proba(cat_frame), np.asarray(cat_model.classes_), classes
    )
    lgb_probabilities = _ordered_probabilities(
        lgb_model.predict_proba(lgb_frame), np.asarray(lgb_model.classes_), classes
    )
    cat_predictions = _apply_thresholds(cat_probabilities, classes, thresholds)
    ensemble_probabilities = (
        catboost_weight * cat_probabilities
        + (1.0 - catboost_weight) * lgb_probabilities
    )
    ensemble_predictions = _apply_thresholds(ensemble_probabilities, classes, thresholds)
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
    actual, cat_probabilities, lgb_probabilities = _validation_predictions(
        engineered_train, train_cat, train_lgb, categorical, thresholds
    )
    classes = np.array(sorted(raw_train[TARGET_COLUMN].astype(str).unique()))
    encoded_actual = np.array([np.where(classes == label)[0][0] for label in actual])
    w1 = optimize_weights(cat_probabilities, lgb_probabilities, encoded_actual)
    w2 = 1.0 - w1
    ensemble_probabilities = w1 * cat_probabilities + w2 * lgb_probabilities
    cat_predictions = _apply_thresholds(cat_probabilities, classes, thresholds)
    ensemble_predictions = _apply_thresholds(
        ensemble_probabilities, classes, thresholds
    )
    with ENSEMBLE_WEIGHTS_PATH.open("w", encoding="utf-8") as weights_file:
        json.dump({"catboost": w1, "lightgbm": w2}, weights_file, indent=2)
    ensemble_after_rules = apply_rules(
        engineered_train.iloc[-len(ensemble_predictions):], ensemble_predictions
    )
    cat_score = f1_score(actual, cat_predictions, average="macro", zero_division=0)
    ensemble_score = f1_score(actual, ensemble_predictions, average="macro", zero_division=0)
    rules_score = f1_score(actual, ensemble_after_rules, average="macro", zero_division=0)

    raw_test = loader.load_test_data(force_reload=True).sort_values("transaction_id").reset_index(drop=True)
    engineered_test, test_cat, test_lgb, _ = _prepare_frames(raw_test, text_bundle)
    cat_test_predictions, ensemble_test_predictions = _predict_test(
        engineered_test, test_cat, test_lgb, classes, thresholds, w1
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
    print(f"Optimized ensemble weights: CatBoost={w1:.4f}, LightGBM={w2:.4f}")
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