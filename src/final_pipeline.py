"""Select the best AuroraGate pipeline, apply rules, and create a submission."""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from src.config import EXPERIMENTS_DIR, MODELS_DIR, SUBMISSIONS_DIR, TARGET_COLUMN, RANDOM_STATE
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
    _model_frame,
    _prepare_text,
    _transform_text_features,
)
from src.training_utils import compute_class_weights


logger = get_logger(__name__)

THRESHOLDS_PATH = Path(MODELS_DIR) / "thresholds.json"
OPTUNA_SUMMARY_PATH = Path(EXPERIMENTS_DIR) / "optuna_summary.json"
ENSEMBLE_SUMMARY_PATH = Path(EXPERIMENTS_DIR) / "ensemble_summary.json"
RULES_PATH = Path(EXPERIMENTS_DIR) / "error_analysis.json"
SUBMISSION_PATH = Path(SUBMISSIONS_DIR) / "submission_final.csv"
LOGISTIC_MODEL_PATH = Path(MODELS_DIR) / "logistic_regression_model.pkl"
WEAK_CLASSES = ("Subscriptions", "Entertainment", "Miscellaneous")


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


SURGICAL_CLASS_NAMES = (
    "Bills & Utilities",
    "Entertainment",
    "Food & Dining",
    "Groceries",
    "Health & Fitness",
    "Miscellaneous",
    "Shopping",
    "Subscriptions",
    "Transportation",
    "Travel",
)


def apply_surgical_rules(
    df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    confidence_threshold: float = 0.6,
) -> np.ndarray:
    """Apply narrow keyword corrections only to low-confidence predictions."""
    if "description" not in df.columns:
        raise ValueError("Rule post-processing requires a description column")
    final_predictions = np.asarray(predictions, dtype=object).copy()
    if len(final_predictions) != len(df) or len(probabilities) != len(df):
        raise ValueError("The number of predictions must equal the number of rows")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(SURGICAL_CLASS_NAMES):
        raise ValueError("Probabilities must have one column per target category")

    rule_keywords = (
        ("Subscriptions", (
    "NETFLIX", "SPOTIFY", "APPLE.COM/BILL", 
    "AMAZON PRIME", "HULU", "DISNEY+", "HBOMAX"
)),
        ("Miscellaneous", ("ZELLE", "VENMO", "PAYPAL", "TRANSFER")),
        ("Transportation", ("UBER", "LYFT", "TAXI")),
        ("Entertainment", ("CINEMA", "MOVIE", "CONCERT", "THEATER")),
    )
    descriptions = df["description"].fillna("").astype(str).str.upper().to_numpy()
    low_confidence = np.max(probabilities, axis=1) < confidence_threshold
    for row_index, description in enumerate(descriptions):
        if not low_confidence[row_index]:
            continue
        for category, keywords in rule_keywords:
            if any(keyword in description for keyword in keywords):
                final_predictions[row_index] = category
                break
    return final_predictions


def find_optimal_confidence_threshold(
    df: pd.DataFrame,
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: Iterable[str],
) -> Tuple[float, float]:
    """Find the confidence threshold that maximizes validation Macro F1."""
    del class_names
    best_threshold = 0.6
    best_f1 = f1_score(y_true, predictions, average="macro", zero_division=0)
    for threshold in np.arange(0.3, 0.85, 0.05):
        candidate = apply_surgical_rules(
            df,
            predictions,
            probabilities,
            confidence_threshold=float(threshold),
        )
        score = f1_score(y_true, candidate, average="macro", zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, float(best_f1)


def audit_feature_importance(
    model: Any, feature_names: Iterable[str], model_name: str = "CatBoost"
) -> pd.DataFrame:
    """Print and return the ten most important model features."""
    if hasattr(model, "get_feature_importance"):
        importances = np.asarray(model.get_feature_importance())
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    else:
        raise TypeError(f"{model_name} does not expose feature importances")
    names = list(feature_names)
    if len(importances) != len(names):
        raise ValueError(
            f"{model_name} importance count ({len(importances)}) does not match "
            f"feature count ({len(names)})"
        )
    importance_df = pd.DataFrame(
        {"feature": names, "importance": importances}
    ).sort_values("importance", ascending=False)
    print(f"\n=== Top 10 features for {model_name} ===")
    print(importance_df.head(10).to_string(index=False))
    return importance_df


def plot_fold5_confusion_matrix(
    y_true_5: np.ndarray,
    y_pred_5: np.ndarray,
    class_names: Iterable[str],
    save_path: Path = Path(EXPERIMENTS_DIR) / "fold5_confusion_matrix.png",
) -> np.ndarray:
    """Save and return the confusion matrix for the final time-series fold."""
    labels = list(class_names)
    matrix = pd.crosstab(
        pd.Series(y_true_5, name="Actual"),
        pd.Series(y_pred_5, name="Predicted"),
    ).reindex(index=labels, columns=labels, fill_value=0).to_numpy()
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix - Fold 5 (Closest to Test Set)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Fold 5 Confusion Matrix saved to {output_path}")
    return matrix


def train_logistic_regression(
    text_features: np.ndarray, labels: np.ndarray
) -> LogisticRegression:
    """Train the TF-IDF/SVD Logistic Regression ensemble component."""
    model = LogisticRegression(
        max_iter=1000,
        C=0.1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(text_features, labels)
    return model


def optimize_catboost_lr_weights(
    catboost_proba: np.ndarray,
    lr_proba: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[float, float]:
    """Optimize CatBoost and Logistic Regression blend weights."""
    def objective(weight: np.ndarray) -> float:
        cat_weight = float(weight[0])
        blended = cat_weight * catboost_proba + (1.0 - cat_weight) * lr_proba
        predictions = np.argmax(blended, axis=1)
        return -f1_score(y_true, predictions, average="macro", zero_division=0)

    result = minimize(objective, x0=[0.7], bounds=[(0.0, 1.0)], method="Powell")
    cat_weight = float(np.clip(result.x[0], 0.0, 1.0))
    lr_weight = 1.0 - cat_weight
    print(
        f"Optimized weights: CatBoost={cat_weight:.3f}, "
        f"LogisticRegression={lr_weight:.3f}"
    )
    return cat_weight, lr_weight


def optimize_ensemble_weights(
    cat_proba: np.ndarray, lgb_proba: np.ndarray, y_true: np.ndarray
) -> Tuple[float, float]:
    """Optimize CatBoost and Logistic Regression blend weights on validation probabilities."""
    def objective(weight: np.ndarray) -> float:
        w1 = float(weight[0])
        blended = w1 * cat_proba + (1.0 - w1) * lgb_proba
        predictions = np.argmax(blended, axis=1)
        return -f1_score(y_true, predictions, average="macro", zero_division=0)

    result = minimize(objective, x0=[0.5], bounds=[(0.0, 1.0)], method="Powell")
    cat_weight = float(np.clip(result.x[0], 0.0, 1.0))
    lgb_weight = 1.0 - cat_weight
    print(
        f"Optimized weights: CatBoost={cat_weight:.3f}, "
        f"LogisticRegression={lgb_weight:.3f}"
    )
    return cat_weight, lgb_weight


def optimize_class_thresholds(
    y_true: np.ndarray,
    oof_probas: np.ndarray,
    class_names: list,
    target_classes: list = ["Subscriptions", "Entertainment", "Miscellaneous"],
    base_threshold: float = 0.5,
) -> Dict[str, float]:
    """Optimize probability scaling thresholds for selected classes."""
    labels = np.asarray(class_names, dtype=object)
    if oof_probas.ndim != 2 or oof_probas.shape[1] != len(labels):
        raise ValueError("OOF probabilities must have one column per class")
    if len(y_true) != len(oof_probas):
        raise ValueError("OOF labels and probabilities must have matching rows")
    if not 0.0 < base_threshold < 1.0:
        raise ValueError("base_threshold must be between zero and one")

    best_thresholds = {str(label): float(base_threshold) for label in labels}
    baseline_predictions = labels[np.argmax(oof_probas, axis=1)]
    baseline_f1 = f1_score(
        y_true, baseline_predictions, average="macro", zero_division=0
    )
    for class_name in target_classes:
        if class_name not in labels:
            logger.warning("Skipping threshold optimization for unknown class: %s", class_name)
            continue
        class_index = int(np.where(labels == class_name)[0][0])
        best_f1 = baseline_f1
        best_threshold = base_threshold
        for threshold in np.arange(0.20, 0.80, 0.02):
            adjusted = oof_probas.copy()
            adjusted[:, class_index] *= base_threshold / float(threshold)
            candidate_predictions = labels[np.argmax(adjusted, axis=1)]
            score = f1_score(
                y_true, candidate_predictions, average="macro", zero_division=0
            )
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(threshold)
        best_thresholds[str(class_name)] = best_threshold
        print(
            f"Optimal threshold for {class_name}: {best_threshold:.2f} "
            f"(Local F1 impact: {best_f1 - baseline_f1:+.4f})"
        )
    return best_thresholds


def _apply_optimized_class_thresholds(
    probabilities: np.ndarray,
    classes: np.ndarray,
    class_thresholds: Dict[str, float],
    base_threshold: float = 0.5,
) -> np.ndarray:
    """Apply optimized class scaling and return string category predictions."""
    adjusted = probabilities.copy()
    for index, label in enumerate(classes):
        threshold = float(class_thresholds.get(str(label), base_threshold))
        adjusted[:, index] *= base_threshold / threshold
    return classes[np.argmax(adjusted, axis=1)]


def optimize_weights(
    cat_proba: np.ndarray, lgb_proba: np.ndarray, y_true: np.ndarray
) -> float:
    """Backward-compatible wrapper returning only the CatBoost weight."""
    return optimize_ensemble_weights(cat_proba, lgb_proba, y_true)[0]


def optimize_thresholds_for_weak_classes(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    weak_classes: Iterable[str],
    classes: np.ndarray,
) -> Dict[str, float]:
    """Tune probability thresholds for weak classes against validation labels."""
    class_to_idx = {str(label): index for index, label in enumerate(classes)}
    weak_indices = [class_to_idx[label] for label in weak_classes if label in class_to_idx]
    if not weak_indices:
        return {}

    def objective(thresholds: np.ndarray) -> float:
        predictions = np.argmax(y_proba, axis=1)
        for threshold, class_index in zip(thresholds, weak_indices):
            mask = y_proba[:, class_index] > threshold
            predictions[mask] = class_index
        return -f1_score(y_true, predictions, average="macro", zero_division=0)

    result = minimize(
        objective,
        x0=np.full(len(weak_indices), 0.5),
        bounds=[(0.1, 0.9)] * len(weak_indices),
        method="Powell",
    )
    return {
        str(classes[class_index]): float(np.clip(threshold, 0.1, 0.9))
        for threshold, class_index in zip(result.x, weak_indices)
    }


def _apply_weak_thresholds(
    probabilities: np.ndarray,
    classes: np.ndarray,
    base_thresholds: Dict[str, float],
    weak_thresholds: Dict[str, float],
) -> np.ndarray:
    """Apply saved thresholds, then allow tuned weak classes to override argmax."""
    predictions = _apply_thresholds(probabilities, classes, base_thresholds)
    for label, threshold in weak_thresholds.items():
        class_indices = np.where(classes == label)[0]
        if len(class_indices):
            predictions[probabilities[:, class_indices[0]] > threshold] = label
    return predictions


def _prepare_frames(
    raw_df: pd.DataFrame,
    text_bundle: Dict[str, Any],
    target_encoding_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], np.ndarray]:
    """Engineer features and create CatBoost and LightGBM model frames."""
    ordered = raw_df.sort_values("transaction_id").reset_index(drop=True).copy()
    transaction_ids = pd.to_numeric(ordered["transaction_id"], errors="coerce")
    dates = pd.to_datetime(ordered["date"], errors="coerce")
    ordered["time_since_last_transaction"] = transaction_ids.diff().fillna(0)
    ordered["transaction_count_per_day"] = (
        ordered.groupby(dates.dt.normalize(), dropna=False).cumcount() + 1
    )
    engineered = engineer_features(
        ordered,
        is_train=TARGET_COLUMN in ordered.columns,
        target_encoding_stats=target_encoding_stats,
    )
    text = _prepare_text(engineered)
    text_features = _transform_text_features(
        text,
        text_bundle["char_vectorizer"],
        text_bundle["word_vectorizer"],
        text_bundle["svd"],
    )
    cat_frame, categorical = _model_frame(engineered, text_features)
    lgb_frame = _numeric_frame(cat_frame, categorical)
    return engineered, cat_frame, lgb_frame, categorical, text_features


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


def _model_matches_frame(model: Any, frame: pd.DataFrame) -> bool:
    """Return whether a persisted model has the current feature schema."""
    expected = getattr(model, "feature_names_", None) or getattr(
        model, "feature_name_", None
    )
    return bool(expected) and list(expected) == list(frame.columns)


def _fit_full_models(
    engineered: pd.DataFrame,
    cat_frame: pd.DataFrame,
    lgb_frame: pd.DataFrame,
    categorical: List[str],
    classes: np.ndarray,
) -> Tuple[Any, Any]:
    """Train and persist models when existing artifacts use an older schema."""
    labels = engineered[TARGET_COLUMN].astype(str).to_numpy()
    encoded = np.array([np.where(classes == label)[0][0] for label in labels])
    weights = compute_class_weights(pd.Series(labels))
    cat_indices = [cat_frame.columns.get_loc(column) for column in categorical]
    cat_model = _build_model(weights)
    try:
        cat_model.fit(cat_frame, labels, cat_features=cat_indices)
    except Exception as error:
        logger.error("Training failed: %s", error)
        logger.info("Falling back to default parameters...")
        cat_model = _build_model(weights)
        cat_model.fit(cat_frame, labels, cat_features=cat_indices)

    lgb_model = _build_lightgbm()
    lgb_model.fit(lgb_frame, encoded)
    joblib.dump(cat_model, CATBOOST_MODEL_PATH)
    joblib.dump(lgb_model, LIGHTGBM_MODEL_PATH)
    joblib.dump(
        {
            "feature_columns": lgb_frame.columns.tolist(),
            "categorical_columns": categorical,
            "classes": classes.tolist(),
        },
        LIGHTGBM_MODEL_PATH.with_name("lightgbm_metadata.pkl"),
    )
    logger.info("Persisted models retrained with the current feature schema")
    return cat_model, lgb_model


def _load_or_train_models(
    engineered: pd.DataFrame,
    cat_frame: pd.DataFrame,
    lgb_frame: pd.DataFrame,
    categorical: List[str],
    classes: np.ndarray,
) -> Tuple[Any, Any]:
    """Load compatible artifacts or retrain them with the current features."""
    if CATBOOST_MODEL_PATH.is_file() and LIGHTGBM_MODEL_PATH.is_file():
        cat_model = joblib.load(CATBOOST_MODEL_PATH)
        lgb_model = joblib.load(LIGHTGBM_MODEL_PATH)
        if (_model_matches_frame(cat_model, cat_frame) and
        _model_matches_frame(lgb_model, lgb_frame)):
            return cat_model, lgb_model
        logger.info("Persisted model schema is outdated; retraining with new features")
    return _fit_full_models(engineered, cat_frame, lgb_frame, categorical, classes)


def _validation_predictions(
    engineered: pd.DataFrame,
    cat_frame: pd.DataFrame,
    text_features: np.ndarray,
    categorical: List[str],
    thresholds: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate leakage-free baseline OOF predictions for the validation summary."""
    labels = engineered[TARGET_COLUMN].astype(str).to_numpy()
    classes = np.array(sorted(np.unique(labels)))
    encoded = np.array([np.where(classes == label)[0][0] for label in labels])
    weights = compute_class_weights(pd.Series(labels))
    cat_indices = [cat_frame.columns.get_loc(column) for column in categorical]
    splitter = TimeSeriesSplit(n_splits=5)
    actual_parts, cat_parts, lr_parts, frame_parts = [], [], [], []
    fold5_actual = np.array([], dtype=object)
    fold5_predictions = np.array([], dtype=object)
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
        lr_model = train_logistic_regression(
            text_features[train_indices],
            labels[train_indices],
        )
        lr_probabilities = _ordered_probabilities(
            lr_model.predict_proba(text_features[valid_indices]),
            np.asarray(lr_model.classes_),
            classes,
        )
        if fold == 5:
            fold5_actual = labels[valid_indices]
            fold5_predictions = classes[np.argmax(cat_probabilities, axis=1)]
        actual_parts.append(labels[valid_indices])
        cat_parts.append(cat_probabilities)
        lr_parts.append(lr_probabilities)
        frame_parts.append(engineered.iloc[valid_indices])
        logger.info("Validation fold %d complete", fold)
    return (
        np.concatenate(actual_parts),
        np.concatenate(cat_parts),
        np.concatenate(lr_parts),
        pd.concat(frame_parts, axis=0).reset_index(drop=True),
        fold5_actual,
        fold5_predictions,
    )


def _predict_test(
    engineered_test: pd.DataFrame,
    cat_frame: pd.DataFrame,
    text_features: np.ndarray,
    classes: np.ndarray,
    thresholds: Dict[str, float],
    catboost_weight: float,
    weak_thresholds: Dict[str, float],
    cat_model: Any,
    lr_model: LogisticRegression,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict test categories using CatBoost and Logistic Regression."""
    if not CATBOOST_MODEL_PATH.is_file():
        raise FileNotFoundError("The persisted CatBoost model is required")
    if not _model_matches_frame(cat_model, cat_frame):
        cat_frame = _align_persisted_frame(cat_frame, cat_model, engineered_test)
    cat_probabilities = _ordered_probabilities(
        cat_model.predict_proba(cat_frame), np.asarray(cat_model.classes_), classes
    )
    lr_probabilities = _ordered_probabilities(
        lr_model.predict_proba(text_features),
        np.asarray(lr_model.classes_),
        classes,
    )
    cat_predictions = _apply_thresholds(cat_probabilities, classes, thresholds)
    ensemble_probabilities = (
        catboost_weight * cat_probabilities
        + (1.0 - catboost_weight) * lr_probabilities
    )
    ensemble_predictions = _apply_weak_thresholds(
        ensemble_probabilities, classes, thresholds, weak_thresholds
    )
    return cat_predictions, ensemble_predictions, ensemble_probabilities


def _load_or_train_logistic(
    text_features: np.ndarray, labels: np.ndarray
) -> LogisticRegression:
    """Load the full-data Logistic Regression model or train and persist it."""
    if LOGISTIC_MODEL_PATH.is_file():
        model = joblib.load(LOGISTIC_MODEL_PATH)
        if isinstance(model, LogisticRegression):
            return model
    model = train_logistic_regression(text_features, labels)
    LOGISTIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, LOGISTIC_MODEL_PATH)
    return model


def run_final_pipeline() -> Dict[str, Any]:
    """Evaluate the selected pipeline, post-process rules, and write submission."""
    for directory in (Path(MODELS_DIR), Path(EXPERIMENTS_DIR), Path(SUBMISSIONS_DIR)):
        directory.mkdir(parents=True, exist_ok=True)
    selection = _select_pipeline()
    thresholds = _load_thresholds()

    loader = DataLoader(use_cache=False)
    raw_train = (
    loader.load_train_data(force_reload=True)
    .sort_values("transaction_id")
    .reset_index(drop=True)
)
    train_engineered_for_text = engineer_features(raw_train, is_train=True)
    text_bundle = _load_text_artifacts(_prepare_text(train_engineered_for_text))
    engineered_train, train_cat, train_lgb, categorical, train_text_features = _prepare_frames(
        raw_train, text_bundle
    )
    (
        actual,
        cat_probabilities,
        lr_probabilities,
        validation_frames,
        fold5_actual,
        fold5_predictions,
    ) = _validation_predictions(
        engineered_train, train_cat, train_text_features, categorical, thresholds
    )
    classes = np.array(sorted(raw_train[TARGET_COLUMN].astype(str).unique()))
    encoded_actual = np.array([np.where(classes == label)[0][0] for label in actual])
    fold5_confusion_matrix = plot_fold5_confusion_matrix(
        fold5_actual,
        fold5_predictions,
        classes,
    )
    print("Fold 5 confusion matrix:")
    print(fold5_confusion_matrix)
    w1, w2 = optimize_catboost_lr_weights(
        cat_probabilities, lr_probabilities, encoded_actual
    )
    ensemble_probabilities = w1 * cat_probabilities + w2 * lr_probabilities
    optimized_class_thresholds = optimize_class_thresholds(
        actual,
        ensemble_probabilities,
        classes.tolist(),
    )
    optimized_threshold_predictions = _apply_optimized_class_thresholds(
        ensemble_probabilities,
        classes,
        optimized_class_thresholds,
    )
    weak_thresholds = optimize_thresholds_for_weak_classes(
        encoded_actual,
        ensemble_probabilities,
        WEAK_CLASSES,
        classes,
    )
    cat_predictions = _apply_thresholds(cat_probabilities, classes, thresholds)
    ensemble_predictions = _apply_weak_thresholds(
        ensemble_probabilities,
        classes,
        thresholds,
        weak_thresholds,
    )
    with ENSEMBLE_WEIGHTS_PATH.open("w", encoding="utf-8") as weights_file:
        json.dump({"catboost": w1, "logistic_regression": w2}, weights_file, indent=2)
    validation_threshold, surgical_score = find_optimal_confidence_threshold(
        validation_frames,
        actual,
        ensemble_predictions,
        ensemble_probabilities,
        classes,
    )
    cat_score = f1_score(actual, cat_predictions, average="macro", zero_division=0)
    ensemble_score = f1_score(actual, ensemble_predictions, average="macro", zero_division=0)
    optimized_threshold_score = f1_score(
        actual,
        optimized_threshold_predictions,
        average="macro",
        zero_division=0,
    )
    surgical_validation_predictions = apply_surgical_rules(
        validation_frames,
        optimized_threshold_predictions,
        ensemble_probabilities,
        confidence_threshold=validation_threshold,
    )
    rules_score = f1_score(
        actual, surgical_validation_predictions, average="macro", zero_division=0
    )

    models = _load_or_train_models(
        engineered_train,
        train_cat,
        train_lgb,
        categorical,
        classes,
    )
    audit_feature_importance(models[0], train_cat.columns, "CatBoost")
    lr_model = _load_or_train_logistic(
        train_text_features,
        raw_train[TARGET_COLUMN].astype(str).to_numpy(),
    )

    raw_test = (
    loader.load_test_data(force_reload=True)
    .sort_values("transaction_id")
    .reset_index(drop=True)
)
    engineered_test, test_cat, test_lgb, _, test_text_features = _prepare_frames(
        raw_test,
        text_bundle,
    )
    cat_test_predictions, ensemble_test_predictions, ensemble_test_probabilities = _predict_test(
        engineered_test,
        test_cat,
        test_text_features,
        classes,
        thresholds,
        w1,
        weak_thresholds,
        models[0],
        lr_model,
    )
    optimized_test_predictions = _apply_optimized_class_thresholds(
        ensemble_test_probabilities,
        classes,
        optimized_class_thresholds,
    )
    rule_test_predictions = apply_surgical_rules(
        engineered_test,
        optimized_test_predictions,
        ensemble_test_probabilities,
        confidence_threshold=validation_threshold,
    )
    use_lr_ensemble = ensemble_score >= 0.9100 and ensemble_score > cat_score
    if use_lr_ensemble and rules_score > max(cat_score, ensemble_score, optimized_threshold_score):
        final_strategy = "weighted_ensemble_with_rules"
        test_predictions = rule_test_predictions
    elif use_lr_ensemble and optimized_threshold_score > max(cat_score, ensemble_score):
        final_strategy = "weighted_ensemble_with_class_thresholds"
        test_predictions = optimized_test_predictions
    elif not use_lr_ensemble or cat_score >= ensemble_score:
        final_strategy = "catboost"
        test_predictions = cat_test_predictions
    else:
        final_strategy = "weighted_ensemble"
        test_predictions = ensemble_test_predictions
    submission = pd.DataFrame({
    "transaction_id": raw_test["transaction_id"],
    TARGET_COLUMN: test_predictions
})
    submission.to_csv(SUBMISSION_PATH, index=False)

    result = {
        "selection": selection,
        "validation": {
            "catboost_macro_f1": float(cat_score),
            "ensemble_macro_f1_before_rules": float(ensemble_score),
            "optimized_threshold_macro_f1": float(optimized_threshold_score),
            "ensemble_macro_f1_after_rules": float(rules_score),
            "rule_impact": float(rules_score - ensemble_score),
            "weak_class_thresholds": weak_thresholds,
            "optimized_class_thresholds": optimized_class_thresholds,
            "surgical_rule_confidence_threshold": float(validation_threshold),
            "surgical_rule_validation_macro_f1": float(surgical_score),
            "final_strategy": final_strategy,
            "logistic_regression_enabled": use_lr_ensemble,
            "fold5_confusion_matrix_path": str(
                Path(EXPERIMENTS_DIR) / "fold5_confusion_matrix.png"
            ),
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
    print(
        f"Optimized ensemble weights: CatBoost={w1:.4f}, "
        f"LogisticRegression={w2:.4f}"
    )
    print(f"Optimized class thresholds: {optimized_class_thresholds}")
    print(f"Weak-class thresholds: {weak_thresholds}")
    print(
        f"Optimal surgical-rule confidence threshold: "
        f"{validation_threshold:.2f} (validation Macro F1: {surgical_score:.4f})"
    )
    print(f"Ensemble before rules: {ensemble_score:.4f}")
    print(f"Validation F1 with optimized class thresholds: {optimized_threshold_score:.4f}")
    print(f"Ensemble after rules: {rules_score:.4f}")
    print(f"Logistic Regression ensemble enabled: {use_lr_ensemble}")
    print(f"Final submission strategy: {final_strategy}")
    print(f"Submission shape: {submission.shape}")
    print("Submission distribution:")
    print(submission[TARGET_COLUMN].value_counts().to_string())
    print(f"Saved submission: {SUBMISSION_PATH}")
    print("=" * 70)
    return result


if __name__ == "__main__":
    run_final_pipeline()