"""Validation-fold error analysis for the AuroraGate classifier."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
from src.config import EXPERIMENTS_DIR, TARGET_COLUMN

matplotlib_config_dir = Path(EXPERIMENTS_DIR) / ".matplotlib"
matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit

from src.data_loader import DataLoader
from src.feature_engineering import engineer_features
from src.train_model import (
    MODEL_PATH,
    THRESHOLDS_PATH,
    VECTORIZER_PATH,
    _apply_thresholds,
    _build_model,
    _model_frame,
    _prepare_text,
    _transform_text_features,
    _class_weights,
)
from src.logger import get_logger


logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


def _json_default(value: Any) -> Any:
    """Convert NumPy and pandas values to JSON-compatible values."""
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (np.ndarray, pd.Series, pd.Index)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _require_files() -> None:
    """Validate that all model artifacts required for analysis exist."""
    required = [MODEL_PATH, VECTORIZER_PATH, THRESHOLDS_PATH]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts. Run `python -m src.train_model` first: "
            + ", ".join(missing)
        )


def _load_artifacts() -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """Load the final model, vectorizers/SVD bundle and class thresholds."""
    _require_files()
    model = joblib.load(MODEL_PATH)
    vectorizers = joblib.load(VECTORIZER_PATH)
    with Path(THRESHOLDS_PATH).open("r", encoding="utf-8") as threshold_file:
        thresholds = {str(key): float(value) for key, value in json.load(threshold_file).items()}
    required_vectorizer_keys = {"char_vectorizer", "word_vectorizer", "svd"}
    if not required_vectorizer_keys.issubset(vectorizers):
        raise ValueError(f"Vectorizer artifact must contain {sorted(required_vectorizer_keys)}")
    return model, vectorizers, thresholds


def _fold_predictions(
    engineered: pd.DataFrame,
    vectorizers: Dict[str, Any],
    thresholds: Dict[str, float],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Train fold models and collect out-of-fold probabilities and predictions."""
    text_features = _transform_text_features(
        _prepare_text(engineered),
        vectorizers["char_vectorizer"],
        vectorizers["word_vectorizer"],
        vectorizers["svd"],
    )
    model_frame, categorical = _model_frame(engineered, text_features)
    labels = engineered[TARGET_COLUMN].astype(str).to_numpy()
    classes = np.array(sorted(np.unique(labels)))
    class_weights = _class_weights(pd.Series(labels))
    splitter = TimeSeriesSplit(n_splits=5)
    validation_rows = []
    fold_summaries = []

    for fold, (train_indices, valid_indices) in enumerate(splitter.split(model_frame), start=1):
        model = _build_model(class_weights)
        categorical_indices = [model_frame.columns.get_loc(column) for column in categorical]
        model.fit(
            model_frame.iloc[train_indices],
            labels[train_indices],
            cat_features=categorical_indices,
            eval_set=(model_frame.iloc[valid_indices], labels[valid_indices]),
            use_best_model=False,
        )
        probabilities = model.predict_proba(model_frame.iloc[valid_indices])
        fold_classes = np.asarray(model.classes_).astype(str)
        ordered_probabilities = np.zeros((len(valid_indices), len(classes)))
        for model_index, label in enumerate(fold_classes):
            ordered_probabilities[:, np.where(classes == label)[0][0]] = probabilities[:, model_index]
        predictions = _apply_thresholds(ordered_probabilities, classes, thresholds)
        fold_labels = labels[valid_indices]
        fold_score = f1_score(fold_labels, predictions, average="macro", zero_division=0)
        logger.info("Fold %d error-analysis macro F1: %.4f", fold, fold_score)
        fold_summaries.append({"fold": fold, "macro_f1": float(fold_score), "rows": len(valid_indices)})
        fold_data = engineered.iloc[valid_indices].copy()
        fold_data["fold"] = fold
        fold_data["actual"] = fold_labels
        fold_data["predicted"] = predictions
        fold_data["max_probability"] = ordered_probabilities.max(axis=1)
        validation_rows.append(fold_data)

    validation_data = pd.concat(validation_rows, ignore_index=True)
    actual = validation_data["actual"].to_numpy()
    predicted = validation_data["predicted"].to_numpy()
    return validation_data, actual, predicted, classes, fold_summaries


def _per_class_analysis(validation_data: pd.DataFrame, classes: np.ndarray) -> Dict[str, Any]:
    """Calculate per-class metrics and representative error descriptions."""
    actual = validation_data["actual"].to_numpy()
    predicted = validation_data["predicted"].to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=classes, zero_division=0
    )
    results = {}
    for index, label in enumerate(classes):
        false_negatives = (actual == label) & (predicted != label)
        false_positives = (actual != label) & (predicted == label)
        errors = validation_data[false_negatives | false_positives]
        examples = []
        for _, row in errors.head(10).iterrows():
            examples.append(
                {
                    "description": str(row["description"]),
                    "actual": str(row["actual"]),
                    "predicted": str(row["predicted"]),
                    "amount": float(row["amount"]),
                    "fold": int(row["fold"]),
                }
            )
        results[str(label)] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "false_positives": int(false_positives.sum()),
            "false_negatives": int(false_negatives.sum()),
            "error_rate": float((false_negatives.sum() + false_positives.sum()) / max(1, (actual == label).sum())),
            "misclassified_examples": examples,
        }
    return results


def _pattern_analysis(validation_data: pd.DataFrame) -> Dict[str, Any]:
    """Measure error rates for ambiguous terms, amounts, calendar periods and stores."""
    data = validation_data.copy()
    data["is_error"] = data["actual"] != data["predicted"]
    descriptions = data["description"].fillna("").astype(str)
    patterns = {
        "market": r"\bmarket\b",
        "payment": r"\bpayment\b",
        "booking": r"\bbooking\b",
        "target": r"\btarget\b",
        "amazon": r"\bamazon\b|\bamzn\b",
        "cvs": r"\bcvs\b",
    }
    keyword_results = {}
    for name, pattern in patterns.items():
        mask = descriptions.str.contains(pattern, case=False, regex=True)
        keyword_results[name] = {
            "count": int(mask.sum()),
            "errors": int(data.loc[mask, "is_error"].sum()),
            "error_rate": float(data.loc[mask, "is_error"].mean()) if mask.any() else 0.0,
            "actual_distribution": data.loc[mask, "actual"].value_counts().to_dict(),
        }

    amounts = pd.to_numeric(data["amount"], errors="coerce")
    amount_masks = {
        "round_amount_over_500": amounts.mod(1).eq(0) & amounts.gt(500),
        "decimal_amount": amounts.mod(1).ne(0),
    }
    amount_results = {
        name: {
            "count": int(mask.sum()),
            "errors": int(data.loc[mask, "is_error"].sum()),
            "error_rate": float(data.loc[mask, "is_error"].mean()) if mask.any() else 0.0,
        }
        for name, mask in amount_masks.items()
    }

    dates = pd.to_datetime(data["date"], errors="coerce")
    time_masks = {
        "month_start": dates.dt.day.le(10),
        "month_end": dates.dt.day.ge(21),
        "weekend": dates.dt.dayofweek.ge(5),
    }
    time_results = {
        name: {
            "count": int(mask.sum()),
            "errors": int(data.loc[mask, "is_error"].sum()),
            "error_rate": float(data.loc[mask, "is_error"].mean()) if mask.any() else 0.0,
        }
        for name, mask in time_masks.items()
    }

    store_results = {}
    for store, group in data.groupby("store_name"):
        if len(group) >= 5:
            store_results[str(store)] = {
                "count": int(len(group)),
                "errors": int(group["is_error"].sum()),
                "error_rate": float(group["is_error"].mean()),
            }
    store_results = dict(sorted(store_results.items(), key=lambda item: item[1]["error_rate"], reverse=True)[:30])
    return {
        "ambiguous_keywords": keyword_results,
        "amount_patterns": amount_results,
        "time_patterns": time_results,
        "stores_highest_error_rate": store_results,
    }


def _suggest_rules(patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Suggest conservative post-processing rules for strong transaction signals."""
    suggestions = [
        {"rule": "IRS or TAX in description", "suggested_category": "Bills & Utilities", "reason": "Tax-related payments are usually utilities or bills."},
        {"rule": "UBER or LYFT in description", "suggested_category": "Transportation", "reason": "Ride-share merchants are transport transactions."},
    ]
    keyword_categories = {
        "amazon": "Shopping",
        "cvs": "Health & Fitness",
        "booking": "Travel",
    }
    for keyword, category in keyword_categories.items():
        result = patterns["ambiguous_keywords"][keyword]
        if result["count"] and result["error_rate"] > 0.10:
            suggestions.append(
                {
                    "rule": f"{keyword.upper()} in description",
                    "suggested_category": category,
                    "reason": f"Observed error rate is {result['error_rate']:.2%}; merchant keyword is actionable.",
                }
            )
    return suggestions


def _save_visualizations(
    validation_data: pd.DataFrame,
    classes: np.ndarray,
    figures_dir: Path,
    raw_matrix: np.ndarray,
) -> None:
    """Save normalized confusion matrix and per-class F1 visualizations."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    row_totals = raw_matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        raw_matrix,
        row_totals,
        out=np.zeros_like(raw_matrix, dtype=float),
        where=row_totals != 0,
    )
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axis,
    )
    axis.set(title="Normalized Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(figures_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    metrics = _per_class_analysis(validation_data, classes)
    f1_values = [metrics[str(label)]["f1"] for label in classes]
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(classes), y=f1_values, ax=axis, color="#2a9d8f")
    axis.set(title="Validation F1 by Class", xlabel="Class", ylabel="F1")
    axis.tick_params(axis="x", rotation=35)
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(figures_dir / "per_class_f1.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_error_analysis() -> Dict[str, Any]:
    """Run fold validation error analysis and save the complete report."""
    model, vectorizers, thresholds = _load_artifacts()
    loader = DataLoader(use_cache=False)
    raw_train = loader.load_train_data(force_reload=True)
    raw_train = raw_train.sort_values("transaction_id").reset_index(drop=True)
    engineered = engineer_features(raw_train, is_train=True)

    validation_data, actual, predicted, classes, fold_summaries = _fold_predictions(
        engineered, vectorizers, thresholds
    )
    raw_matrix = confusion_matrix(actual, predicted, labels=classes)
    row_totals = raw_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        raw_matrix,
        row_totals,
        out=np.zeros_like(raw_matrix, dtype=float),
        where=row_totals != 0,
    )
    per_class = _per_class_analysis(validation_data, classes)
    patterns = _pattern_analysis(validation_data)
    report = {
        "model": {
            "path": str(MODEL_PATH),
            "loaded_model_type": type(model).__name__,
            "thresholds": thresholds,
        },
        "folds": fold_summaries,
        "focus_fold": 3,
        "confusion_matrix": {
            "labels": classes.tolist(),
            "raw": raw_matrix.tolist(),
            "normalized_by_true_label": normalized_matrix.tolist(),
        },
        "per_class_metrics": per_class,
        "error_patterns": patterns,
        "suggested_rules": _suggest_rules(patterns),
    }

    figures_dir = Path(EXPERIMENTS_DIR) / "figures"
    _save_visualizations(validation_data, classes, figures_dir, raw_matrix)
    report_path = Path(EXPERIMENTS_DIR) / "error_analysis.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False, default=_json_default)

    overall_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
    worst_classes = sorted(per_class.items(), key=lambda item: item[1]["f1"])[:3]
    print("=" * 70)
    print("AuroraGate Error Analysis")
    print("=" * 70)
    print(f"Validation rows: {len(validation_data)}")
    print(f"Overall out-of-fold macro F1: {overall_f1:.4f}")
    print("Fold scores:")
    for fold in fold_summaries:
        print(f"  Fold {fold['fold']}: {fold['macro_f1']:.4f}")
    print("Lowest-F1 classes:")
    for label, metrics in worst_classes:
        print(f"  {label}: F1={metrics['f1']:.4f}, FN={metrics['false_negatives']}, FP={metrics['false_positives']}")
    print(f"Report: {report_path}")
    print("=" * 70)
    return report


if __name__ == "__main__":
    run_error_analysis()