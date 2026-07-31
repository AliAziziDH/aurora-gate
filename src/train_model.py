"""Train and evaluate an AuroraGate CatBoost transaction classifier."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    CATBOOST_PARAMS,
    EXPERIMENTS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.data_loader import DataLoader, logger as data_loader_logger
from src.feature_engineering import categorical_feature_names, engineer_features


logger = logging.getLogger(__name__)
logger.setLevel(data_loader_logger.level)

MODEL_PATH = Path(MODELS_DIR) / "catboost_model.pkl"
THRESHOLDS_PATH = Path(MODELS_DIR) / "thresholds.json"
VECTORIZER_PATH = Path(MODELS_DIR) / "tfidf_vectorizer.pkl"


def _prepare_text(df: pd.DataFrame) -> pd.Series:
    """Build a stable text field for both TF-IDF vectorizers."""
    return (
        df["description"].fillna("").astype(str)
        + " "
        + df["store_name"].fillna("UNKNOWN").astype(str)
        + " "
        + df["day_of_week"].fillna("UNKNOWN").astype(str)
    )


def _fit_text_features(text: pd.Series) -> Tuple[object, object, TruncatedSVD, np.ndarray]:
    """Fit both requested TF-IDF vectorizers and reduce their combination."""
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=50000, sublinear_tf=True
    )
    word_vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=20000, sublinear_tf=True,
        min_df=1,
    )
    char_matrix = char_vectorizer.fit_transform(text)
    word_matrix = word_vectorizer.fit_transform(text)
    combined = hstack([char_matrix, word_matrix], format="csr")
    components = min(64, max(2, combined.shape[1] - 1), max(2, combined.shape[0] - 1))
    svd = TruncatedSVD(n_components=components, random_state=RANDOM_STATE)
    reduced = svd.fit_transform(combined)
    return char_vectorizer, word_vectorizer, svd, reduced


def _transform_text_features(
    text: pd.Series, char_vectorizer: object, word_vectorizer: object, svd: TruncatedSVD
) -> np.ndarray:
    """Transform text with fitted vectorizers and SVD."""
    char_matrix = char_vectorizer.transform(text)
    word_matrix = word_vectorizer.transform(text)
    return svd.transform(hstack([char_matrix, word_matrix], format="csr"))


def _model_frame(df: pd.DataFrame, text_features: np.ndarray) -> Tuple[pd.DataFrame, List[str]]:
    """Create CatBoost's numeric and categorical feature frame."""
    categorical = [column for column in categorical_feature_names() if column in df.columns]
    excluded = {"date", "description", TARGET_COLUMN}
    numeric_columns = [
        column for column in df.columns
        if column not in excluded and column not in categorical
    ]
    frame = df[numeric_columns].copy()
    for column in categorical:
        frame[column] = df[column].fillna("unknown").astype(str)
    text_frame = pd.DataFrame(
        text_features,
        index=df.index,
        columns=[f"tfidf_svd_{index}" for index in range(text_features.shape[1])],
    )
    frame = pd.concat([frame.reset_index(drop=True), text_frame.reset_index(drop=True)], axis=1)
    return frame, categorical


def _class_weights(target: pd.Series) -> Dict[str, float]:
    """Calculate balanced class weights from the observed target classes."""
    classes = np.sort(target.unique())
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=target)
    return {str(label): float(weight) for label, weight in zip(classes, weights)}


def _build_model(class_weights: Dict[str, float]) -> CatBoostClassifier:
    """Create a CatBoost classifier using project defaults and class weights."""
    params = dict(CATBOOST_PARAMS)
    params.update(
        {
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1:average=Macro",
            "class_weights": class_weights,
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "allow_writing_files": False,
        }
    )
    return CatBoostClassifier(**params)


def _fit_fold_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    categorical: List[str],
    class_weights: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit one time-series fold and return validation predictions and labels."""
    model = _build_model(class_weights)
    categorical_indices = [train_x.columns.get_loc(column) for column in categorical]
    model.fit(
        train_x,
        train_y,
        cat_features=categorical_indices,
        eval_set=(valid_x, valid_y),
        use_best_model=False,
    )
    return model.predict_proba(valid_x), valid_y.to_numpy()


def _tune_thresholds(
    probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> Dict[str, float]:
    """Tune one-vs-rest class thresholds using a macro-F1 objective."""
    thresholds = {str(label): 1.0 for label in classes}
    for class_index, label in enumerate(classes):
        best_threshold = 1.0
        best_score = f1_score(labels, classes[np.argmax(probabilities, axis=1)], average="macro")
        for threshold in np.linspace(0.10, 1.00, 19):
            adjusted = probabilities.copy()
            adjusted[:, class_index] = adjusted[:, class_index] / threshold
            predictions = classes[np.argmax(adjusted, axis=1)]
            score = f1_score(labels, predictions, average="macro", zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds[str(label)] = best_threshold
    return thresholds


def _apply_thresholds(probabilities: np.ndarray, classes: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """Convert probabilities into predictions using class thresholds."""
    adjusted = probabilities.copy()
    for index, label in enumerate(classes):
        adjusted[:, index] /= thresholds.get(str(label), 1.0)
    return classes[np.argmax(adjusted, axis=1)]


def train_model() -> Dict[str, object]:
    """Train the final CatBoost model and save all inference artifacts."""
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(EXPERIMENTS_DIR).mkdir(parents=True, exist_ok=True)

    loader = DataLoader(use_cache=False)
    raw_train = loader.load_train_data(force_reload=True)
    raw_train = raw_train.sort_values("transaction_id").reset_index(drop=True)
    engineered = engineer_features(raw_train, is_train=True)
    text = _prepare_text(engineered)
    char_vectorizer, word_vectorizer, svd, text_features = _fit_text_features(text)
    model_frame, categorical = _model_frame(engineered, text_features)
    target = engineered[TARGET_COLUMN].astype(str)
    classes = np.array(sorted(target.unique()))
    weights = _class_weights(target)

    splitter = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    all_probabilities = []
    all_labels = []
    for fold, (train_indices, valid_indices) in enumerate(splitter.split(model_frame), start=1):
        probabilities, labels = _fit_fold_model(
            model_frame.iloc[train_indices],
            target.iloc[train_indices],
            model_frame.iloc[valid_indices],
            target.iloc[valid_indices],
            categorical,
            weights,
        )
        predictions = classes[np.argmax(probabilities, axis=1)]
        macro_score = f1_score(labels, predictions, average="macro", zero_division=0)
        per_class = f1_score(labels, predictions, labels=classes, average=None, zero_division=0)
        fold_scores.append({"fold": fold, "macro_f1": float(macro_score), "per_class_f1": dict(zip(classes, per_class))})
        all_probabilities.append(probabilities)
        all_labels.append(labels)
        logger.info("Fold %d macro F1: %.4f", fold, macro_score)
        logger.info("Fold %d per-class F1: %s", fold, dict(zip(classes, per_class.round(4))))

    validation_probabilities = np.vstack(all_probabilities)
    validation_labels = np.concatenate(all_labels)
    thresholds = _tune_thresholds(validation_probabilities, validation_labels, classes)
    tuned_predictions = _apply_thresholds(validation_probabilities, classes, thresholds)
    tuned_macro = f1_score(validation_labels, tuned_predictions, average="macro", zero_division=0)

    final_model = _build_model(weights)
    categorical_indices = [model_frame.columns.get_loc(column) for column in categorical]
    final_model.fit(model_frame, target, cat_features=categorical_indices)

    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(
        {
            "char_vectorizer": char_vectorizer,
            "word_vectorizer": word_vectorizer,
            "svd": svd,
            "categorical_columns": categorical,
            "feature_columns": model_frame.columns.tolist(),
        },
        VECTORIZER_PATH,
    )
    with THRESHOLDS_PATH.open("w", encoding="utf-8") as threshold_file:
        json.dump(thresholds, threshold_file, indent=2)

    result = {
        "fold_scores": fold_scores,
        "tuned_validation_macro_f1": float(tuned_macro),
        "classes": classes.tolist(),
        "thresholds": thresholds,
        "model_path": str(MODEL_PATH),
        "vectorizer_path": str(VECTORIZER_PATH),
    }
    logger.info("Tuned validation macro F1: %.4f", tuned_macro)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    train_model()