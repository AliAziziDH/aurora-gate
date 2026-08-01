"""Train and evaluate an AuroraGate CatBoost transaction classifier."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    CATBOOST_PARAMS,
    EXPERIMENTS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TFIDF_PARAMS,
)
from src.data_loader import DataLoader
from src.feature_engineering import categorical_feature_names, engineer_features
from src.logger import get_logger
from src.training_utils import (
    run_cv_training,
    compute_class_weights,
    save_model_artifacts,
    save_training_summary,
)


logger = get_logger(__name__)

MODEL_NAME = "catboost"
MODEL_PATH = Path(MODELS_DIR) / f"{MODEL_NAME}_model.pkl"
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
    from src.config import TFIDF_PARAMS
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=TFIDF_PARAMS["max_features"], sublinear_tf=True
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
    excluded = {"date", "description", "transaction_id", TARGET_COLUMN}
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


# Using compute_class_weights from training_utils


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
            "od_type": "Iter",
            "od_wait": 50,  # Early stopping after 50 iterations without improvement
        }
    )
    return CatBoostClassifier(**params)


def _build_model_builder(class_weights: Dict[str, float]):
    """Create a model builder function for use with run_cv_training."""
    def model_builder():
        """Build and return a CatBoost classifier."""
        return _build_model(class_weights)
    return model_builder


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


def _train_baseline_models(
    model_frame: pd.DataFrame,
    target: pd.Series,
    classes: np.ndarray
) -> Dict[str, float]:
    """Train and evaluate baseline models for comparison."""
    baseline_results = {}
    
    # Dummy classifier (most frequent class)
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy_scores, _, _ = run_cv_training(
        model_builder=lambda: dummy,
        train_data=model_frame,
        target=target,
        eval_metric="macro_f1"
    )
    baseline_results["dummy_most_frequent"] = float(np.mean(dummy_scores))
    logger.info("Dummy (most frequent) macro F1: %.4f", baseline_results["dummy_most_frequent"])
    
    # Logistic Regression baseline
    try:
        lr = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        lr_scores, _, _ = run_cv_training(
            model_builder=lambda: lr,
            train_data=model_frame,
            target=target,
            eval_metric="macro_f1"
        )
        baseline_results["logistic_regression"] = float(np.mean(lr_scores))
        logger.info("Logistic Regression macro F1: %.4f", baseline_results["logistic_regression"])
    except Exception as e:
        logger.warning("Logistic Regression baseline failed: %s", e)
        baseline_results["logistic_regression"] = 0.0
    
    return baseline_results


def train_model() -> Dict[str, object]:
    """Train the final CatBoost model and save all inference artifacts."""
    # Load and prepare data
    loader = DataLoader(use_cache=False)
    raw_train = loader.load_train_data(force_reload=True)
    raw_train = raw_train.sort_values("transaction_id").reset_index(drop=True)
    engineered = engineer_features(raw_train, is_train=True)
    text = _prepare_text(engineered)
    char_vectorizer, word_vectorizer, svd, text_features = _fit_text_features(text)
    model_frame, categorical = _model_frame(engineered, text_features)
    target = engineered[TARGET_COLUMN].astype(str)
    classes = np.array(sorted(target.unique()))
    weights = compute_class_weights(target)

    # Run cross-validated training
    model_builder = _build_model_builder(weights)
    
    # Prepare fit parameters for CatBoost
    categorical_indices = [model_frame.columns.get_loc(column) for column in categorical]
    fit_params = {
        "cat_features": categorical_indices,
        "eval_set": None,  # Will be set by run_cv_training
        "use_best_model": False,
        "early_stopping_rounds": 50,
    }
    
    fold_scores, oof_predictions, oof_probabilities = run_cv_training(
        model_builder=model_builder,
        train_data=model_frame,
        target=target,
        fit_params=fit_params,
        eval_metric="macro_f1",
    )

    # Tune thresholds
    thresholds = _tune_thresholds(oof_probabilities, target[oof_predictions.index].to_numpy(), classes)
    tuned_predictions = _apply_thresholds(oof_probabilities, classes, thresholds)
    tuned_macro = f1_score(target[oof_predictions.index].to_numpy(), tuned_predictions, average="macro", zero_division=0)

    # Train and evaluate baseline models
    baseline_results = _train_baseline_models(model_frame, target, classes)
    
    # Train final model on full data
    final_model = _build_model(weights)
    try:
        final_model.fit(model_frame, target, cat_features=categorical_indices)
    except Exception as error:
        logger.error("Training failed: %s", error)
        logger.info("Falling back to default parameters...")
        final_model = _build_model(weights)
        final_model.fit(model_frame, target, cat_features=categorical_indices)

    # Save artifacts
    saved_paths = save_model_artifacts(
        model=final_model,
        artifacts={
            "char_vectorizer": char_vectorizer,
            "word_vectorizer": word_vectorizer,
            "svd": svd,
            "categorical_columns": categorical,
            "feature_columns": model_frame.columns.tolist(),
            "vectorizer_path": VECTORIZER_PATH,
        },
        model_name=MODEL_NAME,
    )
    
    with THRESHOLDS_PATH.open("w", encoding="utf-8") as threshold_file:
        json.dump(thresholds, threshold_file, indent=2)

    # Prepare result
    result = {
        "fold_scores": [{"fold": i+1, "macro_f1": score} for i, score in enumerate(fold_scores)],
        "tuned_validation_macro_f1": float(tuned_macro),
        "baseline_models": baseline_results,
        "improvement_over_baseline": float(tuned_macro - baseline_results.get("logistic_regression", 0.0)),
        "classes": classes.tolist(),
        "thresholds": thresholds,
        "model_path": str(saved_paths["model"]),
        "vectorizer_path": str(VECTORIZER_PATH),
    }
    
    # Save training summary
    save_training_summary(result, f"{MODEL_NAME}_training_summary")
    
    logger.info("Tuned validation macro F1: %.4f", tuned_macro)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    train_model()