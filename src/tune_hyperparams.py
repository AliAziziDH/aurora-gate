"""Optuna hyperparameter tuning for the AuroraGate model ensemble."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import EXPERIMENTS_DIR, MODELS_DIR, RANDOM_STATE, TARGET_COLUMN
from src.data_loader import DataLoader
from src.feature_engineering import engineer_features
from src.train_lightgbm import (
    _apply_thresholds,
    _load_text_artifacts,
    _numeric_frame,
    _ordered_probabilities,
)
from src.train_model import (
    _class_weights,
    _model_frame,
    _prepare_text,
    _transform_text_features,
)
from src.logger import get_logger


logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 25
N_SPLITS = 5
CATBOOST_PARAMS_PATH = Path(MODELS_DIR) / "catboost_best_params.json"
LIGHTGBM_PARAMS_PATH = Path(MODELS_DIR) / "lightgbm_best_params.json"
SUMMARY_PATH = Path(EXPERIMENTS_DIR) / "optuna_summary.json"
THRESHOLDS_PATH = Path(MODELS_DIR) / "thresholds.json"


def _load_thresholds() -> Dict[str, float]:
    """Load the class-specific thresholds produced by CatBoost training."""
    if not THRESHOLDS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {THRESHOLDS_PATH}; run `python -m src.train_model` first."
        )
    with THRESHOLDS_PATH.open("r", encoding="utf-8") as threshold_file:
        return {str(key): float(value) for key, value in json.load(threshold_file).items()}


def _suggest_parameters(trial: optuna.Trial) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Sample CatBoost and LightGBM parameters from the requested search spaces."""
    catboost_params = {
        "iterations": trial.suggest_int("cat_iterations", 300, 1000),
        "learning_rate": trial.suggest_float("cat_learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("cat_depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("cat_l2_leaf_reg", 1.0, 10.0, log=True),
        "subsample": trial.suggest_float("cat_subsample", 0.6, 1.0),
    }
    lightgbm_params = {
        "n_estimators": trial.suggest_int("lgb_n_estimators", 300, 1000),
        "learning_rate": trial.suggest_float("lgb_learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("lgb_num_leaves", 15, 50),
        "subsample": trial.suggest_float("lgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("lgb_colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("lgb_min_child_samples", 5, 30),
    }
    return catboost_params, lightgbm_params


def _build_catboost(parameters: Dict[str, Any], class_weights: Dict[str, float]) -> CatBoostClassifier:
    """Build a fold-specific CatBoost model with early stopping enabled."""
    return CatBoostClassifier(
        **parameters,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        class_weights=class_weights,
        bootstrap_type="Bernoulli",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=50,
        thread_count=4,
    )


def _build_lightgbm(parameters: Dict[str, Any]) -> lgb.LGBMClassifier:
    """Build a fold-specific balanced LightGBM model with early stopping."""
    return lgb.LGBMClassifier(
        **parameters,
        objective="multiclass",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=1,
    )


def _load_training_arrays() -> Dict[str, Any]:
    """Load, engineer and transform the training data once for all trials."""
    loader = DataLoader(use_cache=False)
    raw_train = loader.load_train_data(force_reload=True)
    raw_train = raw_train.sort_values("transaction_id").reset_index(drop=True)
    engineered = engineer_features(raw_train, is_train=True)
    text = _prepare_text(engineered)
    text_bundle = _load_text_artifacts(text)
    text_features = _transform_text_features(
        text,
        text_bundle["char_vectorizer"],
        text_bundle["word_vectorizer"],
        text_bundle["svd"],
    )
    cat_frame, categorical = _model_frame(engineered, text_features)
    lgb_frame = _numeric_frame(cat_frame, categorical)
    labels = engineered[TARGET_COLUMN].astype(str).to_numpy()
    classes = np.array(sorted(np.unique(labels)))
    encoded_labels = np.array([np.where(classes == label)[0][0] for label in labels])
    return {
        "cat_frame": cat_frame,
        "lgb_frame": lgb_frame,
        "categorical": categorical,
        "labels": labels,
        "encoded_labels": encoded_labels,
        "classes": classes,
        "class_weights": _class_weights(pd.Series(labels)),
        "thresholds": _load_thresholds(),
        "splitter": TimeSeriesSplit(n_splits=N_SPLITS),
    }


def _objective(trial: optuna.Trial, arrays: Dict[str, Any]) -> float:
    """Train both models over all folds and return mean ensemble macro F1."""
    catboost_params, lightgbm_params = _suggest_parameters(trial)
    cat_frame = arrays["cat_frame"]
    lgb_frame = arrays["lgb_frame"]
    labels = arrays["labels"]
    encoded_labels = arrays["encoded_labels"]
    classes = arrays["classes"]
    categorical = arrays["categorical"]
    thresholds = arrays["thresholds"]
    fold_scores: List[float] = []
    categorical_indices = [cat_frame.columns.get_loc(column) for column in categorical]

    for fold, (train_indices, valid_indices) in enumerate(
        arrays["splitter"].split(lgb_frame), start=1
    ):
        catboost_model = _build_catboost(catboost_params, arrays["class_weights"])
        catboost_model.fit(
            cat_frame.iloc[train_indices],
            labels[train_indices],
            cat_features=categorical_indices,
            eval_set=(cat_frame.iloc[valid_indices], labels[valid_indices]),
            use_best_model=True,
        )
        cat_probabilities = _ordered_probabilities(
            catboost_model.predict_proba(cat_frame.iloc[valid_indices]),
            np.asarray(catboost_model.classes_),
            classes,
        )

        lightgbm_model = _build_lightgbm(lightgbm_params)
        lightgbm_model.fit(
            lgb_frame.iloc[train_indices],
            encoded_labels[train_indices],
            eval_X=[lgb_frame.iloc[valid_indices].to_numpy()],
            eval_y=encoded_labels[valid_indices],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        lgb_probabilities = _ordered_probabilities(
            lightgbm_model.predict_proba(lgb_frame.iloc[valid_indices]),
            np.asarray(lightgbm_model.classes_),
            classes,
        )
        ensemble_probabilities = (cat_probabilities + lgb_probabilities) / 2.0
        predictions = _apply_thresholds(ensemble_probabilities, classes, thresholds)
        score = f1_score(labels[valid_indices], predictions, average="macro", zero_division=0)
        fold_scores.append(float(score))
        mean_score = float(np.mean(fold_scores))
        trial.report(mean_score, step=fold)
        logger.info("Trial %d fold %d ensemble macro F1: %.4f", trial.number, fold, score)
        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned after fold {fold} at score {mean_score:.4f}")

    return float(np.mean(fold_scores))


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Create the parent directory and save a JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, default=str)


def tune_hyperparameters() -> Dict[str, Any]:
    """Run the Optuna study and save the best parameters and summary."""
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(EXPERIMENTS_DIR).mkdir(parents=True, exist_ok=True)
    arrays = _load_training_arrays()
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    logger.info("Starting Optuna study with %d trials", N_TRIALS)
    study.optimize(lambda trial: _objective(trial, arrays), n_trials=N_TRIALS, gc_after_trial=True)

    best_trial = study.best_trial
    best_catboost, best_lightgbm = _suggest_parameters_from_values(best_trial.params)
    _save_json(CATBOOST_PARAMS_PATH, best_catboost)
    _save_json(LIGHTGBM_PARAMS_PATH, best_lightgbm)
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED]
    summary = {
        "n_trials_requested": N_TRIALS,
        "n_trials_completed": len(completed_trials),
        "n_trials_pruned": len(pruned_trials),
        "best_value_macro_f1": float(best_trial.value),
        "best_trial_number": best_trial.number,
        "best_catboost_params": best_catboost,
        "best_lightgbm_params": best_lightgbm,
        "baseline_catboost_macro_f1": 0.9111,
        "baseline_lightgbm_macro_f1": 0.9012,
        "baseline_equal_weight_ensemble_macro_f1": 0.9108,
        "improvement_over_equal_weight_baseline": float(best_trial.value - 0.9108),
        "study_trials": [
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
            }
            for trial in study.trials
        ],
    }
    _save_json(SUMMARY_PATH, summary)
    print("=" * 70)
    print("AuroraGate Optuna Hyperparameter Tuning")
    print("=" * 70)
    print(f"Best ensemble macro F1: {best_trial.value:.4f}")
    print(f"Completed trials: {len(completed_trials)} | Pruned trials: {len(pruned_trials)}")
    print("Best CatBoost parameters:")
    print(json.dumps(best_catboost, indent=2))
    print("Best LightGBM parameters:")
    print(json.dumps(best_lightgbm, indent=2))
    print(f"Improvement over equal-weight baseline: {best_trial.value - 0.9108:+.4f}")
    print(f"Summary: {SUMMARY_PATH}")
    print("=" * 70)
    return summary


def _suggest_parameters_from_values(values: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract model parameter dictionaries from Optuna's flattened parameter names."""
    catboost = {
        "iterations": values["cat_iterations"],
        "learning_rate": values["cat_learning_rate"],
        "depth": values["cat_depth"],
        "l2_leaf_reg": values["cat_l2_leaf_reg"],
        "subsample": values["cat_subsample"],
    }
    lightgbm = {
        "n_estimators": values["lgb_n_estimators"],
        "learning_rate": values["lgb_learning_rate"],
        "num_leaves": values["lgb_num_leaves"],
        "subsample": values["lgb_subsample"],
        "colsample_bytree": values["lgb_colsample_bytree"],
        "min_child_samples": values["lgb_min_child_samples"],
    }
    return catboost, lightgbm


if __name__ == "__main__":
    tune_hyperparameters()