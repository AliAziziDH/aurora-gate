"""
Utility functions for model training and cross-validation.

This module provides reusable functions for common training tasks including:
- Cross-validation loops
- Early stopping callbacks
- Model evaluation
- Out-of-fold prediction generation
"""

from typing import Any, Dict, List, Tuple, Callable, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, BaseCrossValidator
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
from pathlib import Path

from src.config import RANDOM_STATE, N_SPLITS, EXPERIMENTS_DIR, MODELS_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def run_cv_training(
    model_builder: Callable,
    train_data: pd.DataFrame,
    target: pd.Series,
    cv_strategy: BaseCrossValidator = None,
    fit_params: Dict[str, Any] = None,
    eval_metric: str = "macro_f1",
    early_stopping_rounds: int = 50,
    use_best_model: bool = True,
) -> Tuple[List[float], np.ndarray, np.ndarray]:
    """
    Run cross-validated training and return fold scores, OOF predictions, and probabilities.
    
    Args:
        model_builder: Function that returns an initialized model instance
        train_data: Training features DataFrame
        target: Target variable Series
        cv_strategy: Cross-validation strategy (default: TimeSeriesSplit with N_SPLITS)
        fit_params: Additional parameters to pass to model.fit()
        eval_metric: Evaluation metric to use (default: 'macro_f1')
        early_stopping_rounds: Number of rounds for early stopping
        use_best_model: Whether to use the best model from early stopping
        
    Returns:
        Tuple containing:
        - List of fold scores
        - Array of out-of-fold predictions
        - Array of out-of-fold probabilities
    """
    if cv_strategy is None:
        cv_strategy = TimeSeriesSplit(n_splits=N_SPLITS)
    
    if fit_params is None:
        fit_params = {}
    
    fold_scores = []
    oof_predictions = []
    oof_probabilities = []
    
    classes = np.array(sorted(target.unique()))
    
    for fold, (train_indices, valid_indices) in enumerate(cv_strategy.split(train_data), start=1):
        logger.info("Starting fold %d", fold)
        
        # Split data
        X_train, X_valid = train_data.iloc[train_indices], train_data.iloc[valid_indices]
        y_train, y_valid = target.iloc[train_indices], target.iloc[valid_indices]
        
        # Build and train model
        model = model_builder()
        
        # Add early stopping if eval_set is provided
        if 'eval_set' not in fit_params:
            fit_params['eval_set'] = [(X_valid, y_valid)]
        
        # Train model
        model.fit(X_train, y_train, **fit_params)
        
        # Generate predictions
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_valid)
            preds = classes[np.argmax(probas, axis=1)]
            oof_probabilities.append(probas)
        else:
            preds = model.predict(X_valid)
            oof_probabilities.append(np.zeros((len(y_valid), len(classes))))
        
        # Calculate and store score
        if eval_metric == "macro_f1":
            score = f1_score(y_valid, preds, average="macro", zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {eval_metric}")
        
        fold_scores.append(float(score))
        oof_predictions.append(preds)
        
        logger.info("Fold %d %s: %.4f", fold, eval_metric, score)
    
    return fold_scores, np.concatenate(oof_predictions), np.vstack(oof_probabilities)


def compute_class_weights(target: pd.Series) -> Dict[str, float]:
    """
    Compute balanced class weights for imbalanced datasets.
    
    Args:
        target: Target variable Series
        
    Returns:
        Dictionary mapping class labels to weights
    """
    classes = np.sort(target.unique())
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=target)
    return {str(label): float(weight) for label, weight in zip(classes, weights)}


def save_model_artifacts(
    model: Any,
    artifacts: Dict[str, Any],
    model_name: str,
    save_dir: Path = MODELS_DIR
) -> Dict[str, Path]:
    """
    Save model and related artifacts to disk.
    
    Args:
        model: Trained model to save
        artifacts: Dictionary of additional artifacts to save
        model_name: Base name for the saved files
        save_dir: Directory to save artifacts
        
    Returns:
        Dictionary mapping artifact names to their saved paths
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    # Save main model
    model_path = save_dir / f"{model_name}_model.pkl"
    joblib.dump(model, model_path)
    saved_paths["model"] = model_path
    
    # Save additional artifacts
    for artifact_name, artifact in artifacts.items():
        if artifact_name == "model":
            continue
            
        if artifact_name.endswith("_path"):
            # This is already a path, just store it
            saved_paths[artifact_name] = artifact
        else:
            # Save the artifact
            artifact_path = save_dir / f"{model_name}_{artifact_name}.pkl"
            joblib.dump(artifact, artifact_path)
            saved_paths[artifact_name] = artifact_path
    
    logger.info("Saved model artifacts to: %s", save_dir)
    return saved_paths


def load_model_artifacts(
    model_name: str,
    load_dir: Path = MODELS_DIR
) -> Dict[str, Any]:
    """
    Load model and related artifacts from disk.
    
    Args:
        model_name: Base name of the saved files
        load_dir: Directory containing saved artifacts
        
    Returns:
        Dictionary containing loaded artifacts
    """
    artifacts = {}
    
    # Load main model
    model_path = load_dir / f"{model_name}_model.pkl"
    if model_path.exists():
        artifacts["model"] = joblib.load(model_path)
    
    # Load additional artifacts
    for artifact_file in load_dir.glob(f"{model_name}_*.pkl"):
        if artifact_file == model_path:
            continue
        
        artifact_name = artifact_file.stem.replace(f"{model_name}_", "")
        artifacts[artifact_name] = joblib.load(artifact_file)
    
    logger.info("Loaded model artifacts from: %s", load_dir)
    return artifacts


def save_training_summary(
    summary: Dict[str, Any],
    summary_name: str,
    save_dir: Path = EXPERIMENTS_DIR
) -> Path:
    """
    Save training summary as JSON file.
    
    Args:
        summary: Dictionary containing training summary
        summary_name: Base name for the summary file
        save_dir: Directory to save summary
        
    Returns:
        Path to the saved summary file
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = save_dir / f"{summary_name}.json"
    
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info("Saved training summary to: %s", summary_path)
    return summary_path
