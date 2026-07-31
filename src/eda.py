"""Exploratory data analysis for the AuroraGate expense categorization data."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

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
from sklearn.feature_extraction.text import CountVectorizer

from src.data_loader import DataLoader, logger as data_loader_logger


logger = logging.getLogger(__name__)
logger.setLevel(data_loader_logger.level)
sns.set_theme(style="whitegrid", context="notebook")


def _json_default(value: Any) -> Any:
    """Convert common NumPy and pandas values into JSON-compatible values."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.ndarray, pd.Series, pd.Index)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _save_figure(figure: plt.Figure, figures_dir: Path, filename: str) -> None:
    """Save a figure and release its resources."""
    output_path = figures_dir / filename
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figure: %s", output_path)


def _description_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate text statistics and the fifty most frequent terms."""
    descriptions = df["description"].fillna("").astype(str)
    vectorizer = CountVectorizer(stop_words="english", max_features=50)

    try:
        matrix = vectorizer.fit_transform(descriptions)
        terms = vectorizer.get_feature_names_out()
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        top_words = {
            term: int(count)
            for term, count in sorted(
                zip(terms, counts), key=lambda item: item[1], reverse=True
            )
        }
    except ValueError:
        top_words = {}

    text_lengths = descriptions.str.len()
    description_stats = df.assign(description_length=text_lengths).groupby(
        TARGET_COLUMN
    )["description_length"].mean().sort_values(ascending=False)

    return {
        "top_50_words": top_words,
        "average_description_length_by_category": description_stats.round(2).to_dict(),
        "average_description_length": float(text_lengths.mean()),
        "median_description_length": float(text_lengths.median()),
    }


def _amount_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate amount statistics and IQR-based outlier information."""
    amount = pd.to_numeric(df["amount"], errors="coerce")
    first_quartile = amount.quantile(0.25)
    third_quartile = amount.quantile(0.75)
    iqr = third_quartile - first_quartile
    lower_bound = first_quartile - 1.5 * iqr
    upper_bound = third_quartile + 1.5 * iqr
    outliers = (amount < lower_bound) | (amount > upper_bound)

    category_stats = df.assign(amount=amount).groupby(TARGET_COLUMN)["amount"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )

    return {
        "overall_statistics": amount.describe().round(2).to_dict(),
        "statistics_by_category": category_stats.round(2).to_dict(orient="index"),
        "iqr": {
            "q1": float(first_quartile),
            "q3": float(third_quartile),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        },
        "outlier_count": int(outliers.sum()),
        "outlier_percentage": float(outliers.mean() * 100),
    }


def _time_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate calendar and transaction-frequency statistics."""
    dates = pd.to_datetime(df["date"], errors="coerce")
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekday_counts = df["day_of_week"].value_counts().reindex(weekday_order, fill_value=0)
    monthly_counts = dates.dt.to_period("M").value_counts().sort_index()
    quarterly_counts = dates.dt.to_period("Q").value_counts().sort_index()

    return {
        "transactions_by_month": {
            str(period): int(count) for period, count in monthly_counts.items()
        },
        "transactions_by_weekday": weekday_counts.astype(int).to_dict(),
        "transactions_by_quarter": {
            str(period): int(count) for period, count in quarterly_counts.items()
        },
        "date_range": {
            "start": dates.min(),
            "end": dates.max(),
            "days": int((dates.max() - dates.min()).days),
        },
    }


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create candidate features for downstream modeling experiments."""
    features = df.copy()
    descriptions = features["description"].fillna("").astype(str)
    dates = pd.to_datetime(features["date"], errors="coerce")

    features["description_word_count"] = descriptions.str.split().str.len()
    features["description_char_count"] = descriptions.str.len()
    features["description_digit_count"] = descriptions.str.count(r"\d")
    features["description_special_char_count"] = descriptions.str.count(r"[^\w\s]")
    features["month"] = dates.dt.month
    features["quarter"] = dates.dt.quarter
    features["is_weekend"] = dates.dt.dayofweek >= 5
    features["day_of_month"] = dates.dt.day
    features["log_amount"] = np.log1p(pd.to_numeric(features["amount"], errors="coerce"))
    features["amount_bins"] = pd.cut(
        features["amount"], bins=[-np.inf, 25, 100, np.inf], labels=["small", "medium", "large"]
    )
    return features


def _create_visualizations(df: pd.DataFrame, figures_dir: Path) -> None:
    """Create and save the EDA plots."""
    category_counts = df[TARGET_COLUMN].value_counts().sort_values()
    figure, axis = plt.subplots(figsize=(10, 6))
    category_counts.plot.barh(ax=axis, color="#2a9d8f")
    axis.set(title="Target Distribution", xlabel="Transactions", ylabel="Category")
    _save_figure(figure, figures_dir, "target_distribution.png")

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["amount"], bins=40, kde=True, ax=axes[0], color="#e76f51")
    axes[0].set(title="Amount Distribution", xlabel="Amount", ylabel="Transactions")
    sns.boxplot(x=df["amount"], ax=axes[1], color="#f4a261")
    axes[1].set(title="Amount Boxplot", xlabel="Amount")
    _save_figure(figure, figures_dir, "amount_distribution.png")

    figure, axis = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=df, x=TARGET_COLUMN, y="amount", ax=axis, color="#457b9d")
    axis.tick_params(axis="x", rotation=35)
    axis.set(title="Amount by Category", xlabel="Category", ylabel="Amount")
    _save_figure(figure, figures_dir, "amount_by_category.png")

    dates = pd.to_datetime(df["date"], errors="coerce")
    monthly_counts = dates.dt.to_period("M").value_counts().sort_index()
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekday_counts = df["day_of_week"].value_counts().reindex(weekday_order, fill_value=0)
    figure, axes = plt.subplots(2, 1, figsize=(12, 9))
    axes[0].plot(monthly_counts.index.astype(str), monthly_counts.values, marker="o")
    axes[0].set(title="Transactions over Time", xlabel="Month", ylabel="Transactions")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(weekday_counts.index, weekday_counts.values, color="#264653")
    axes[1].set(title="Transactions by Weekday", xlabel="Day", ylabel="Transactions")
    axes[1].tick_params(axis="x", rotation=25)
    _save_figure(figure, figures_dir, "time_analysis.png")

    descriptions = df["description"].fillna("").astype(str)
    vectorizer = CountVectorizer(stop_words="english", max_features=50)
    try:
        matrix = vectorizer.fit_transform(descriptions)
        terms = vectorizer.get_feature_names_out()
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        top_terms = pd.Series(counts, index=terms).sort_values().tail(20)
        figure, axis = plt.subplots(figsize=(10, 7))
        top_terms.plot.barh(ax=axis, color="#e9c46a")
        axis.set(title="Top Description Terms", xlabel="Occurrences", ylabel="Term")
        _save_figure(figure, figures_dir, "top_description_terms.png")
    except ValueError:
        logger.warning("No valid description terms available for plotting")


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print the main findings in a readable format."""
    overview = summary["overview"]
    amount = summary["amount_analysis"]
    time = summary["time_analysis"]
    print("=" * 70)
    print("AuroraGate Exploratory Data Analysis")
    print("=" * 70)
    print(f"Shape: {overview['shape'][0]} rows x {overview['shape'][1]} columns")
    print(f"Columns: {', '.join(overview['columns'])}")
    print(f"Missing values: {overview['total_missing_values']}")
    print("\nTarget distribution:")
    for category, count in overview["target_distribution"].items():
        print(f"  {category}: {count}")
    print(f"\nAmount outliers (IQR): {amount['outlier_count']} ({amount['outlier_percentage']:.2f}%)")
    print(f"Date range: {time['date_range']['start']} to {time['date_range']['end']}")
    print("\nTop description words:")
    print(", ".join(summary["text_analysis"]["top_50_words"].keys()))
    print("=" * 70)


def run_eda(train_file: str = "train.csv") -> Dict[str, Any]:
    """Run all EDA steps, save figures and write the JSON summary report."""
    experiments_dir = Path(EXPERIMENTS_DIR)
    figures_dir = experiments_dir / "figures"
    try:
        figures_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        logger.error("Could not create EDA output directory %s: %s", figures_dir, error)
        raise

    loader = DataLoader(use_cache=False)
    df = loader.load_train_data(train_file=train_file, force_reload=True)
    required_columns = {
        "transaction_id",
        "date",
        "description",
        "amount",
        "day_of_week",
        TARGET_COLUMN,
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"EDA requires missing columns: {sorted(missing_columns)}")

    engineered = _engineer_features(df)
    transaction_id = pd.to_numeric(df["transaction_id"], errors="coerce")
    sequential_id = bool(
        transaction_id.notna().all()
        and transaction_id.is_monotonic_increasing
        and transaction_id.diff().dropna().eq(1).all()
    )
    overview = {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(),
        "total_missing_values": int(df.isna().sum().sum()),
        "target_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
    }
    summary = {
        "overview": overview,
        "text_analysis": _description_analysis(df),
        "amount_analysis": _amount_analysis(df),
        "time_analysis": _time_analysis(df),
        "feature_engineering": {
            "candidate_columns": [
                "description_word_count",
                "description_char_count",
                "description_digit_count",
                "description_special_char_count",
                "month",
                "quarter",
                "is_weekend",
                "day_of_month",
                "log_amount",
                "amount_bins",
            ],
            "transaction_id_is_sequential": sequential_id,
            "transaction_id_suggestion": (
                "Use transaction_id as a time index because it is sequential."
                if sequential_id
                else "Do not treat transaction_id as a time index until ordering is verified."
            ),
            "engineered_feature_preview": engineered.head(3).to_dict(orient="records"),
        },
    }
    _create_visualizations(df, figures_dir)

    report_path = experiments_dir / "eda_summary.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(summary, report_file, indent=2, ensure_ascii=False, default=_json_default)
    logger.info("Saved EDA summary: %s", report_path)
    _print_summary(summary)
    return summary


if __name__ == "__main__":
    run_eda()