"""Feature engineering utilities for AuroraGate transaction data."""

import re
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


STATE_CODES = (
    "AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|"
    "NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC"
)
STATE_PATTERN = re.compile(rf"(?<![A-Z])(?:{STATE_CODES})(?![A-Z])")
STORE_PATTERNS = (
    re.compile(r"\b(?:SQ\s*\*|SQUARE\s*\*|AMZN\s*Mktp\s*\*?|AMAZON\s*\*?|WMT\s*#?)\s*([A-Z0-9][A-Z0-9 &.'-]{1,30})", re.I),
    re.compile(r"\b(?:UBER|LYFT|SPOTIFY|NETFLIX|WHOLE\s+FOODS|TARGET|WALMART|COSTCO|CVS|SHELL|CHEVRON)\b", re.I),
)

KEYWORD_PATTERNS = {
    "has_grocery_keyword": r"grocery|groceries|whole\s+foods|walmart|safeway|kroger|aldi|trader\s+joe|supercenter|market",
    "has_food_keyword": r"restaurant|food|cafe|coffee|starbucks|chipotle|mcdonald|pizza|doordash|grubhub",
    "has_transport_keyword": r"uber|lyft|taxi|shell|chevron|exxon|fuel|gas|parking|toll|transit|train",
    "has_transfer_keyword": r"transfer|payment|paypal|venmo|zelle|wire|withdrawal|deposit",
    "has_travel_keyword": r"air|airline|hotel|motel|flight|booking|travel|airport|expedia|airbnb",
}


def _extract_store_name(description: str) -> str:
    """Extract a normalized merchant or store name from a description."""
    text = str(description).upper().strip()
    for pattern in STORE_PATTERNS:
        match = pattern.search(text)
        if match:
            store = match.group(1) if match.lastindex else match.group(0)
            store = re.sub(r"\s+", " ", store).strip(" *#.-")
            return store[:40] or "UNKNOWN"
    first_token = re.split(r"\s+|\*|#", text, maxsplit=1)[0]
    return first_token[:40] if first_token else "UNKNOWN"


def _decimal_digits(value: object) -> int:
    """Return the number of decimal digits in a monetary value."""
    if pd.isna(value):
        return 0
    text = f"{float(value):.10f}".rstrip("0")
    return len(text.split(".", 1)[1]) if "." in text else 0


def _days_to_weekend(day_number: int) -> int:
    """Return days until Saturday, with weekend days represented as zero."""
    return int(max(0, 5 - day_number))


def add_recurring_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add store-frequency and repeated-amount indicators."""
    features = df.copy()
    if "store_name" not in features.columns:
        descriptions = features["description"].fillna("").astype(str)
        features["store_name"] = descriptions.map(_extract_store_name).astype(str)

    store_counts = features.groupby("store_name")["transaction_id"].transform("count")
    features["store_frequency"] = store_counts.fillna(0).astype("int32")

    amount_store_pair = (
        features["store_name"].astype(str)
        + "_"
        + pd.to_numeric(features["amount"], errors="coerce").round(8).astype(str)
    )
    pair_counts = amount_store_pair.groupby(amount_store_pair).transform("count")
    features["is_recurring"] = pair_counts.gt(1).astype("int8")
    return features


def fit_target_encoding_stats(
    df: pd.DataFrame, target_col: str = "category"
) -> Dict[str, Any]:
    """Fit leakage-safe store target-encoding statistics from training data."""
    if target_col not in df.columns:
        raise ValueError(f"Target column is required to fit encoding: {target_col}")
    store_names = (
        df["store_name"].astype(str)
        if "store_name" in df.columns
        else df["description"].fillna("").astype(str).map(_extract_store_name)
    )
    values = pd.Categorical(df[target_col])
    encoded_target = pd.Series(values.codes, index=df.index, dtype="float64")
    working = pd.DataFrame(
        {
            "store_name": store_names,
            "encoded_target": encoded_target,
        },
        index=df.index,
    )
    return {
        "store_means": working.groupby("store_name")["encoded_target"].mean().to_dict(),
        "global_mean": float(encoded_target[encoded_target.ge(0)].mean()),
    }


def add_target_encoding(
    df: pd.DataFrame,
    target_col: str = "category",
    k_fold: int = 5,
    random_state: int = 42,
    target_encoding_stats: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Add smoothed, K-fold store target encoding without validation leakage."""
    del random_state
    features = df.copy()
    if "store_name" not in features.columns:
        descriptions = features["description"].fillna("").astype(str)
        features["store_name"] = descriptions.map(_extract_store_name).astype(str)

    if target_col in features.columns:
        target_codes = pd.Series(
            pd.Categorical(features[target_col]).codes,
            index=features.index,
            dtype="float64",
        )
        valid_codes = target_codes.ge(0)
        global_mean = float(target_codes[valid_codes].mean())
        encoded = pd.Series(global_mean, index=features.index, dtype="float64")
        fold_count = min(k_fold, len(features))
        if fold_count >= 2:
            from sklearn.model_selection import KFold

            splitter = KFold(n_splits=fold_count, shuffle=True, random_state=42)
            for train_indices, valid_indices in splitter.split(features):
                train_rows = features.iloc[train_indices].copy()
                train_rows["_encoded_target"] = target_codes.iloc[train_indices].to_numpy()
                means = train_rows.groupby("store_name")["_encoded_target"].mean()
                encoded.iloc[valid_indices] = (
                    features.iloc[valid_indices]["store_name"]
                    .map(means)
                    .fillna(global_mean)
                    .to_numpy()
                )
        features["store_name_target_enc"] = encoded.astype("float32")
    elif target_encoding_stats is not None:
        store_means = target_encoding_stats.get("store_means", {})
        global_mean = float(target_encoding_stats.get("global_mean", 0.0))
        features["store_name_target_enc"] = (
            features["store_name"].map(store_means).fillna(global_mean).astype("float32")
        )
    else:
        features["store_name_target_enc"] = 0.0
    return features


def engineer_features(
    df: pd.DataFrame,
    is_train: bool = True,
    target_encoding_stats: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Add transaction, text, calendar, amount and keyword features.

    Args:
        df: Training or test transactions.
        is_train: Kept for API compatibility; the same deterministic features
            are produced for both training and test frames.
    """
    del is_train
    required = {"transaction_id", "date", "description", "amount", "day_of_week"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {sorted(missing)}")

    features = df.copy()
    descriptions = features["description"].fillna("").astype(str)
    dates = pd.to_datetime(features["date"], errors="coerce")
    amounts = pd.to_numeric(features["amount"], errors="coerce")
    transaction_ids = pd.to_numeric(features["transaction_id"], errors="coerce")

    features["store_name"] = descriptions.map(_extract_store_name).astype(str)
    features = add_recurring_features(features)
    # Target encoding is intentionally disabled until it is revalidated against
    # the chronological test distribution.
    features["is_round_amount"] = amounts.mod(1).fillna(0).eq(0).astype(int)
    features["decimal_digits"] = amounts.map(_decimal_digits).astype(int)
    features["amount_percentile"] = amounts.groupby(dates.dt.to_period("M"), dropna=False).rank(pct=True)
    features["day_of_year"] = dates.dt.dayofyear.fillna(0).astype(int)
    features["week_of_year"] = dates.dt.isocalendar().week.fillna(0).astype(int)
    features["month_period"] = pd.cut(
        dates.dt.day, bins=[0, 10, 20, 31], labels=["start", "mid", "end"]
    ).astype(object).fillna("unknown")
    features["days_to_weekend"] = dates.dt.dayofweek.map(_days_to_weekend).fillna(0).astype(int)
    features["is_weekend"] = dates.dt.dayofweek.ge(5).fillna(False).astype(int)
    features["month"] = dates.dt.month.fillna(0).astype(int)
    features["quarter"] = dates.dt.quarter.fillna(0).astype(int)
    features["day_of_month"] = dates.dt.day.fillna(0).astype(int)

    features["has_state_code"] = descriptions.str.upper().str.contains(STATE_PATTERN, regex=True).astype(int)
    features["time_since_last_transaction"] = transaction_ids.diff().fillna(0)
    features["transaction_count_per_day"] = (
        features.groupby(dates.dt.normalize(), dropna=False).cumcount() + 1
    )
    features["transaction_gap"] = features["time_since_last_transaction"]
    features["transaction_number_in_day"] = features["transaction_count_per_day"]

    features["description_word_count"] = descriptions.str.split().str.len().astype(int)
    features["description_char_count"] = descriptions.str.len().astype(int)
    features["description_digit_count"] = descriptions.str.count(r"\d").astype(int)
    features["description_special_char_count"] = descriptions.str.count(r"[^\w\s]").astype(int)
    features["log_amount"] = np.log1p(amounts.clip(lower=0))
    features["amount_bins"] = pd.cut(
        amounts, bins=[-np.inf, 25, 100, np.inf], labels=["small", "medium", "large"]
    ).astype(object).fillna("unknown")

    upper_descriptions = descriptions.str.upper()
    for feature_name, pattern in KEYWORD_PATTERNS.items():
        features[feature_name] = upper_descriptions.str.contains(pattern, regex=True).astype(int)

    return features


def categorical_feature_names() -> Iterable[str]:
    """Return engineered columns that should be treated as categorical."""
    return ("day_of_week", "month", "quarter", "month_period", "amount_bins", "store_name")