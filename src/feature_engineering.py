"""Feature engineering utilities for AuroraGate transaction data."""

import re
from typing import Iterable

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


def engineer_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
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
    features["transaction_gap"] = transaction_ids.diff().fillna(0)
    features["transaction_number_in_day"] = features.groupby(dates.dt.normalize(), dropna=False).cumcount() + 1

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