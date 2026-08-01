"""Feature engineering utilities for AuroraGate transaction data.

This module provides comprehensive feature extraction and engineering capabilities
following the modular design patterns from the Transformer architecture.

Classes:
    StoreNameExtractor: Extracts and normalizes merchant names from descriptions
    TextFeatureExtractor: Extracts text-based features from descriptions
    AmountFeatureExtractor: Extracts amount-based features
    TemporalFeatureExtractor: Extracts time-based features
    TransactionFeatureExtractor: Extracts transaction sequence features

Functions:
    engineer_features: Main feature engineering pipeline
    add_target_encoding: Adds target encoding with K-fold smoothing
    fit_target_encoding_stats: Fits target encoding statistics
"""

import re
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE


class StoreNameExtractor:
    """Extracts and normalizes merchant names from transaction descriptions.
    
    This class implements the store name extraction logic following the pattern
    matching approach similar to the Transformer's input embedding layer.
    
    Attributes:
        STATE_CODES: Regular expression pattern for US state codes
        STATE_PATTERN: Compiled regex for state code detection
        STORE_PATTERNS: Tuple of compiled regex patterns for merchant extraction
        MAX_STORE_LENGTH: Maximum length for extracted store names
    """
    
    STATE_CODES = (
        "AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|"
        "NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC"
    )
    STATE_PATTERN = re.compile(rf"(?<![A-Z])(?:{STATE_CODES})(?![A-Z])")
    STORE_PATTERNS = (
        re.compile(r"\b(?:SQ\s*\*|SQUARE\s*\*|AMZN\s*Mktp\s*\*?|AMAZON\s*\*?|WMT\s*#?)\s*([A-Z0-9][A-Z0-9 &.'-]{1,30})", re.I),
        re.compile(r"\b(?:UBER|LYFT|SPOTIFY|NETFLIX|WHOLE\s+FOODS|TARGET|WALMART|COSTCO|CVS|SHELL|CHEVRON)\b", re.I),
    )
    MAX_STORE_LENGTH = 40
    
    def __init__(self):
        """Initialize the StoreNameExtractor with compiled patterns."""
        pass
    
    def __call__(self, description: str) -> str:
        """Extract and normalize merchant name from description.
        
        Args:
            description: Raw transaction description string
            
        Returns:
            Normalized store name (uppercase, trimmed, limited to MAX_STORE_LENGTH)
            
        Examples:
            >>> extractor = StoreNameExtractor()
            >>> extractor("SQ *WHOLE FOODS STORE")
            'WHOLE FOODS STORE'
            >>> extractor("AMAZON MKTP US*1234567")
            'AMAZON MKTP US*1234567'
        """
        text = str(description).upper().strip()
        
        # Try pattern matching first
        for pattern in self.STORE_PATTERNS:
            match = pattern.search(text)
            if match:
                store = match.group(1) if match.lastindex else match.group(0)
                store = re.sub(r"\s+", " ", store).strip(" *#.-")
                return store[:self.MAX_STORE_LENGTH] or "UNKNOWN"
        
        # Fallback to first token extraction
        first_token = re.split(r"\s+|\*|#", text, maxsplit=1)[0]
        return first_token[:self.MAX_STORE_LENGTH] if first_token else "UNKNOWN"


class TextFeatureExtractor:
    """Extracts text-based features from transaction descriptions.
    
    This class implements various text analysis features similar to the
    Transformer's input embedding and positional encoding layers.
    
    Attributes:
        KEYWORD_PATTERNS: Dictionary of keyword patterns for different categories
    """
    
    KEYWORD_PATTERNS = {
        "has_grocery_keyword": r"grocery|groceries|whole\s+foods|walmart|safeway|kroger|aldi|trader\s+joe|supercenter|market",
        "has_food_keyword": r"restaurant|food|cafe|coffee|starbucks|chipotle|mcdonald|pizza|doordash|grubhub",
        "has_transport_keyword": r"uber|lyft|taxi|shell|chevron|exxon|fuel|gas|parking|toll|transit|train",
        "has_transfer_keyword": r"transfer|payment|paypal|venmo|zelle|wire|withdrawal|deposit",
        "has_travel_keyword": r"air|airline|hotel|motel|flight|booking|travel|airport|expedia|airbnb",
    }
    
    SUBSCRIPTION_KEYWORDS = [
        'NETFLIX', 'SPOTIFY', 'AMAZON PRIME', 'HULU', 'DISNEY', 'HBOMAX',
        'SUBSCRIPTION', 'MONTHLY', 'RECURRING', 'RENEWAL', 'AUTO-PAY', 'MEMBERSHIP'
    ]
    
    ENTERTAINMENT_KEYWORDS = [
        'CINEMA', 'MOVIE', 'CONCERT', 'THEATER', 'BOWLING', 'AMUSEMENT', 'ARCADE',
        'PARK', 'ZOO', 'MUSEUM', 'GALLERY', 'FESTIVAL', 'EVENT', 'SHOW'
    ]
    
    def __init__(self):
        """Initialize the TextFeatureExtractor."""
        pass
    
    def extract_keyword_features(self, descriptions: pd.Series) -> pd.DataFrame:
        """Extract keyword-based binary features from descriptions.
        
        Args:
            descriptions: Series of transaction descriptions
            
        Returns:
            DataFrame with binary keyword features
            
        Examples:
            >>> extractor = TextFeatureExtractor()
            >>> descriptions = pd.Series(["UBER TRIP", "WHOLE FOODS STORE"])
            >>> extractor.extract_keyword_features(descriptions)
        """
        features = {}
        upper_descriptions = descriptions.str.upper()
        
        for feature_name, pattern in self.KEYWORD_PATTERNS.items():
            features[feature_name] = upper_descriptions.str.contains(pattern, regex=True).astype(int)
        
        # Add subscription detection features
        features['is_subscription_keyword'] = upper_descriptions.str.contains(
            '|'.join(self.SUBSCRIPTION_KEYWORDS), regex=True
        ).astype(int)
        
        # Add entertainment keyword count
        entertainment_counts = upper_descriptions.apply(
            lambda x: sum(1 for kw in self.ENTERTAINMENT_KEYWORDS if kw in x)
        )
        features['entertainment_keyword_count'] = entertainment_counts.astype(int)
        
        # Add specific merchant detection for entertainment
        features['is_entertainment_merchant'] = upper_descriptions.str.contains(
            r'CINEMA|MOVIE|THEATER|BOWLING|AMUSEMENT|ARCADE', regex=True
        ).astype(int)
        
        return pd.DataFrame(features, index=descriptions.index)
    
    def extract_text_statistics(self, descriptions: pd.Series) -> pd.DataFrame:
        """Extract text statistics features from descriptions.
        
        Args:
            descriptions: Series of transaction descriptions
            
        Returns:
            DataFrame with text statistics features
        """
        return pd.DataFrame({
            'description_word_count': descriptions.str.split().str.len().astype(int),
            'description_char_count': descriptions.str.len().astype(int),
            'description_digit_count': descriptions.str.count(r"\d").astype(int),
            'description_special_char_count': descriptions.str.count(r"[^\w\s]").astype(int),
        }, index=descriptions.index)


class AmountFeatureExtractor:
    """Extracts amount-based features from transaction amounts.
    
    This class implements various numerical features similar to the
    Transformer's positional encoding for numerical values.
    """
    
    def __init__(self):
        """Initialize the AmountFeatureExtractor."""
        pass
    
    def extract_amount_features(self, amounts: pd.Series, dates: pd.Series) -> pd.DataFrame:
        """Extract comprehensive amount-based features.
        
        Args:
            amounts: Series of transaction amounts
            dates: Series of transaction dates
            
        Returns:
            DataFrame with amount-based features
        """
        features = pd.DataFrame(index=amounts.index)
        
        # Basic amount features
        features['is_round_amount'] = amounts.mod(1).fillna(0).eq(0).astype(int)
        features['decimal_digits'] = amounts.map(self._decimal_digits).astype(int)
        features['log_amount'] = np.log1p(amounts.clip(lower=0))
        
        # Enhanced amount pattern features
        features['amount_roundness'] = amounts.mod(1).fillna(0).abs()  # Distance from round number
        features['is_common_subscription_amount'] = amounts.isin([9.99, 12.99, 14.99, 19.99]).astype(int)
        features['is_high_value'] = (amounts >= 100).astype(int)
        features['is_low_value'] = (amounts < 10).astype(int)
        
        # Amount bins (categorical)
        features['amount_bins'] = pd.cut(
            amounts, 
            bins=[-np.inf, 25, 100, np.inf], 
            labels=["small", "medium", "large"]
        ).astype(object).fillna("unknown")
        
        # Temporal amount features
        features['amount_percentile'] = amounts.groupby(dates.dt.to_period("M"), dropna=False).rank(pct=True)
        
        return features
    
    def _decimal_digits(self, value: object) -> int:
        """Return the number of decimal digits in a monetary value.
        
        Args:
            value: Monetary value
            
        Returns:
            Number of decimal digits (0 for integers or missing values)
        """
        if pd.isna(value):
            return 0
        text = f"{float(value):.10f}".rstrip("0")
        return len(text.split(".", 1)[1]) if "." in text else 0


class TemporalFeatureExtractor:
    """Extracts temporal features from transaction dates.
    
    This class implements time-based features similar to the
    Transformer's positional encoding for sequence data.
    """
    
    def __init__(self):
        """Initialize the TemporalFeatureExtractor."""
        pass
    
    def extract_temporal_features(self, dates: pd.Series) -> pd.DataFrame:
        """Extract comprehensive temporal features from dates.
        
        Args:
            dates: Series of transaction dates
            
        Returns:
            DataFrame with temporal features
        """
        # Extract time components if available
        has_time = dates.dt.time.any()
        
        features = {
            'day_of_year': dates.dt.dayofyear.fillna(0).astype(int),
            'week_of_year': dates.dt.isocalendar().week.fillna(0).astype(int),
            'month': dates.dt.month.fillna(0).astype(int),
            'quarter': dates.dt.quarter.fillna(0).astype(int),
            'day_of_month': dates.dt.day.fillna(0).astype(int),
            'days_to_weekend': dates.dt.dayofweek.map(self._days_to_weekend).fillna(0).astype(int),
            'is_weekend': dates.dt.dayofweek.ge(5).fillna(False).astype(int),
            'month_period': pd.cut(
                dates.dt.day, bins=[0, 10, 20, 31], labels=["start", "mid", "end"]
            ).astype(object).fillna("unknown"),
        }
        
        # Add time-of-day features if time data is available
        if has_time:
            features['hour'] = dates.dt.hour.fillna(0).astype(int)
            features['is_lunch_hour'] = dates.dt.hour.between(11, 14).astype(int)
            features['is_dinner_hour'] = dates.dt.hour.between(18, 21).astype(int)
            features['is_business_hours'] = dates.dt.hour.between(9, 17).astype(int)
        else:
            # Default values if no time data
            features['hour'] = 12  # Midday
            features['is_lunch_hour'] = 0
            features['is_dinner_hour'] = 0
            features['is_business_hours'] = 1
        
        return pd.DataFrame(features, index=dates.index)
    
    def _days_to_weekend(self, day_number: int) -> int:
        """Calculate days until Saturday (weekend).
        
        Args:
            day_number: Day of week (0=Monday, 6=Sunday)
            
        Returns:
            Days until Saturday (0 for weekend days)
        """
        return int(max(0, 5 - day_number))


class TransactionFeatureExtractor:
    """Extracts transaction sequence features.
    
    This class implements sequence-based features similar to the
    Transformer's attention to positional relationships.
    """
    
    def __init__(self):
        """Initialize the TransactionFeatureExtractor."""
        pass
    
    def extract_sequence_features(self, transaction_ids: pd.Series, dates: pd.Series) -> pd.DataFrame:
        """Extract transaction sequence and timing features.
        
        Args:
            transaction_ids: Series of transaction IDs
            dates: Series of transaction dates
            
        Returns:
            DataFrame with sequence features
        """
        features = pd.DataFrame(index=transaction_ids.index)
        
        # Transaction sequence features
        features['time_since_last_transaction'] = transaction_ids.diff().fillna(0)
        features['transaction_count_per_day'] = (
            pd.Series(transaction_ids.index).groupby(dates.dt.normalize(), dropna=False).cumcount() + 1
        )
        features['transaction_gap'] = features['time_since_last_transaction']
        features['transaction_number_in_day'] = features['transaction_count_per_day']
        
        # State detection
        descriptions = pd.Series(["" for _ in range(len(transaction_ids))], index=transaction_ids.index)
        features['has_state_code'] = descriptions.str.upper().str.contains(
            StoreNameExtractor.STATE_PATTERN, regex=True
        ).astype(int)
        
        return features


# Initialize extractors (singleton pattern)
_store_extractor = StoreNameExtractor()
_text_extractor = TextFeatureExtractor()
_amount_extractor = AmountFeatureExtractor()
_temporal_extractor = TemporalFeatureExtractor()
_transaction_extractor = TransactionFeatureExtractor()


def engineer_features(
    df: pd.DataFrame,
    is_train: bool = True,
    target_encoding_stats: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Engineer comprehensive features from transaction data using modular extractors.
    
    This function follows the Transformer architecture pattern of composing
    multiple specialized modules (extractors) to build complex features.
    """
    # Input validation (like Transformer input validation)
    required_columns = {"transaction_id", "date", "description", "amount", "day_of_week"}
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        raise ValueError(
            f"Feature engineering requires missing columns: {sorted(missing_columns)}".upper()
        )
    
    # Validate data types (like Transformer tensor validation)
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            raise ValueError(f"Cannot convert 'date' to datetime: {e}") from e
    
    # Validate numeric columns (like Transformer numerical validation)
    try:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['transaction_id'] = pd.to_numeric(df['transaction_id'], errors='coerce')
    except Exception as e:
        raise ValueError(f"Cannot convert numeric columns: {e}") from e
    
    # Validate no empty descriptions (like Transformer input masking)
    if df['description'].isna().all():
        raise ValueError("All descriptions are missing - cannot extract features")
    
    # Fill missing values (like Transformer padding)
    df = df.copy()
    df['description'] = df['description'].fillna("")
    df['amount'] = df['amount'].fillna(0)
    df['transaction_id'] = df['transaction_id'].fillna(0)
    
    # Feature extraction pipeline (like Transformer forward pass)
    features = df.copy()
    
    # 1. Store name extraction (like token embedding)
    descriptions = features["description"].fillna("").astype(str)
    features["store_name"] = descriptions.apply(_store_extractor).astype(str)
    
    # 2. Recurring transaction detection (like self-attention)
    features = add_recurring_features(features)
    
    # 3. Text features (like positional encoding)
    text_features = _text_extractor.extract_text_statistics(descriptions)
    keyword_features = _text_extractor.extract_keyword_features(descriptions)
    
    # 4. Amount features (like value encoding)
    amounts = pd.to_numeric(features["amount"], errors="coerce")
    dates = pd.to_datetime(features["date"], errors="coerce")
    amount_features = _amount_extractor.extract_amount_features(amounts, dates)
    
    # 5. Temporal features (like temporal encoding)
    temporal_features = _temporal_extractor.extract_temporal_features(dates)
    
    # 6. Transaction sequence features (like attention patterns)
    transaction_ids = pd.to_numeric(features["transaction_id"], errors="coerce")
    sequence_features = _transaction_extractor.extract_sequence_features(transaction_ids, dates)
    
    # Combine all features (like Transformer output concatenation)
    all_features = [features, text_features, keyword_features, amount_features, temporal_features, sequence_features]
    
    for feature_df in all_features:
        # Validate feature dimensions (like Transformer shape validation)
        assert len(feature_df) == len(features), f"Feature dimension mismatch: {len(feature_df)} != {len(features)}"
        assert feature_df.index.is_unique, "Feature index must be unique"
    
    # Concatenate features (like Transformer final output)
    result = pd.concat(all_features, axis=1)
    
    # Final validation (like Transformer output validation)
    assert len(result) == len(df), "Final feature count mismatch"
    assert not result.empty, "Empty feature result"
    
    return result


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
        features["store_name"] = descriptions.apply(_store_extractor).astype(str)

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
        else df["description"].fillna("").astype(str).apply(_store_extractor)
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
    random_state: int = RANDOM_STATE,
    target_encoding_stats: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Add smoothed, K-fold store target encoding without validation leakage."""
    # random_state parameter is kept for API compatibility but not used
    # since we use RANDOM_STATE from config for reproducibility
    features = df.copy()
    if "store_name" not in features.columns:
        descriptions = features["description"].fillna("").astype(str)
        features["store_name"] = descriptions.apply(_store_extractor).astype(str)

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
            from src.config import RANDOM_STATE
            splitter = KFold(n_splits=fold_count, shuffle=True, random_state=RANDOM_STATE)
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
    """Engineer comprehensive features from transaction data using modular extractors.
    
    This function follows the Transformer architecture pattern of composing
    multiple specialized modules (extractors) to build complex features.
    
    The feature engineering pipeline includes:
    1. Input validation and preprocessing (like Transformer embedding)
    2. Store name extraction (like token embedding)
    3. Text feature extraction (like positional encoding)
    4. Amount feature extraction (like value encoding)
    5. Temporal feature extraction (like temporal encoding)
    6. Transaction sequence features (like attention patterns)
    7. Recurring transaction detection (like self-attention)
    
    Args:
        df: Input DataFrame containing transaction data. Must include:
            - transaction_id: Unique transaction identifier (like position IDs)
            - date: Transaction date (like temporal positions)
            - description: Transaction description (like input tokens)
            - amount: Transaction amount (like input values)
            - day_of_week: Day of week (like categorical embeddings)
        is_train: Boolean indicating if this is training data. Kept for API
                 compatibility but features are deterministic (like frozen layers).
        target_encoding_stats: Optional pre-computed target encoding statistics
                             for test data consistency (like learned embeddings).
        
    Returns:
        DataFrame with original columns plus engineered features. The output
        follows the Transformer pattern of enriched representations:
        
        - Input features (original columns): transaction_id, date, description, amount, day_of_week
        - Embedded features: store_name (like token embeddings)
        - Positional features: day_of_year, week_of_year, month_period (like positional encodings)
        - Value features: amount_bins, log_amount, amount_percentile (like value encodings)
        - Attention features: is_recurring, store_frequency (like self-attention scores)
        - Sequence features: transaction_gap, transaction_number_in_day (like sequence patterns)
        - Categorical features: has_grocery_keyword, has_food_keyword, etc. (like classification tokens)
        
    Raises:
        ValueError: If required columns are missing (like missing input embeddings)
        
    Note:
        Target encoding is currently disabled until revalidated against
        the chronological test distribution (like frozen embedding layers).
        
    Examples:
        >>> import pandas as pd
        >>> from src.feature_engineering import engineer_features
        >>> 
        >>> # Create sample data
        >>> data = pd.DataFrame({
        ...     'transaction_id': [1, 2, 3],
        ...     'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        ...     'description': ['AMAZON MARKETPLACE', 'WHOLE FOODS', 'UBER TRIP'],
        ...     'amount': [25.50, 42.75, 15.20],
        ...     'day_of_week': ['Sunday', 'Monday', 'Tuesday']
        ... })
        >>> 
        >>> # Apply feature engineering
        >>> engineered = engineer_features(data)
        >>> print(f"Engineered {len(engineered.columns)} features from {len(data)} transactions")
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

    features["store_name"] = descriptions.apply(_store_extractor).astype(str)
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

    features["has_state_code"] = descriptions.str.upper().str.contains(StoreNameExtractor.STATE_PATTERN, regex=True).astype(int)
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
    for feature_name, pattern in TextFeatureExtractor.KEYWORD_PATTERNS.items():
        features[feature_name] = upper_descriptions.str.contains(pattern, regex=True).astype(int)

    return features


def categorical_feature_names() -> Iterable[str]:
    """Return engineered columns that should be treated as categorical."""
    return ("day_of_week", "month", "quarter", "month_period", "amount_bins", "store_name")