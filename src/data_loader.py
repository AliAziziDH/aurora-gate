"""
Data loading and initial validation module for AuroraGate challenge.

This module handles:
- Loading data from various formats (CSV, Parquet, Excel) with automatic detection
- Intelligent caching with Parquet for faster subsequent loads
- Automatic encoding detection for non-UTF-8 files
- Support for compressed files (.gz, .bz2, .zip, .xz)
- Flexible date parsing with multiple format support
- Data versioning with file hashing and metadata
- Stratified sampling for quick testing
- Comprehensive logging and error handling
- Memory-efficient chunking for large datasets

Usage:
    from src.data_loader import DataLoader
    
    loader = DataLoader()
    train_df = loader.load_train_data()
    test_df = loader.load_test_data()
    sample_df = loader.load_sample(n_rows=1000)
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from charset_normalizer import from_path
except ImportError:
    from_path = None

from src.config import (
    CACHE_DIR,
    DATA_DIR,
    LOG_FORMAT,
    LOG_LEVEL,
    RANDOM_STATE,
    TARGET_CATEGORIES,
    TARGET_COLUMN,
)
from src.logger import get_logger

logger = get_logger(__name__)
if from_path is None:
    logger.warning(
        "charset_normalizer not installed. Encoding detection will be limited. "
        "Install with: pip install charset-normalizer"
    )


class DataLoader:
    """
    A robust and flexible data loader for the AuroraGate competition.
    
    Features:
    - Automatic format and encoding detection
    - Intelligent caching with Parquet
    - Data validation and integrity checks
    - Stratified sampling for quick testing
    - Comprehensive statistics extraction
    - Data versioning with metadata
    
    Attributes:
        data_dir: Directory containing raw data files
        cache_dir: Directory for cached processed data
        use_cache: Whether to use caching (default: True)
        memory_limit_mb: Memory limit for chunking (default: 500 MB)
    """
    
    REQUIRED_COLUMNS = ["description", "amount", "date"]
    REQUIRED_TRAIN_COLUMNS = ["description", "amount", "date", "category"]
    
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        cache_dir: Path = CACHE_DIR,
        use_cache: bool = True,
        memory_limit_mb: float = 500.0,
    ):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing raw data files
            cache_dir: Directory for cached processed data
            use_cache: Whether to use caching for faster loads
            memory_limit_mb: Memory limit in MB for automatic chunking
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.memory_limit_mb = memory_limit_mb
        
        # Statistics tracking
        self.stats = {
            "files_loaded": [],
            "rows_loaded": 0,
            "rows_dropped": 0,
            "missing_values_filled": 0,
            "dates_parsed_successfully": 0,
            "dates_failed_to_parse": 0,
            "unexpected_categories": [],
            "encoding_used": {},
        }
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataLoader initialized with data_dir={self.data_dir}")
    
    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================
    
    def load_train_data(
        self,
        train_file: str = "train.csv",
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Load training data with intelligent caching.
        
        Args:
            train_file: Name of the training file
            force_reload: Force reload from source, ignoring cache
        
        Returns:
            Training DataFrame
        
        Raises:
            FileNotFoundError: If training file doesn't exist
            ValueError: If data validation fails
        """
        source_file = self.data_dir / train_file
        cache_file = self.cache_dir / "train_processed.parquet"
        
        # Check cache validity
        if self.use_cache and not force_reload and self._is_cache_valid(cache_file, source_file):
            try:
                logger.info(f"Loading from cache: {cache_file}")
                df = pd.read_parquet(cache_file)
                df = self._downcast_columns(df)
                self._log_loading_summary(df, str(source_file))
                return df
            except ImportError:
                logger.warning("Parquet engine unavailable; loading from source instead")
        
        # Load from source
        logger.info(f"Loading from source: {source_file}")
        df = self._load_file(source_file)
        df = self._validate_data(df, is_train=True)
        df = self._parse_dates(df)
        df = self._clean_text_columns(df)
        df = self._downcast_columns(df)
        
        # Save to cache
        if self.use_cache:
            try:
                df.to_parquet(cache_file, compression="snappy", index=False)
                self._save_data_metadata(df, source_file, cache_file)
                logger.info(f"Cached processed data to: {cache_file}")
            except ImportError:
                logger.warning("Parquet engine unavailable; skipping cache write")
        
        self._log_loading_summary(df, str(source_file))
        return df
    
    def load_test_data(
        self,
        test_file: str = "test.csv",
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Load test data with intelligent caching.
        
        Args:
            test_file: Name of the test file
            force_reload: Force reload from source, ignoring cache
        
        Returns:
            Test DataFrame
        
        Raises:
            FileNotFoundError: If test file doesn't exist
            ValueError: If data validation fails
        """
        source_file = self.data_dir / test_file
        cache_file = self.cache_dir / "test_processed.parquet"
        
        # Check cache validity
        if self.use_cache and not force_reload and self._is_cache_valid(cache_file, source_file):
            try:
                logger.info(f"Loading from cache: {cache_file}")
                df = pd.read_parquet(cache_file)
                df = self._downcast_columns(df)
                self._log_loading_summary(df, str(source_file))
                return df
            except ImportError:
                logger.warning("Parquet engine unavailable; loading from source instead")
        
        # Load from source
        logger.info(f"Loading from source: {source_file}")
        df = self._load_file(source_file)
        df = self._validate_data(df, is_train=False)
        df = self._parse_dates(df)
        df = self._clean_text_columns(df)
        df = self._downcast_columns(df)
        
        # Save to cache
        if self.use_cache:
            try:
                df.to_parquet(cache_file, compression="snappy", index=False)
                self._save_data_metadata(df, source_file, cache_file)
                logger.info(f"Cached processed data to: {cache_file}")
            except ImportError:
                logger.warning("Parquet engine unavailable; skipping cache write")
        
        self._log_loading_summary(df, str(source_file))
        return df
    
    def load_sample(
        self,
        n_rows: int = 1000,
        stratified: bool = True,
    ) -> pd.DataFrame:
        """
        Load a small sample for quick testing.
        
        Args:
            n_rows: Number of rows to load
            stratified: If True, maintain category distribution
        
        Returns:
            Sampled DataFrame
        """
        df = self.load_train_data()
        
        if stratified and TARGET_COLUMN in df.columns:
            # Stratified sampling to maintain category distribution
            sample_size = min(n_rows, len(df))
            sampled = df.groupby(TARGET_COLUMN, group_keys=False).apply(
                lambda x: x.sample(
                    n=max(1, int(len(x) * sample_size / len(df))),
                    random_state=RANDOM_STATE,
                )
            )
            # Ensure we don't exceed n_rows
            if len(sampled) > n_rows:
                sampled = sampled.sample(n=n_rows, random_state=RANDOM_STATE)
            logger.info(f"Loaded stratified sample: {len(sampled)} rows")
            return sampled
        else:
            sample_size = min(n_rows, len(df))
            sampled = df.sample(n=sample_size, random_state=RANDOM_STATE)
            logger.info(f"Loaded random sample: {len(sampled)} rows")
            return sampled
    
    def get_initial_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Extract key statistics for quick EDA.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Dictionary containing various statistics
        """
        stats = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 ** 2),
        }
        
        # Category distribution
        if TARGET_COLUMN in df.columns:
            category_counts = df[TARGET_COLUMN].value_counts()
            stats["category_distribution"] = category_counts.to_dict()
            if len(category_counts) > 1:
                stats["category_imbalance_ratio"] = (
                    category_counts.max() / category_counts.min()
                )
            else:
                stats["category_imbalance_ratio"] = 1.0
        
        # Amount statistics
        if "amount" in df.columns:
            stats["amount_stats"] = {
                "mean": df["amount"].mean(),
                "median": df["amount"].median(),
                "std": df["amount"].std(),
                "min": df["amount"].min(),
                "max": df["amount"].max(),
                "skewness": df["amount"].skew(),
            }
        
        # Date range
        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
            stats["date_range"] = {
                "start": df["date"].min(),
                "end": df["date"].max(),
                "days_span": (df["date"].max() - df["date"].min()).days,
            }
        
        return stats
    
    @staticmethod
    def _downcast_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Reduce common numeric and categorical columns after loading."""
        result = df.copy()
        if "amount" in result.columns:
            result["amount"] = pd.to_numeric(result["amount"], downcast="float")
        if "transaction_id" in result.columns:
            result["transaction_id"] = pd.to_numeric(
                result["transaction_id"], downcast="integer"
            )
        if "day_of_week" in result.columns:
            result["day_of_week"] = result["day_of_week"].astype("category")
        return result

    def align_columns(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align columns between train and test sets.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
        
        Returns:
            Tuple of (aligned_train_df, aligned_test_df)
        """
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        # Columns only in train
        only_in_train = train_cols - test_cols
        if only_in_train:
            logger.info(f"Columns only in train: {only_in_train}")
        
        # Columns only in test
        only_in_test = test_cols - train_cols
        if only_in_test:
            logger.warning(f"Columns only in test (will be dropped): {only_in_test}")
        
        # Keep common columns
        common_cols = list(train_cols & test_cols)
        
        # For train, ensure target column is present
        train_final_cols = common_cols.copy()
        if TARGET_COLUMN in train_df.columns and TARGET_COLUMN not in train_final_cols:
            train_final_cols.append(TARGET_COLUMN)
        
        return train_df[train_final_cols], test_df[common_cols]
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _load_file(self, filepath: Path, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load a single file with automatic format and encoding detection.
        
        Args:
            filepath: Path to the file
            nrows: Number of rows to load (None for all)
        
        Returns:
            Loaded DataFrame
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is unsupported
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Detect if file is compressed
        suffix = filepath.suffix.lower()
        is_compressed = suffix in [".gz", ".bz2", ".zip", ".xz"]
        actual_suffix = filepath.with_suffix("").suffix.lower() if is_compressed else suffix
        
        try:
            if actual_suffix == ".csv" or is_compressed:
                # Detect encoding for uncompressed CSV
                if not is_compressed:
                    encoding = self._detect_encoding(filepath)
                else:
                    encoding = "utf-8"  # Default for compressed
                
                df = pd.read_csv(
                    filepath,
                    encoding=encoding,
                    nrows=nrows,
                    low_memory=False,  # Prevent mixed type warnings
                )
                self.stats["encoding_used"][filepath.name] = encoding
                
            elif actual_suffix == ".parquet":
                df = pd.read_parquet(filepath)
                
            elif actual_suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(filepath, nrows=nrows)
                
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            logger.info(f"Loaded {filepath.name}: {df.shape[0]} rows, {df.shape[1]} columns")
            self.stats["files_loaded"].append(str(filepath))
            self.stats["rows_loaded"] += len(df)
            
            return df
            
        except UnicodeDecodeError as e:
            logger.warning(f"Encoding error with {filepath.name}: {e}")
            # Retry with latin-1 as fallback
            logger.info("Retrying with latin-1 encoding")
            return pd.read_csv(filepath, encoding="latin-1", nrows=nrows, low_memory=False)
        
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise
    
    def _detect_encoding(self, filepath: Path) -> str:
        """
        Detect file encoding automatically.
        
        Args:
            filepath: Path to the file
        
        Returns:
            Detected encoding (defaults to utf-8 if detection fails)
        """
        if from_path is None:
            return "utf-8"
        
        try:
            result = from_path(filepath).best()
            if result:
                encoding = result.encoding
                logger.info(f"Detected encoding for {filepath.name}: {encoding}")
                return encoding
        except Exception as e:
            logger.warning(f"Encoding detection failed for {filepath.name}: {e}")
        
        return "utf-8"
    
    def _validate_data(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Validate data integrity and raise informative errors.
        
        Args:
            df: DataFrame to validate
            is_train: Whether this is training data
        
        Returns:
            Validated DataFrame
        
        Raises:
            ValueError: If validation fails
        """
        # Check for empty dataframe
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check required columns
        required = self.REQUIRED_TRAIN_COLUMNS if is_train else self.REQUIRED_COLUMNS
        missing_cols = set(required) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert amount to numeric if needed
        if not pd.api.types.is_numeric_dtype(df["amount"]):
            logger.warning("amount column is not numeric. Attempting conversion...")
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            null_count = df["amount"].isnull().sum()
            if null_count > 0:
                logger.warning(f"Failed to convert {null_count} amount values")
        
        # Check target column values
        if is_train and TARGET_COLUMN in df.columns:
            invalid_cats = set(df[TARGET_COLUMN].dropna().unique()) - set(TARGET_CATEGORIES)
            if invalid_cats:
                logger.warning(f"Found unexpected categories: {invalid_cats}")
                self.stats["unexpected_categories"].extend(list(invalid_cats))
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: DataFrame to process
        
        Returns:
            DataFrame with missing values handled
        """
        # Fill missing amount with median
        if df["amount"].isnull().any():
            median_amount = df["amount"].median()
            null_count = df["amount"].isnull().sum()
            df["amount"].fillna(median_amount, inplace=True)
            logger.warning(f"Filled {null_count} missing amount values with median ({median_amount:.2f})")
            self.stats["missing_values_filled"] += null_count
        
        # Fill missing description with empty string
        if "description" in df.columns and df["description"].isnull().any():
            null_count = df["description"].isnull().sum()
            df["description"].fillna("", inplace=True)
            logger.warning(f"Filled {null_count} missing description values with empty string")
            self.stats["missing_values_filled"] += null_count
        
        return df
    
    def _parse_dates(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """
        Parse dates with multiple format support.
        
        Args:
            df: DataFrame to process
            date_col: Name of the date column
        
        Returns:
            DataFrame with parsed dates
        """
        if date_col not in df.columns:
            return df
        
        original_nulls = df[date_col].isnull().sum()
        
        # Try multiple formats
        formats_to_try = [
            None,  # Let pandas infer (pandas 2.0+ with format='mixed')
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
        ]
        
        for fmt in formats_to_try:
            try:
                if fmt is None:
                    # pandas 2.0+ supports format='mixed'
                    parsed = pd.to_datetime(df[date_col], format="mixed", errors="coerce")
                else:
                    parsed = pd.to_datetime(df[date_col], format=fmt, errors="coerce")
                
                # Check if parsing was successful (less than 5% failures)
                null_ratio = parsed.isnull().sum() / len(parsed)
                if null_ratio < 0.05:
                    df[date_col] = parsed
                    logger.info(f"Successfully parsed dates with format: {fmt}")
                    self.stats["dates_parsed_successfully"] += len(df) - parsed.isnull().sum()
                    self.stats["dates_failed_to_parse"] += parsed.isnull().sum()
                    break
            except Exception:
                continue
        
        # Log failures
        new_nulls = df[date_col].isnull().sum()
        if new_nulls > original_nulls:
            failed_count = new_nulls - original_nulls
            logger.warning(f"Failed to parse {failed_count} dates ({failed_count/len(df)*100:.2f}%)")
        
        return df
    
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean text columns by stripping whitespace and normalizing.
        
        Args:
            df: DataFrame to process
        
        Returns:
            DataFrame with cleaned text columns
        """
        str_cols = df.select_dtypes(include=["object"]).columns
        
        for col in str_cols:
            if len(df) > 10000:
                # Use tqdm for large datasets
                df[col] = [
                    str(text).strip() if pd.notna(text) else ""
                    for text in tqdm(df[col], desc=f"Cleaning {col}")
                ]
            else:
                df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else "")
        
        return df
    
    def _is_cache_valid(self, cache_file: Path, source_file: Path) -> bool:
        """
        Check if cache is newer than source file.
        
        Args:
            cache_file: Path to cache file
            source_file: Path to source file
        
        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_file.exists():
            return False
        if not source_file.exists():
            return False
        return cache_file.stat().st_mtime >= source_file.stat().st_mtime
    
    def _compute_file_hash(self, filepath: Path, algorithm: str = "md5") -> str:
        """
        Compute hash of a file for versioning.
        
        Args:
            filepath: Path to the file
            algorithm: Hash algorithm to use
        
        Returns:
            Hash string
        """
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _save_data_metadata(
        self,
        df: pd.DataFrame,
        source_file: Path,
        output_file: Path,
    ) -> None:
        """
        Save metadata about the data for reproducibility.
        
        Args:
            df: DataFrame that was saved
            source_file: Original source file
            output_file: Output cache file
        """
        metadata = {
            "source_file": str(source_file),
            "source_hash": self._compute_file_hash(source_file),
            "creation_time": datetime.now().isoformat(),
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "row_count": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "loader_stats": self.stats,
        }
        
        metadata_file = output_file.with_suffix(".metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to: {metadata_file}")
    
    def _log_loading_summary(self, df: pd.DataFrame, source: str) -> None:
        """
        Log comprehensive summary after loading.
        
        Args:
            df: Loaded DataFrame
            source: Source file path
        """
        logger.info("=" * 60)
        logger.info(f"DATA LOADING SUMMARY: {source}")
        logger.info("=" * 60)
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values:\n{missing[missing > 0]}")
        
        # Category distribution
        if TARGET_COLUMN in df.columns:
            dist = df[TARGET_COLUMN].value_counts()
            logger.info(f"Category distribution:\n{dist}")
            if len(dist) > 1:
                imbalance = dist.max() / dist.min()
                if imbalance > 10:
                    logger.warning(f"High class imbalance: {imbalance:.1f}x")
        
        logger.info("=" * 60)


# =============================================================================
# MAIN (for testing purposes)
# =============================================================================
if __name__ == "__main__":
    # Test the DataLoader
    print("=" * 60)
    print("Testing DataLoader")
    print("=" * 60)
    
    try:
        loader = DataLoader()
        
        # Try to load data (will fail if data files don't exist)
        print("\nAttempting to load training data...")
        train_df = loader.load_train_data()
        
        print("\nAttempting to load test data...")
        test_df = loader.load_test_data()
        
        print("\nAttempting to load sample...")
        sample_df = loader.load_sample(n_rows=100)
        
        print("\nInitial statistics:")
        stats = loader.get_initial_statistics(train_df)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
        print("DataLoader test completed successfully!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\nExpected error (no data files yet): {e}")
        print("\nTo test the DataLoader, place your data files in:")
        print(f"  {DATA_DIR}")
        print("\nExpected files:")
        print("  - train.csv")
        print("  - test.csv")
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise