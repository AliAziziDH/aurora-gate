# AuroraGate Expense Categorization

A machine learning project for categorizing financial transactions into expense categories.

## Project Structure

```
src/
├── config.py                # Configuration and hyperparameters
├── data_loader.py           # Data loading and preprocessing
├── feature_engineering.py   # Feature extraction and engineering
├── training_utils.py        # Reusable training utilities
├── train_model.py           # CatBoost model training
├── train_lightgbm.py        # LightGBM model training and ensemble
├── tune_hyperparams.py      # Hyperparameter optimization
├── final_pipeline.py        # Final model selection and submission
├── eda.py                   # Exploratory data analysis
├── error_analysis.py        # Model error analysis
└── logger.py                # Logging configuration

tests/
├── test_features.py         # Unit tests for feature engineering
└── test_imputation.py       # Unit tests for imputation functions
```

## Setup

### Requirements

- Python 3.8+
- Required packages in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/auroragate.git
cd auroragate

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

Place your data files in the `data/` directory:
- `train.csv` - Training data with transaction records and categories
- `test.csv` - Test data for prediction

## Training Models

### Train CatBoost Model

```bash
python -m src.train_model
```

### Train LightGBM and Ensemble

```bash
python -m src.train_lightgbm
```

### Hyperparameter Tuning

```bash
python -m src.tune_hyperparams
```

## Running Tests

### Unit Tests

The project includes unit tests for feature engineering and imputation:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_features.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Structure

- `tests/test_features.py`: Tests feature engineering functions
- `tests/test_imputation.py`: Tests data imputation functions

### Adding New Tests

When adding new functionality, please add corresponding unit tests. Tests should:

1. Focus on individual functions
2. Use small, synthetic datasets
3. Test edge cases (missing values, empty inputs)
4. Be fast-running (< 1 second per test)

## Running EDA

```bash
python -m src.eda
```

This will generate:
- Statistical summaries in `experiments/eda_summary.json`
- Visualizations in `experiments/figures/`

## Error Analysis

```bash
python -m src.error_analysis
```

Generates detailed error analysis with:
- Confusion matrices
- Per-class metrics
- Error pattern identification
- Rule suggestions for post-processing

## Final Pipeline

```bash
python -m src.final_pipeline
```

This runs the complete pipeline:
1. Selects best model (CatBoost vs. Ensemble)
2. Applies post-processing rules
3. Generates final submission file in `submissions/`

## Configuration

Edit `src/config.py` to modify:
- File paths
- Hyperparameters
- Training settings
- Text processing parameters

## Model Artifacts

Trained models and artifacts are saved in:
- `models/` - Serialized models and vectorizers
- `experiments/` - Training summaries and analysis
- `submissions/` - Prediction files

## Logging

Logs are saved in `experiments/` with timestamps. Use:

```python
from src.logger import get_logger
logger = get_logger(__name__)
```

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Keep lines under 100 characters
- Write comprehensive docstrings
- Add unit tests for new functionality

### Adding New Features

1. Add configuration to `config.py`
2. Implement in appropriate module
3. Add unit tests
4. Update documentation

## License

[MIT License](LICENSE)