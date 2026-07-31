# AuroraGate Expense Categorization Challenge

**Automated classification of bank transactions into ten expense categories**

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-AuroraGate-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/aurora-gate-expense-categorization-challenge)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost-FFCD00)](https://catboost.ai/)

## Table of Contents

- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [The Story Behind the Project](#the-story-behind-the-project)
- [Technical Approach](#technical-approach)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Reproducibility and Generated Artifacts](#reproducibility-and-generated-artifacts)
- [Lessons Learned](#lessons-learned)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Project Overview

AuroraGate is a complete machine-learning solution for the **AuroraGate Expense Categorization Challenge** on Kaggle. The task is to classify bank transactions into ten expense categories using:

- Transaction descriptions
- Transaction amounts
- Transaction dates
- Day-of-week information
- Derived text, merchant, amount, calendar, and transaction-order features

The dataset contains **8,400 labeled training transactions** and **3,600 unlabeled test transactions**. The competition metric is **Macro F1**, which gives every category equal importance and is therefore more informative than accuracy for this imbalanced multi-class problem.

### Target Categories

`Food & Dining` • `Groceries` • `Transportation` • `Entertainment` • `Shopping` • `Bills & Utilities` • `Health & Fitness` • `Miscellaneous` • `Subscriptions` • `Travel`

## Key Results

| Metric | Result |
|:---|:---:|
| **Best Public LB score** | **0.90169** |
| **Validation Macro F1** | **0.9111** |
| **Primary model** | **CatBoost Classifier** |
| **Validation design** | Five-fold chronological `TimeSeriesSplit` |
| **Final submission strategy** | CatBoost-only, selected by validation gating |

The public leaderboard score of `0.90169` is the best confirmed submission result. The local validation score is higher, as expected, because the validation split and the hidden leaderboard set are different samples.

> **Important:** The final pipeline evaluates several candidates, but it does not blindly submit the most complex model. If an ensemble, threshold adjustment, or post-processing rule does not beat the CatBoost baseline on validation, the validation gate rejects it and keeps CatBoost-only predictions.

## The Story Behind the Project

This project evolved through controlled experiments rather than a single modeling attempt. The most valuable outcome was not only the final score, but the validation process used to distinguish useful features from changes that looked promising locally but were unsafe for submission.

### Phase 1: Foundation

- Performed exploratory analysis of category balance, descriptions, amounts, and time patterns.
- Added robust CSV loading with encoding detection and date parsing.
- Added memory downcasting for numeric columns and categorical conversion where appropriate.
- Established a reproducible project configuration and centralized logging module.

### Phase 2: Feature Engineering

The feature pipeline combines:

- Character TF-IDF n-grams (`3-5`) and word TF-IDF representations.
- Truncated SVD text components.
- Merchant extraction through normalized store names.
- Store transaction frequency.
- Repeated store-and-amount indicators.
- Amount transformations such as `log_amount`, amount bins, decimal precision, and monthly percentile.
- Calendar features including month, quarter, day of month, week of year, weekend indicators, and days to weekend.
- Transaction-order features such as time since the previous transaction and transaction number within a day.
- Description statistics and domain keyword indicators.

### Phase 3: Baseline and Controlled Experiments

The strongest stable configuration is CatBoost with the engineered structured and text features. Several alternatives were evaluated:

| Experiment | Validation result | Decision |
|:---|:---:|:---|
| **CatBoost baseline** | **0.9111** | **Selected baseline** |
| CatBoost + LightGBM weighted blend | Approximately 0.9110 | Rejected: no reliable gain |
| CatBoost + Logistic Regression | Approximately 0.9110 | Rejected by the validation gate |
| Confidence-gated keyword rules | Below baseline | Rejected: rules hurt Macro F1 |
| Per-class threshold tuning | Below baseline | Rejected: gains in weak classes did not offset other losses |
| Target-encoding experiments | Unstable across configurations | Disabled in the final path |

The repository still contains the target-encoding helper as an experiment and extension point, but the call is intentionally disabled in `engineer_features()`. This prevents the final baseline from depending on a feature that caused a severe submission regression in an earlier configuration.

### Phase 4: Validation Gate

The final pipeline uses a validation gate that:

1. Sorts transactions by `transaction_id`.
2. Generates chronological out-of-fold predictions using five `TimeSeriesSplit` folds.
3. Measures CatBoost, ensemble, threshold, and rule-based candidates with Macro F1.
4. Compares every candidate against the CatBoost baseline.
5. Selects CatBoost-only when an alternative does not provide a real validation improvement.

This conservative design prevented local overfitting and protected the best known leaderboard submission from experimental regressions.

## Technical Approach

### Data Loading

`src/data_loader.py` provides:

- Automatic source-file encoding detection.
- Date parsing and validation.
- Optional local caching.
- Numeric downcasting for lower memory usage.
- Dataset shape, memory, and category-distribution summaries.

The expected files are:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

The data directory is intentionally ignored by Git. Download the competition data from Kaggle and place the files there locally.

### Feature Engineering

The main implementation is in [src/feature_engineering.py](src/feature_engineering.py). Important stable features include:

| Feature family | Examples | Purpose |
|:---|:---|:---|
| Merchant | `store_name`, `store_frequency` | Capture recurring merchant behavior |
| Text | TF-IDF and `tfidf_svd_*` columns | Represent noisy transaction descriptions |
| Amount | `amount`, `log_amount`, `amount_bins`, `amount_percentile` | Capture spending scale and relative amount |
| Calendar | `month`, `quarter`, `day_of_year`, `is_weekend` | Capture temporal patterns |
| Transaction order | `time_since_last_transaction`, `transaction_count_per_day` | Preserve chronological context |
| Description statistics | word, character, digit, and punctuation counts | Capture formatting and merchant patterns |
| Keyword indicators | grocery, food, transport, transfer, and travel flags | Add interpretable domain signals |

### Target Encoding Status

`add_target_encoding()` and `fit_target_encoding_stats()` remain available for future experiments, including K-fold out-of-fold encoding. However, target encoding is **not applied by the final `engineer_features()` path**. The production submission path intentionally uses the previously validated feature set after target-encoding variants produced unstable or degraded submission behavior.

### Model

The final baseline uses CatBoost with:

```python
{
    "iterations": 500,
    "learning_rate": 0.1,
    "depth": 6,
    "random_seed": 42,
    "verbose": False,
}
```

CatBoost is a strong fit for this problem because it handles mixed numeric and categorical features, supports multi-class classification, and works well with merchant and amount signals.

### Validation and Diagnostics

Validation is chronological rather than randomly shuffled:

```python
TimeSeriesSplit(n_splits=5)
```

The fifth fold is closest to the test period and is used for a dedicated confusion-matrix diagnostic. When the pipeline is run locally, it writes:

```text
experiments/fold5_confusion_matrix.png
```

The `experiments/` directory is ignored by Git, so this image is a local generated artifact and is not guaranteed to render from the public repository README. To showcase it publicly, copy it into a deliberately tracked documentation-assets directory before publishing.

## Repository Structure

```text
aurora-gate/
├── src/
│   ├── __init__.py
│   ├── config.py               # Paths, categories, model parameters, and settings
│   ├── data_loader.py          # Data loading, parsing, caching, and summaries
│   ├── eda.py                  # Exploratory data analysis
│   ├── error_analysis.py       # Out-of-fold error analysis and rule suggestions
│   ├── feature_engineering.py  # Merchant, text, amount, calendar, and transaction features
│   ├── final_pipeline.py      # Validation-gated submission pipeline
│   ├── kaggle_compatibility.py # Optional Kaggle/runtime compatibility helpers
│   ├── logger.py               # Centralized console and file logging
│   ├── train_lightgbm.py       # LightGBM experiment and comparison pipeline
│   ├── train_model.py          # CatBoost training and artifact generation
│   └── tune_hyperparams.py     # Optuna-based hyperparameter experiments
├── data/                       # Local competition data; ignored by Git
├── models/                     # Local trained artifacts; ignored by Git
├── experiments/                # Local logs and reports; ignored by Git
├── submissions/                # Local generated submissions; ignored by Git
├── requirements.txt            # Python dependency version floors
├── .gitignore                  # Data, artifacts, caches, and environment rules
└── README.md                   # Project documentation
```

`kaggle_compatibility.py` is shown as an optional module because the project can be extended with it if the Kaggle runtime requires compatibility handling. The current repository contains the core modules listed in `src/`.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/AliAziziDH/aurora-gate.git
cd aurora-gate
```

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Download the competition data

1. Open the [AuroraGate Kaggle competition page](https://www.kaggle.com/competitions/aurora-gate-expense-categorization-challenge).
2. Download the training, test, and sample-submission files.
3. Place them under `data/`.

The repository intentionally does not include raw data, trained models, generated submissions, caches, or experiment logs.

## Usage

Run commands from the repository root.

### Recommended end-to-end workflow

```bash
# Validate loading, parsing, dtypes, and category distributions
python -m src.data_loader

# Run exploratory data analysis
python -m src.eda

# Train the CatBoost baseline and create preprocessing artifacts
python -m src.train_model

# Run the final chronological validation and submission pipeline
python -m src.final_pipeline
```

The final pipeline writes the generated submission to:

```text
submissions/submission_final.csv
```

### Optional experiments

```bash
# Analyze out-of-fold errors
python -m src.error_analysis

# Run Optuna hyperparameter experiments
python -m src.tune_hyperparams

# Train and evaluate the LightGBM comparison pipeline
python -m src.train_lightgbm

# Initialize or verify centralized logging
python -m src.logger
```

The optional experiment modules may create files under `models/` and `experiments/`. These outputs are local and ignored by Git.

## Reproducibility and Generated Artifacts

The project is reproducible from source, dependencies, and the Kaggle input files. A fresh run may create:

```text
cache/                  # Optional data cache
models/                 # CatBoost, vectorizer, and experiment artifacts
experiments/            # Logs, JSON summaries, and diagnostics
submissions/            # Generated CSV submissions
```

These directories are excluded from the public repository because they can be large, machine-specific, or derived from private competition data. Re-running the documented commands recreates them locally.

## Lessons Learned

### What worked

- CatBoost provided the strongest stable baseline.
- Merchant extraction and store frequency were high-value features.
- Chronological validation was more trustworthy than a random split.
- Centralized logging made long-running experiments easier to inspect.
- Validation gating prevented experimental candidates from replacing a stronger baseline.

### What did not work reliably

- Hard-coded keyword rules overrode correct model predictions.
- LightGBM and Logistic Regression added diversity but did not beat CatBoost consistently.
- Per-class threshold tuning improved some weak classes while reducing overall Macro F1.
- Target-encoding variants were sensitive to distribution and artifact alignment; the final path keeps them disabled until they can be validated against a representative chronological split.

### Final takeaway

> The simplest model that survives strict validation is often more valuable than a more complex model with a higher apparent local score.

## Future Improvements

Potential next steps include:

1. Publishing the Fold 5 confusion matrix under a tracked documentation-assets directory.
2. Adding a dedicated experiment registry so model artifacts record their exact feature schema and data version.
3. Improving merchant normalization for variants such as `WMT`, `WALMART`, and numbered store identifiers.
4. Testing FastText or other subword embeddings for noisy descriptions and merchant typos.
5. Adding calibrated probability analysis for weak categories.
6. Packaging the classifier behind a small FastAPI or Streamlit interface.
7. Adding automated tests for feature schemas, chronological splits, and submission-column validation.

## Author

**Ali Azizi Deh Sorkh**
Industrial Engineer | Data Science and Optimization Enthusiast

- **GitHub:** [AliAziziDH](https://github.com/AliAziziDH)
- **Kaggle:** [aliazizi1](https://www.kaggle.com/aliazizi1)
- **Email:** aliazizi.academy@gmail.com

## Repository Link

https://github.com/AliAziziDH/aurora-gate

## License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

*Built with ❤️, rigorous validation, and a lot of trial and error.*
