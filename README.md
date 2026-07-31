# AuroraGate

## Overview

AuroraGate is a machine-learning solution for the Aurora Gate expense categorization challenge. It loads transaction data, explores the dataset, engineers text and structured features, trains gradient-boosted models, and produces a competition submission.

## Results

- Macro F1: **0.9111**
- Best single model: **CatBoost**

## Setup and Installation

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/AliAziziDH/aurora-gate.git
cd aurora-gate
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The expected input files are in `data/`: `train.csv`, `test.csv`, and `sample_submission.csv`.

## Usage

Run the workflow from the repository root:

```bash
# Inspect and validate the input data
python -m src.data_loader

# Run exploratory data analysis and save the report
python -m src.eda

# Tune CatBoost and LightGBM hyperparameters
python -m src.tune_hyperparams

# Train the CatBoost model and save validation artifacts
python -m src.train_model

# Train the LightGBM model and evaluate the ensemble
python -m src.train_lightgbm

# Analyze out-of-fold errors
python -m src.error_analysis

# Build the final competition submission
python -m src.final_pipeline
```

The generated models and reports are written to `models/` and `experiments/`. The final submission is written to `submissions/submission_final.csv`.

## Project Structure

```text
aurora-gate/
├── data/                 # Competition datasets and sample submission
├── experiments/          # EDA, tuning, ensemble, and error reports
├── models/               # Trained models and preprocessing artifacts
├── notebooks/            # Exploratory notebooks
├── src/                  # Data, feature, training, and evaluation code
├── submissions/          # Generated competition submissions
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Key Features

- Reproducible data loading and validation
- Exploratory analysis with saved summaries and visualizations
- Text and structured feature engineering
- CatBoost and LightGBM model training
- Optuna-based hyperparameter tuning
- Out-of-fold error analysis
- Threshold optimization and ensemble submission generation

## Repository Link

https://github.com/AliAziziDH/aurora-gate
