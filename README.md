# CricketIQ

CricketIQ analyzes Virat Kohli's IPL ball-by-ball batting and develops predictions for delivery outcomes. The planned user-facing predictor estimates the chance of a six and a dot ball from the bowler, innings, over, venue, and opponent.

## Data flow

Keep source files unchanged in `Data/raw/`. Clean each source into a separate file in `Data/processed/cleaned/`. Compile validated sources into the canonical dataset in `Data/processed/master/`. Derive model inputs and labels under `Data/ml/features/` and `Data/ml/targets/`.

```text
Data/
├── raw/                         # Original source files
├── processed/
│   ├── cleaned/                 # Cleaned source datasets
│   └── master/                  # Joined canonical delivery dataset
└── ml/
    ├── features/                # Model input columns and splits
    └── targets/                 # Outcome labels
```

## Current datasets

- `Data/raw/KOhli_delivery_Combined_2009-2026.csv` — source ball-by-ball delivery data.
- `Data/raw/Virat_Ball_Faced_2008-2026.xlsx` — additional source workbook; its role in the pipeline needs confirmation.
- `Data/processed/cleaned/kohli_ipl_with_match_metadata.csv` — delivery data enriched with match metadata.
- `Data/processed/master/CricketIQ_master_v1.csv` — canonical dataset used by analysis and feature generation.
- `Data/ml/features/` and `Data/ml/targets/` — existing training/test exports.

## Code map

- `src/Config.py` — shared paths and data locations.
- `src/data/clean_match_metadata.py` — downloads Cricsheet metadata and enriches raw delivery records into `Data/processed/cleaned/`.
- `src/Finalize_dataset.py` — validates and compiles the master dataset.
- `src/feature_engineering.py` — builds point-in-time delivery features.
- `src/ml_problem.py` — creates prediction labels and chronological splits.
- `src/eda.py` and `src/hypothesis_testing.py` — analysis and research reports.
- `src/train_baseline.py`, `src/finalize_ml.py`, `src/final_test.py` — baseline evaluation, model selection, and final evaluation.
- `src/validation/master.py` — master data and cricket-rule checks.
- `src/validation/features.py` and `src/test_feature_leakage.py` — engineered-feature and leakage audits.
- `main.py` — command-line predictor for six and dot-ball probabilities. Run with `python main.py`.

## Setup

Install the pinned dependencies listed in `requirements.md`. Run `python main.py` from the project root; it fits its models from the master dataset at startup.

## Important status

No exact duplicate files were found; the repeated master-data checks were consolidated into one validation module. Similar data files were retained because they represent different pipeline stages.
