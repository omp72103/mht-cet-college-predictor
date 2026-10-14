# AGENTS.md — MHT-CET College Predictor

## What this project is
A pure Python ML project (no web frontend). Trains an ensemble of RandomForest + HistGradientBoosting models to predict MHT-CET college admission cutoffs, with uncertainty calibration. Outputs a `bundle.joblib` model, `metrics.json`, and figures in `docs/figures/`.

## Running it
- **Docker (Base44 sandbox):** `docker compose -f docker-compose.base44.yml up -d` — starts JupyterLab on port 3000. The startup command installs deps, trains the model if `models/bundle.joblib` doesn't exist, then launches JupyterLab (no token, XSRF disabled for preview embedding).
- **Local (no Docker):** `pip install -r requirements-dev.txt && python -m src.train` (~2 min). Add `--robustness` for 5-fold CV / unseen-college / temporal splits (~15 min).
- **Tests:** `pytest -q` (8 tests, ~2s).

## Key details
- The model bundle is git-ignored (`models/*.joblib`). It's regenerated from `data/combined_cutoffs.csv` on first startup.
- `src/model.py` classes are referenced by joblib pickles — keep module paths stable (`src.*`).
- `sml_rank` is deliberately excluded as a feature (target leakage, correlation −0.98).
- The Colab notebook at `notebooks/MHT_CET_College_Predictor_Colab.ipynb` runs the same pipeline end-to-end.
- No external credentials or secrets are needed — all data is in `data/combined_cutoffs.csv`.
