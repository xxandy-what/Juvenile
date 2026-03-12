# Next-Generation SOA Juvenile Mortality Table — Predictive Analytics Framework

A modular, configuration-driven Python pipeline for experience analysis and
predictive modeling of juvenile life insurance mortality data (ILEC 2012–2019,
issue ages 0–17), following the SOA's 2024 "A Predictive Analytics Framework"
best-practice guidelines.

---

## Project Architecture

```
juvenile_mortality/
├── config.yaml                  # All parameters — edit here, not in source
├── config.py                    # Python constants mirroring config.yaml
├── main.py                      # Orchestration: runs all 4 modules sequentially
├── trial_run.py                 # Quick end-to-end smoke test
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Module 1 — Ingestion & Preprocessing
│   ├── actuarial_metrics.py     # Module 2 — A/E Ratios, CIs, PCA
│   ├── mortality_models.py      # Module 3 — Crude qx, GBM+SHAP, GAM
│   └── model_validation.py     # Module 4 — Monotonicity, Grading
│
├── data/
│   └── juvenile_cache.parquet  # Auto-generated Parquet cache (first run only)
│
└── outputs/
    ├── tables/                  # CSV exports of key result tables
    └── figures/                 # Saved plots (notebooks / dashboard)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `shap` requires a C compiler on some platforms.
> If installation fails, install without it and set `compute_shap: false`
> in `config.yaml`.

### 2. Configure your data source

Edit `config.yaml`:

```yaml
data:
  source_path: "/path/to/your/Juvenile_cleaned.txt"
  parquet_cache: "data/juvenile_cache.parquet"
```

All other parameters (join age, GAM splines, A/E groupings, SHAP settings)
are also controlled from `config.yaml`.

### 3. Run the full pipeline

```bash
python main.py
# or with a custom config:
python main.py --config path/to/my_config.yaml
```

### 4. Quick smoke test (all 4 modules, minimal output)

```bash
python trial_run.py
```

---

## Module Overview

### Module 1 — Data Ingestion & Preprocessing (`src/data_loader.py`)

- Loads TSV / CSV / Parquet source files via `JuvenileDataLoader`.
- **Parquet caching**: on first run the processed DataFrame is written to
  `data/juvenile_cache.parquet`; subsequent runs load from cache (~10× faster)
  and the raw CSV is released from memory immediately.
- Study-scope exclusions: `Attained_Age == 0`, `SOA_Post_Lvl_Ind == "PLT"`,
  `Issue_Age > 17`.
- ANB → ALB age conversion (`Age_ALB = Age_ANB − 0.5`).
- Feature engineering: quinquennial age bands, face-amount ordinals, duration
  groups, sex standardisation, log-exposure offsets.

### Module 2 — Actuarial Metrics & EDA (`src/actuarial_metrics.py`)

- `compute_ae_ratios()` — A/E ratios by policy count and face amount.
- `ae_confidence_intervals()` — 95 % Poisson normal-approximation CIs;
  LLID limited-fluctuation credibility factors (full credibility = 1,082 deaths).
- `compare_juvenile_adult_pca()` — PCA centroid distance and Bhattacharyya
  coefficient to quantify structural differences between cohorts.

### Module 3 — Modeling Engine (`src/mortality_models.py`)

- `compute_crude_qx()` — Exposure-weighted crude mortality rates.
- `run_feature_importance()` — Gradient Boosting (or Random Forest) with:
  - sklearn `feature_importances_` (MDI / gain)
  - **SHAP TreeExplainer** values for explicit, additive feature attribution
    (toggle with `compute_shap` in `config.yaml`).
- `fit_mortality_gam()` — `pyGAM` PoissonGAM with:
  - **ALB predictor**: `Attained_Age_mod = Attained_Age − 0.5` (Age Last
    Birthday, fractional values 0.5, 1.5, …) is used as the age predictor
    instead of the raw ANB integer age.
  - **Sex-stratified fitting**: both univariate and tensor-product GAMs are
    fit separately for Male and Female; results stored in `gam_results["M"]`
    / `gam_results["F"]`.
  - Univariate B-spline smooth `s(Attained_Age_mod)`.
  - Optional tensor-product smooth `te(Attained_Age_mod, Duration)` for
    VBT-style interaction modeling.
  - UBRE criterion (correct for known-scale Poisson models).

### Module 4 — Validation & Guardrails (`src/model_validation.py`)

- `check_monotonicity()` — Flags non-increasing qx outside the juvenile-dip
  region (ages 10–14), with configurable tolerance.
- `grade_into_adult_table()` — Log-linear blending of the juvenile table into
  the adult reference at a configurable join age.  The notebook uses the
  **2019 US Population Life Table** (`data/2019_US_LifeTable.txt`) as the
  adult reference, graded separately for M and F.  Life table integer ALB
  ages are shifted +0.5 to align with the `Attained_Age_mod` scale; default
  join age is ALB 70 (mod 70.5) with full transition to adult rates by
  ALB 85 (mod 85.5).
- `validation_report()` — Consolidated validation summary with table export.

---

## Adapting to a New Data Source

1. Update `data.source_path` in `config.yaml`.
2. If column names differ from the ILEC layout, update the constant maps
   in `config.py` (`ACTUAL_CNT_COL`, `EXPECTED_CNT_COL`, etc.).
3. Delete the old Parquet cache (`data/juvenile_cache.parquet`) so it is
   regenerated from the new source.
4. Re-run `python main.py`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy`, `pyarrow` | Data manipulation and Parquet I/O |
| `pyyaml` | Configuration file parsing |
| `scipy`, `statsmodels` | Statistical tests and distributions |
| `scikit-learn` | GBM / RF models, PCA, cross-validation |
| `shap` | SHAP feature attribution (TreeExplainer) |
| `pygam` | PoissonGAM for smoothed mortality curves |
| `matplotlib`, `seaborn`, `plotly` | Visualisation |
| `streamlit` | Interactive dashboard |

---

## References

- SOA (2024). *A Predictive Analytics Framework for Experience Studies.*
- SOA (2022). *2015 VBT Primary Tables.*
- ILEC (2021). *2012–2019 Individual Life Experience Study.*
- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R.*
- Lundberg, S. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model
  Predictions (SHAP).* NeurIPS 2017.
