"""
main.py
=======
Workflow orchestration for the Next-Generation SOA Juvenile Mortality
Table pipeline.

All parameters are driven by ``config.yaml`` — no hard-coded paths or
constants here.  To adapt the pipeline to a new data source, update
``config.yaml`` and re-run this script.

Usage
-----
    python main.py                     # uses default config.yaml
    python main.py --config my.yaml    # alternative config file
"""

from __future__ import annotations

import argparse
import gc
import io
import logging
import sys
import warnings
from pathlib import Path

# Force UTF-8 console output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml

# ── Ensure project root is importable ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import JuvenileDataLoader
from src.actuarial_metrics import (
    compute_ae_ratios,
    ae_confidence_intervals,
    compare_juvenile_adult_pca,
)
from src.mortality_models import compute_crude_qx, run_feature_importance, fit_mortality_gam
from src.model_validation import validation_report

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("main")

SEP = "\n" + "=" * 70 + "\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from '%s'.", path)
    return cfg


def _ensure_dirs(cfg: dict) -> None:
    for key in ("tables_dir", "figures_dir"):
        p = ROOT / cfg["outputs"][key]
        p.mkdir(parents=True, exist_ok=True)


def _save_table(df: pd.DataFrame, name: str, cfg: dict) -> None:
    if cfg["outputs"].get("save_tables", True):
        dest = ROOT / cfg["outputs"]["tables_dir"] / f"{name}.csv"
        df.to_csv(dest, index=False)
        logger.info("Saved table → %s", dest.name)


# ---------------------------------------------------------------------------
# Module 1 — Data Ingestion
# ---------------------------------------------------------------------------

def run_module1(cfg: dict) -> pd.DataFrame:
    print(SEP + "MODULE 1 — Data Ingestion & Preprocessing")

    data_cfg = cfg["data"]
    source_path = Path(data_cfg["source_path"])
    cache_path  = ROOT / data_cfg["parquet_cache"]

    loader = JuvenileDataLoader(
        path=source_path,
        sep=data_cfg.get("sep", "\t"),
        parquet_cache=cache_path,
        verbose=True,
    )
    df = loader.load()
    print(loader.summary())
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Module 2 — Actuarial Metrics
# ---------------------------------------------------------------------------

def run_module2(df: pd.DataFrame, cfg: dict) -> dict:
    print(SEP + "MODULE 2 — Actuarial Metrics & EDA")

    m2 = cfg["ae_ratios"]
    results: dict = {}

    # 2a. A/E by primary groupby dimensions
    ae = compute_ae_ratios(df, groupby=m2["groupby"],
                           min_exposure_cnt=m2.get("min_exposure_cnt", 0.0))
    ae = ae_confidence_intervals(ae, ci_level=m2.get("ci_level", 0.95))
    print(f"\nA/E Ratios by {m2['groupby']}:")
    cols = [c for c in [
        *m2["groupby"], "Policies_Exposed", "Death_Count",
        "Exp_Count", "AE_Count", "AE_Count_CI_Lo", "AE_Count_CI_Hi",
        "Credibility_Count",
    ] if c in ae.columns]
    print(ae[cols].to_string(index=False))
    results["ae_primary"] = ae
    _save_table(ae, "ae_by_sex_age_group", cfg)

    # 2b. A/E by Attained Age (single dimension)
    ae_age = compute_ae_ratios(df, groupby=m2.get("age_groupby", ["Attained_Age"]))
    ae_age = ae_confidence_intervals(ae_age, ci_level=m2.get("ci_level", 0.95))
    print("\nA/E by Attained Age (first 20 ages):")
    print(ae_age[["Attained_Age", "Policies_Exposed", "Death_Count",
                  "Exp_Count", "AE_Count", "AE_Count_CI_Lo",
                  "AE_Count_CI_Hi"]].head(20).to_string(index=False))
    results["ae_age"] = ae_age
    _save_table(ae_age, "ae_by_attained_age", cfg)

    # 2c. PCA structural comparison (low vs high duration as proxy)
    pca_cfg = m2.get("pca", {})
    split   = pca_cfg.get("duration_split", 5)
    df_juv_proxy   = df[df["Duration"] <= split]
    df_adult_proxy = df[df["Duration"] >  split]

    print(f"\nPCA Structural Comparison (Duration ≤{split} vs >{split}):")
    if len(df_juv_proxy) > 10 and len(df_adult_proxy) > 10:
        pca_result = compare_juvenile_adult_pca(
            df_juv_proxy, df_adult_proxy,
            n_components=pca_cfg.get("n_components", 4),
        )
        print(pca_result["summary"])
        results["pca"] = pca_result
    else:
        print("  Insufficient rows for PCA comparison.")

    return results


# ---------------------------------------------------------------------------
# Module 3 — Modeling Engine
# ---------------------------------------------------------------------------

def run_module3(df: pd.DataFrame, cfg: dict) -> dict:
    print(SEP + "MODULE 3 — Mortality Models")
    results: dict = {}

    from config import ACTUAL_CNT_COL, EXPECTED_CNT_COL, EXPOSURE_CNT

    # 3a. Crude qx by Attained Age
    crude = compute_crude_qx(df, by="Attained_Age", basis="count")
    print("Crude qx by Attained Age (count basis):")
    print(crude.to_string(index=False))
    results["crude_qx"] = crude
    _save_table(crude, "crude_qx_by_age", cfg)

    # 3b. Feature Importance + SHAP
    fi_cfg = cfg["feature_importance"]
    fi_dims_present = [c for c in fi_cfg.get("feature_dims", []) if c in df.columns]

    df_agg_fi = (
        df.groupby(fi_dims_present, dropna=False, observed=True)
        .agg(
            Death_Count=(ACTUAL_CNT_COL, "sum"),
            ExpDth_VBT2015_Cnt=(EXPECTED_CNT_COL, "sum"),
            Policies_Exposed=(EXPOSURE_CNT, "sum"),
        )
        .reset_index()
    )
    df_agg_fi["AE_Count"] = (
        df_agg_fi["Death_Count"]
        / df_agg_fi["ExpDth_VBT2015_Cnt"].replace(0, float("nan"))
    )
    df_agg_fi = df_agg_fi.dropna(subset=["AE_Count"])
    print(f"\n\nFeature Importance + SHAP  (aggregated cells: {len(df_agg_fi):,})")

    fi_result = run_feature_importance(
        df_agg_fi,
        target=fi_cfg.get("target", "AE_Count"),
        feature_cols=fi_dims_present,
        model_type=fi_cfg.get("model_type", "gbm"),
        n_estimators=fi_cfg.get("n_estimators", 150),
        cv_folds=fi_cfg.get("cv_folds", 3),
        compute_shap=fi_cfg.get("compute_shap", True),
        shap_sample_size=fi_cfg.get("shap_sample_size", 500),
    )
    print(fi_result["summary"])
    results["feature_importance"] = fi_result

    # 3c. Univariate GAM (Attained Age only)
    gam_cfg = cfg["gam"]
    print(f"\n\nPoisson GAM — Univariate smooth on {gam_cfg['age_col']}:")
    df_gam = (
        df.groupby(gam_cfg["age_col"], observed=True)
        .agg(Death_Count=(ACTUAL_CNT_COL, "sum"),
             Policies_Exposed=(EXPOSURE_CNT, "sum"))
        .reset_index()
    )
    df_gam = df_gam[df_gam["Policies_Exposed"] > 0]

    gam_result = fit_mortality_gam(
        df_gam,
        age_col=gam_cfg["age_col"],
        basis=gam_cfg.get("basis", "count"),
        tensor_terms=gam_cfg.get("tensor_terms"),
        n_splines=gam_cfg.get("n_splines", 15),
    )
    print(f"  AIC  : {gam_result['aic']:.2f}")
    print(f"  UBRE : {gam_result['gcv']:.4f}")
    print(f"  λ    : {gam_result['lam']}")
    print("\nSmoothed qx table (all ages):")
    print(gam_result["smooth_qx"].to_string(index=False))
    results["gam"] = gam_result
    _save_table(gam_result["smooth_qx"], "smooth_qx_univariate", cfg)

    # 3d. Tensor-product GAM (Age × Duration)
    te_cfg = gam_cfg.get("tensor", {})
    print("\n\nPoisson GAM — Tensor smooth Age × Duration:")
    df_gam_te = (
        df.groupby([gam_cfg["age_col"], "Duration"], observed=True)
        .agg(Death_Count=(ACTUAL_CNT_COL, "sum"),
             Policies_Exposed=(EXPOSURE_CNT, "sum"))
        .reset_index()
    )
    df_gam_te = df_gam_te[df_gam_te["Policies_Exposed"] > 0]

    gam_te = fit_mortality_gam(
        df_gam_te,
        age_col=gam_cfg["age_col"],
        basis=gam_cfg.get("basis", "count"),
        tensor_terms=te_cfg.get("tensor_terms", ["Duration"]),
        n_splines=te_cfg.get("n_splines", 10),
    )
    print(f"  AIC  : {gam_te['aic']:.2f}")
    print(f"  UBRE : {gam_te['gcv']:.4f}")
    print(f"  λ    : {gam_te['lam']}")
    results["gam_tensor"] = gam_te

    return results


# ---------------------------------------------------------------------------
# Module 4 — Validation & Guardrails
# ---------------------------------------------------------------------------

def run_module4(mod3_results: dict, cfg: dict) -> dict:
    print(SEP + "MODULE 4 — Validation & Guardrails")

    val_cfg = cfg["validation"]
    smooth_qx_df = mod3_results["gam"]["smooth_qx"]

    # Build synthetic adult reference (log-linear extension)
    last_age    = int(smooth_qx_df["Attained_Age"].max())
    adult_ages  = np.arange(last_age, last_age + val_cfg.get("adult_projection_ages", 20),
                            dtype=float)
    last_qx     = float(smooth_qx_df["qx_Smooth"].iloc[-1])
    growth_rate = val_cfg.get("adult_growth_rate", 0.085)
    adult_ref   = pd.DataFrame({
        "Attained_Age": adult_ages,
        "qx_Smooth": last_qx * np.exp(growth_rate * (adult_ages - last_age)),
    })

    join_age   = val_cfg.get("join_age", 25)
    blend_ages = val_cfg.get("blend_ages", 5)

    val = validation_report(
        smooth_qx=smooth_qx_df,
        adult_qx=adult_ref,
        age_col="Attained_Age",
        qx_col="qx_Smooth",
        join_age=join_age,
        blend_ages=blend_ages,
    )
    print(val["summary"])
    print("\nGraded Table (full age range):")
    print(val["graded_table"][
        ["Attained_Age", "qx_Juvenile", "qx_Adult", "weight_Adult",
         "qx_Blended", "source"]
    ].to_string(index=False))

    _save_table(val["graded_table"], "graded_qx_table", cfg)
    return val


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SOA Juvenile Mortality Pipeline")
    parser.add_argument(
        "--config", default=str(ROOT / "config.yaml"),
        help="Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    _ensure_dirs(cfg)

    # Run modules sequentially
    df       = run_module1(cfg)
    mod2     = run_module2(df, cfg)
    mod3     = run_module3(df, cfg)
    mod4     = run_module4(mod3, cfg)

    print(SEP + "Pipeline complete.")


if __name__ == "__main__":
    main()
