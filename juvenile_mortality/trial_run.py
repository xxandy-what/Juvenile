"""
trial_run.py
============
Quick end-to-end trial of all four pipeline modules.
Run from the juvenile_mortality/ directory:
    python trial_run.py
"""

import sys
import io
import logging
import warnings

# Force UTF-8 output on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s",
                    stream=sys.stderr)

sys.path.insert(0, ".")   # ensure src/ and config.py are importable

import numpy as np
import pandas as pd

from config import DEFAULT_DATA_PATH
from src.data_loader      import JuvenileDataLoader
from src.actuarial_metrics import compute_ae_ratios, ae_confidence_intervals, compare_juvenile_adult_pca
from src.mortality_models  import compute_crude_qx, run_feature_importance, fit_mortality_gam
from src.model_validation  import validation_report

SEP = "\n" + "=" * 65 + "\n"

# ─────────────────────────────────────────────────────────────────
# MODULE 1 — Data Ingestion
# ─────────────────────────────────────────────────────────────────
print(SEP + "MODULE 1 — Data Ingestion & Preprocessing")

loader = JuvenileDataLoader(DEFAULT_DATA_PATH, verbose=True)
df = loader.load()
print(loader.summary())
print("\nSample (5 rows):")
print(df[["Attained_Age", "Attained_Age_ALB", "Issue_Age", "Issue_Age_ALB",
          "Sex_Clean", "Attained_Age_Group", "Policy_Year_Group",
          "Death_Count", "Policies_Exposed"]].head())
print(f"\nDtypes snapshot:\n{df.dtypes.to_string()}")

# ─────────────────────────────────────────────────────────────────
# MODULE 2 — A/E Ratios + CIs
# ─────────────────────────────────────────────────────────────────
print(SEP + "MODULE 2 — Actuarial Metrics")

# 2a. A/E by Sex × Attained Age Group
ae = compute_ae_ratios(df, groupby=["Sex_Clean", "Attained_Age_Group"])
ae = ae_confidence_intervals(ae)
print("\nA/E Ratios by Sex × Age Group:")
cols = ["Sex_Clean", "Attained_Age_Group", "Policies_Exposed", "Death_Count",
        "Exp_Count", "AE_Count", "AE_Count_CI_Lo", "AE_Count_CI_Hi", "Credibility_Count"]
print(ae[cols].to_string(index=False))

# 2b. A/E by Attained Age (single dimension)
ae_age = compute_ae_ratios(df, groupby=["Attained_Age"])
ae_age = ae_confidence_intervals(ae_age)
print("\n\nA/E by Attained Age (count basis, first 20 ages):")
print(ae_age[["Attained_Age", "Policies_Exposed", "Death_Count",
              "Exp_Count", "AE_Count", "AE_Count_CI_Lo", "AE_Count_CI_Hi"]].head(20).to_string(index=False))

# 2c. PCA structural comparison (Juvenile vs simulated adult — using Duration split as proxy)
print("\n\nPCA Structural Comparison (low vs high duration as proxy):")
df_juv_proxy   = df[df["Duration"] <= 5]
df_adult_proxy = df[df["Duration"] >  5]
if len(df_juv_proxy) > 10 and len(df_adult_proxy) > 10:
    pca_result = compare_juvenile_adult_pca(df_juv_proxy, df_adult_proxy, n_components=4)
    print(pca_result["summary"])
else:
    print("Insufficient rows for PCA comparison.")

# ─────────────────────────────────────────────────────────────────
# MODULE 3 — Modeling
# ─────────────────────────────────────────────────────────────────
print(SEP + "MODULE 3 — Mortality Models")

# 3a. Crude qx by Attained Age
crude = compute_crude_qx(df, by="Attained_Age", basis="count")
print("Crude qx by Attained Age (count basis):")
print(crude.to_string(index=False))

# 3b. Feature importance — aggregate to cell level first so AE is meaningful
print("\n\nFeature Importance (GBM on A/E Count — aggregated cells):")
FI_DIMS = ["Attained_Age", "Duration", "Sex_Clean", "Smoker_Status",
           "Insurance_Plan", "Face_Amount_Band_Numeric", "Slct_Ult_Ind", "SOA_Antp_Lvl_TP"]
fi_dims_present = [c for c in FI_DIMS if c in df.columns]

from config import ACTUAL_CNT_COL, EXPECTED_CNT_COL, EXPOSURE_CNT
df_agg_fi = (
    df.groupby(fi_dims_present, dropna=False, observed=True)
    .agg(Death_Count=(ACTUAL_CNT_COL, "sum"),
         ExpDth_VBT2015_Cnt=(EXPECTED_CNT_COL, "sum"),
         Policies_Exposed=(EXPOSURE_CNT, "sum"))
    .reset_index()
)
df_agg_fi["AE_Count"] = df_agg_fi["Death_Count"] / df_agg_fi["ExpDth_VBT2015_Cnt"].replace(0, float("nan"))
df_agg_fi = df_agg_fi.dropna(subset=["AE_Count"])
print(f"  Aggregated cells for FI: {len(df_agg_fi):,}")

fi_result = run_feature_importance(
    df_agg_fi,
    target="AE_Count",
    feature_cols=fi_dims_present,
    model_type="gbm",
    n_estimators=150,
    cv_folds=3,
)
print(fi_result["summary"])

# 3c. GAM — aggregate to (Attained_Age) level for speed, then fit
print("\n\nPoisson GAM — Smooth qx over Attained Age (on aggregated data):")
df_gam = (
    df.groupby("Attained_Age", observed=True)
    .agg(Death_Count=(ACTUAL_CNT_COL, "sum"),
         Policies_Exposed=(EXPOSURE_CNT, "sum"))
    .reset_index()
)
df_gam = df_gam[df_gam["Policies_Exposed"] > 0]

gam_result = fit_mortality_gam(
    df_gam,
    age_col="Attained_Age",
    basis="count",
    tensor_terms=None,
    n_splines=15,
)
print(f"  AIC  : {gam_result['aic']:.2f}")
print(f"  UBRE : {gam_result['gcv']:.4f}")
print(f"  λ    : {gam_result['lam']}")
print("\nSmoothed qx table (all ages):")
print(gam_result["smooth_qx"].to_string(index=False))

# 3d. GAM with tensor product: Age × Duration (aggregate to 2-way cell level)
print("\n\nPoisson GAM — Tensor smooth Age × Duration:")
df_gam_te = (
    df.groupby(["Attained_Age", "Duration"], observed=True)
    .agg(Death_Count=(ACTUAL_CNT_COL, "sum"),
         Policies_Exposed=(EXPOSURE_CNT, "sum"))
    .reset_index()
)
df_gam_te = df_gam_te[df_gam_te["Policies_Exposed"] > 0]

gam_te = fit_mortality_gam(
    df_gam_te,
    age_col="Attained_Age",
    basis="count",
    tensor_terms=["Duration"],
    n_splines=10,
)
print(f"  AIC  : {gam_te['aic']:.2f}")
print(f"  UBRE : {gam_te['gcv']:.4f}")
print(f"  λ    : {gam_te['lam']}")

# ─────────────────────────────────────────────────────────────────
# MODULE 4 — Validation & Grading
# ─────────────────────────────────────────────────────────────────
print(SEP + "MODULE 4 — Validation & Guardrails")

smooth_qx_df = gam_result["smooth_qx"].rename(columns={"Attained_Age": "Attained_Age"})

# Build a synthetic adult reference table (log-linear extension of GAM output)
last_age = int(smooth_qx_df["Attained_Age"].max())
adult_ages = np.arange(last_age, last_age + 20, dtype=float)
last_qx = float(smooth_qx_df["qx_Smooth"].iloc[-1])
growth_rate = 0.085   # ~8.5% annual Makeham-style increase in adult ages
adult_qx_vals = last_qx * np.exp(growth_rate * (adult_ages - last_age))
adult_ref = pd.DataFrame({"Attained_Age": adult_ages, "qx_Smooth": adult_qx_vals})

val = validation_report(
    smooth_qx=smooth_qx_df,
    adult_qx=adult_ref,
    age_col="Attained_Age",
    qx_col="qx_Smooth",
    join_age=last_age - 2,
    blend_ages=5,
)
print(val["summary"])
print("\nGraded Table (full age range):")
print(val["graded_table"][
    ["Attained_Age", "qx_Juvenile", "qx_Adult", "weight_Adult", "qx_Blended", "source"]
].to_string(index=False))

print(SEP + "Trial run complete.")
