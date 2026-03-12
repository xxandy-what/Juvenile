"""
mortality_models.py
===================
Module 3 – Mortality Modeling Engine

Responsibilities
----------------
* Baseline:  Compute crude mortality rates (qx) weighted by count and amount.
* AI/ML:     Random Forest / Gradient Boosting pipeline to rank predictor
             importance on the A/E ratio.
* GAM:       Fit Poisson GAM over Attained Age (+ optional tensor products)
             to produce a smoothed qx curve suitable for VBT derivation.

Usage
-----
    from src.mortality_models import (
        compute_crude_qx,
        run_feature_importance,
        fit_mortality_gam,
    )

    qx_table = compute_crude_qx(df, by="Attained_Age", basis="count")
    fi_result = run_feature_importance(df)
    gam_result = fit_mortality_gam(df)
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pygam import PoissonGAM, LinearGAM, te, s, f
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from config import (
    ACTUAL_AMT_COL,
    ACTUAL_CNT_COL,
    EXPECTED_AMT_COL,
    EXPECTED_CNT_COL,
    EXPOSURE_AMT,
    EXPOSURE_CNT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Crude qx
# ---------------------------------------------------------------------------

def compute_crude_qx(
    df: pd.DataFrame,
    by: str | list[str],
    basis: Literal["count", "amount"] = "count",
    min_exposure: float = 1.0,
) -> pd.DataFrame:
    """
    Compute crude mortality rates ``qx`` aggregated by one or more dimensions.

    Crude qx = Deaths / Exposure (exposure-weighted mortality rate).

    Parameters
    ----------
    df : pd.DataFrame
        Processed study DataFrame.
    by : str | list[str]
        Dimension(s) to group by (e.g., ``"Attained_Age"`` or
        ``["Attained_Age", "Sex_Clean"]``).
    basis : {"count", "amount"}
        Whether to weight by policy count or face amount.
    min_exposure : float
        Minimum exposure to include a cell (default 1.0 policy-year).

    Returns
    -------
    pd.DataFrame
        Table with ``by`` columns plus:

        * ``Deaths``    – sum of actual deaths
        * ``Expected``  – sum of expected deaths (VBT 2015)
        * ``Exposure``  – total exposure
        * ``qx_Crude``  – crude mortality rate
        * ``AE_Ratio``  – A/E ratio
    """
    if isinstance(by, str):
        by = [by]

    if basis == "count":
        death_col, exp_col, exposure_col = ACTUAL_CNT_COL, EXPECTED_CNT_COL, EXPOSURE_CNT
    else:
        death_col, exp_col, exposure_col = ACTUAL_AMT_COL, EXPECTED_AMT_COL, EXPOSURE_AMT

    agg = (
        df.groupby(by, dropna=False, observed=True)
        .agg(
            Deaths=(death_col, "sum"),
            Expected=(exp_col, "sum"),
            Exposure=(exposure_col, "sum"),
        )
        .reset_index()
    )

    agg = agg[agg["Exposure"] >= min_exposure].copy()
    agg["qx_Crude"] = _safe_div(agg["Deaths"], agg["Exposure"])
    agg["AE_Ratio"]  = _safe_div(agg["Deaths"], agg["Expected"])

    # Sort by the first dimension (typically Attained_Age)
    try:
        agg = agg.sort_values(by[0])
    except Exception:
        pass

    logger.info(
        "Crude qx (%s basis) computed for %d cells by %s.",
        basis, len(agg), by,
    )
    return agg


# ---------------------------------------------------------------------------
# 2. Feature Importance (Random Forest / Gradient Boosting)
# ---------------------------------------------------------------------------

_DEFAULT_FEATURES = [
    "Attained_Age",
    "Duration",
    "Issue_Age",
    "Face_Amount_Band_Numeric",
    "Sex_Clean",
    "Smoker_Status",
    "Insurance_Plan",
    "Slct_Ult_Ind",
    "SOA_Antp_Lvl_TP",
]

_CATEGORICAL_FEATURES = [
    "Sex_Clean",
    "Smoker_Status",
    "Insurance_Plan",
    "Slct_Ult_Ind",
    "SOA_Antp_Lvl_TP",
]


def run_feature_importance(
    df: pd.DataFrame,
    target: str = "AE_Count",
    feature_cols: Optional[list[str]] = None,
    model_type: Literal["rf", "gbm"] = "gbm",
    n_estimators: int = 300,
    cv_folds: int = 5,
    random_state: int = 42,
    compute_shap: bool = True,
    shap_sample_size: int = 500,
) -> dict:
    """
    Train a Random Forest or Gradient Boosting model to rank predictors
    of the mortality A/E ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Processed study DataFrame.  Must contain the columns in
        ``feature_cols`` and the target column.
    target : str
        Target variable.  Defaults to ``"AE_Count"`` (A/E by count).
        Pass ``"AE_Amount"`` for amount basis, or any numeric column.
    feature_cols : list[str], optional
        Predictor columns.  Defaults to :data:`_DEFAULT_FEATURES` that
        are present in ``df``.
    model_type : {"rf", "gbm"}
        ``"rf"``  → sklearn RandomForestRegressor.
        ``"gbm"`` → sklearn GradientBoostingRegressor.
    n_estimators : int
        Number of trees.
    cv_folds : int
        Number of cross-validation folds for the RMSE estimate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``model``          – fitted sklearn estimator
        ``feature_names``  – list of feature names after encoding
        ``importances``    – pd.DataFrame sorted by importance (descending)
        ``cv_rmse``        – cross-validated RMSE (mean ± std)
        ``summary``        – formatted string report
    """
    # ---------- build target column if not present --------------------------
    work = df.copy()
    if target not in work.columns:
        if target == "AE_Count":
            work["AE_Count"] = _safe_div(
                work[ACTUAL_CNT_COL], work[EXPECTED_CNT_COL]
            )
        elif target == "AE_Amount":
            work["AE_Amount"] = _safe_div(
                work[ACTUAL_AMT_COL], work[EXPECTED_AMT_COL]
            )
        else:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

    # ---------- select available feature columns ----------------------------
    if feature_cols is None:
        feature_cols = [c for c in _DEFAULT_FEATURES if c in work.columns]

    work = work[feature_cols + [target]].dropna(subset=[target])

    # ---------- encode categoricals -----------------------------------------
    cat_present = [c for c in _CATEGORICAL_FEATURES if c in feature_cols]
    num_present = [c for c in feature_cols if c not in cat_present]

    if cat_present:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        work[cat_present] = enc.fit_transform(work[cat_present].astype(str))

    # Impute NaN in numeric features with column median (avoids losing all rows
    # when a feature like Face_Amount_Band_Numeric has many NA-coded cells)
    if num_present:
        for col in num_present:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            median_val = work[col].median()
            work[col] = work[col].fillna(median_val if not pd.isna(median_val) else 0.0)

    work = work.dropna(subset=[target])
    X = work[feature_cols].astype(float).values
    y = work[target].astype(float).values

    # ---------- build model pipeline ----------------------------------------
    if model_type == "rf":
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        estimator = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=random_state,
        )

    estimator.fit(X, y)

    # ---------- cross-validated RMSE ----------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(
            estimator, X, y,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
    cv_rmse_mean = float(-cv_scores.mean())
    cv_rmse_std  = float(cv_scores.std())

    # ---------- feature importance table ------------------------------------
    if model_type == "rf":
        raw_imp = estimator.feature_importances_
        imp_source = "Mean Decrease in Impurity (MDI)"
    else:
        raw_imp = estimator.feature_importances_
        imp_source = "Gradient Boost feature_importances_"

    importances = (
        pd.DataFrame({"Feature": feature_cols, "Importance": raw_imp})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    importances["Importance_%"] = (
        importances["Importance"] / importances["Importance"].sum() * 100
    ).round(2)

    # ---------- SHAP values (TreeExplainer for GBM / RF) --------------------
    shap_values: Optional[np.ndarray] = None
    shap_importances: Optional[pd.DataFrame] = None
    shap_summary_str = ""

    if compute_shap:
        try:
            import shap  # type: ignore

            # Sample rows for speed on large datasets
            if len(X) > shap_sample_size:
                rng = np.random.default_rng(random_state)
                idx = rng.choice(len(X), size=shap_sample_size, replace=False)
                X_shap = X[idx]
            else:
                X_shap = X

            explainer = shap.TreeExplainer(estimator)
            shap_vals = explainer.shap_values(X_shap)
            shap_values = shap_vals

            # Mean absolute SHAP per feature
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_importances = (
                pd.DataFrame({"Feature": feature_cols, "SHAP_MeanAbs": mean_abs_shap})
                .sort_values("SHAP_MeanAbs", ascending=False)
                .reset_index(drop=True)
            )
            shap_importances["SHAP_%"] = (
                shap_importances["SHAP_MeanAbs"]
                / shap_importances["SHAP_MeanAbs"].sum() * 100
            ).round(2)

            shap_summary_str = (
                f"\n\nSHAP Feature Attribution  (TreeExplainer, n={len(X_shap):,} rows)\n"
                f"{'─' * 55}\n"
                + shap_importances[["Feature", "SHAP_%"]].to_string(index=False)
            )
            logger.info("SHAP values computed for %d rows.", len(X_shap))

        except ImportError:
            shap_summary_str = (
                "\n\n[SHAP not available — install with: pip install shap]"
            )
            logger.warning("shap package not installed; skipping SHAP computation.")
        except Exception as exc:
            shap_summary_str = f"\n\n[SHAP computation failed: {exc}]"
            logger.warning("SHAP computation failed: %s", exc)

    summary = (
        f"Feature Importance – {model_type.upper()}  (target: {target})\n"
        f"{'─' * 55}\n"
        f"Model          : {type(estimator).__name__}\n"
        f"Importance src : {imp_source}\n"
        f"Training rows  : {len(X):,}\n"
        f"CV RMSE ({cv_folds}-fold): {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}\n\n"
        + importances[["Feature", "Importance_%"]].to_string(index=False)
        + shap_summary_str
    )

    logger.info(summary)

    return {
        "model": estimator,
        "feature_names": feature_cols,
        "importances": importances,
        "cv_rmse": (cv_rmse_mean, cv_rmse_std),
        "shap_values": shap_values,
        "shap_importances": shap_importances,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 3. GAM – Smoothed Mortality Curve
# ---------------------------------------------------------------------------

def fit_mortality_gam(
    df: pd.DataFrame,
    age_col: str = "Attained_Age",
    basis: Literal["count", "amount"] = "count",
    tensor_terms: Optional[list[str]] = None,
    n_splines: int = 20,
    lam_candidates: Optional[list[float]] = None,
    max_iter: int = 200,
) -> dict:
    """
    Fit a Poisson GAM to smooth the mortality curve over Attained Age,
    optionally including tensor-product smooths for interaction effects
    (e.g., Age × Duration) as used in the VBT creation process.

    Model specification
    -------------------
    The canonical VBT-style Poisson model is:

        log(E[Deaths]) = log(Exposure) + s(Age) + [te(Age, Duration)] + ...

    where ``s()`` is a univariate spline smooth and ``te()`` is a
    tensor-product smooth capturing non-linear interactions.

    Parameters
    ----------
    df : pd.DataFrame
        Processed study DataFrame.
    age_col : str
        Continuous age column.  Use ``"Attained_Age_ALB"`` for the
        ALB-converted version.
    basis : {"count", "amount"}
        Whether to model count-based or amount-based deaths.
    tensor_terms : list[str], optional
        Additional numeric columns to interact with ``age_col`` via a
        tensor-product smooth (e.g., ``["Duration"]``).
        If None, only a univariate smooth on age is used.
    n_splines : int
        Number of B-spline basis functions per term.
    lam_candidates : list[float], optional
        Grid of regularisation lambda values for GCV grid search.
        Defaults to a log-scale grid ``[0.001, …, 100]``.
    max_iter : int
        Maximum PIRLS iterations for the Poisson GAM solver.

    Returns
    -------
    dict with keys:
        ``gam``            – fitted pyGAM PoissonGAM object
        ``smooth_qx``      – pd.DataFrame with age grid, fitted deaths,
                             log-exposure offset, and smoothed qx
        ``aic``            – Akaike Information Criterion
        ``summary``        – pyGAM summary string
        ``feature_cols``   – ordered list of terms used in the model
    """
    if basis == "count":
        death_col, exposure_col = ACTUAL_CNT_COL, EXPOSURE_CNT
    else:
        death_col, exposure_col = ACTUAL_AMT_COL, EXPOSURE_AMT

    # ---------- build feature matrix ----------------------------------------
    feature_cols: list[str] = [age_col]
    if tensor_terms:
        feature_cols += [t for t in tensor_terms if t in df.columns and t != age_col]

    work = df[[death_col, exposure_col] + feature_cols].dropna()
    work = work[work[exposure_col] > 0].copy()

    X = work[feature_cols].astype(float).values
    y = work[death_col].astype(float).values
    log_exposure = np.log(work[exposure_col].astype(float).values)

    # ---------- build GAM terms ---------------------------------------------
    if lam_candidates is None:
        lam_candidates = list(np.logspace(-3, 2, 15))

    n_features = X.shape[1]

    if n_features == 1:
        # Simple univariate smooth on age
        terms = s(0, n_splines=n_splines, spline_order=3)
    elif n_features == 2 and tensor_terms:
        # Tensor-product smooth: Age × tensor_term[0]
        terms = te(0, 1, n_splines=[n_splines, n_splines // 2], spline_order=3)
    else:
        # Multiple additive terms
        terms = s(0, n_splines=n_splines, spline_order=3)
        for i in range(1, n_features):
            terms = terms + s(i, n_splines=max(6, n_splines // 2), spline_order=3)

    # ---------- fit with GCV lambda grid search -----------------------------
    lam_grid = np.array([[lam] * n_features for lam in lam_candidates])

    gam = PoissonGAM(terms, max_iter=max_iter)
    gam.gridsearch(X, y, lam=lam_grid, objective="UBRE", progress=False)

    logger.info(
        "Poisson GAM fitted. AIC=%.2f, UBRE score=%.4f, lambda=%s.",
        gam.statistics_["AIC"],
        gam.statistics_["UBRE"],
        gam.lam,
    )

    # ---------- smooth qx prediction over age grid --------------------------
    age_min = float(work[age_col].min())
    age_max = float(work[age_col].max())
    age_grid = np.arange(age_min, age_max + 1.0, 1.0)

    if n_features == 1:
        X_pred = age_grid.reshape(-1, 1)
    else:
        # Hold other features at their median
        medians = np.median(X[:, 1:], axis=0)
        X_pred = np.column_stack(
            [age_grid] + [np.full_like(age_grid, m) for m in medians]
        )

    # Use median log-exposure for prediction
    median_log_exp = float(np.median(log_exposure))
    exposure_offset = np.full(len(age_grid), median_log_exp)

    pred_deaths = gam.predict(X_pred)
    # Implied qx = predicted deaths / median exposure
    median_exposure = np.exp(median_log_exp)
    pred_qx = pred_deaths / median_exposure

    # 95 % prediction intervals
    ci_lo, ci_hi = gam.confidence_intervals(X_pred, width=0.95).T
    qx_lo = ci_lo / median_exposure
    qx_hi = ci_hi / median_exposure

    smooth_qx = pd.DataFrame({
        age_col: age_grid,
        "Predicted_Deaths": pred_deaths,
        "qx_Smooth": pred_qx,
        "qx_Smooth_CI_Lo": qx_lo,
        "qx_Smooth_CI_Hi": qx_hi,
        "Median_Log_Exposure": median_log_exp,
    })

    return {
        "gam": gam,
        "smooth_qx": smooth_qx,
        "aic": gam.statistics_["AIC"],
        "gcv": gam.statistics_["UBRE"],
        "lam": gam.lam,
        "feature_cols": feature_cols,
        "summary": str(gam.summary()),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return (numerator / denom).replace([np.inf, -np.inf], np.nan)
