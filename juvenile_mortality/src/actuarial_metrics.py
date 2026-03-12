"""
actuarial_metrics.py
====================
Module 2 – Actuarial Metrics & Exploratory Data Analysis

Responsibilities
----------------
* Compute Actual-to-Expected (A/E) ratios by policy count and face amount.
* Compute 95 % two-sided Confidence Intervals on A/E ratios using the
  normal approximation to the Poisson (standard actuarial practice for
  credibility-weighted mortality studies).
* PCA and Euclidean-distance tests to quantify structural differences
  between the Juvenile and Adult mortality datasets.

Usage
-----
    from src.actuarial_metrics import (
        compute_ae_ratios,
        ae_confidence_intervals,
        compare_juvenile_adult_pca,
    )

    ae = compute_ae_ratios(df, groupby=["Sex_Clean", "Attained_Age_Group"])
    ae_ci = ae_confidence_intervals(ae)
    pca_result = compare_juvenile_adult_pca(df_juv, df_adult)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    ACTUAL_AMT_COL,
    ACTUAL_CNT_COL,
    EXPECTED_AMT_COL,
    EXPECTED_CNT_COL,
    EXPOSURE_AMT,
    EXPOSURE_CNT,
    CI_LEVEL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. A/E Ratio Computation
# ---------------------------------------------------------------------------

def compute_ae_ratios(
    df: pd.DataFrame,
    groupby: list[str],
    min_exposure_cnt: float = 0.0,
    min_exposure_amt: float = 0.0,
) -> pd.DataFrame:
    """
    Aggregate mortality experience and compute A/E ratios by policy count
    and face amount, weighted by exposure.

    Parameters
    ----------
    df : pd.DataFrame
        Processed study DataFrame from JuvenileDataLoader.
    groupby : list[str]
        Dimensions to aggregate over (e.g., ``["Sex_Clean", "Attained_Age"]``).
    min_exposure_cnt : float
        Minimum policy exposure to include a cell (avoids noise from
        near-zero cells).
    min_exposure_amt : float
        Minimum amount exposure to include a cell.

    Returns
    -------
    pd.DataFrame
        Aggregated table with columns:

        * ``Policies_Exposed``   – sum of policy-count exposure
        * ``Amount_Exposed``     – sum of face-amount exposure
        * ``Death_Count``        – sum of actual deaths (count)
        * ``Death_Claim_Amount`` – sum of actual deaths (amount)
        * ``Exp_Count``          – sum of expected deaths by count
        * ``Exp_Amount``         – sum of expected deaths by amount
        * ``qx_Crude_Count``     – crude mortality rate (count)
        * ``qx_Crude_Amount``    – crude mortality rate (amount)
        * ``AE_Count``           – A/E ratio by count
        * ``AE_Amount``          – A/E ratio by amount
    """
    required = {ACTUAL_CNT_COL, EXPECTED_CNT_COL, ACTUAL_AMT_COL, EXPECTED_AMT_COL,
                EXPOSURE_CNT, EXPOSURE_AMT}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    agg = (
        df.groupby(groupby, dropna=False, observed=True)
        .agg(
            Policies_Exposed=(EXPOSURE_CNT, "sum"),
            Amount_Exposed=(EXPOSURE_AMT, "sum"),
            Death_Count=(ACTUAL_CNT_COL, "sum"),
            Death_Claim_Amount=(ACTUAL_AMT_COL, "sum"),
            Exp_Count=(EXPECTED_CNT_COL, "sum"),
            Exp_Amount=(EXPECTED_AMT_COL, "sum"),
        )
        .reset_index()
    )

    # Apply minimum exposure filters
    agg = agg[
        (agg["Policies_Exposed"] >= min_exposure_cnt)
        & (agg["Amount_Exposed"] >= min_exposure_amt)
    ].copy()

    # Crude mortality rates: qx = Deaths / Exposure
    agg["qx_Crude_Count"] = _safe_div(agg["Death_Count"], agg["Policies_Exposed"])
    agg["qx_Crude_Amount"] = _safe_div(agg["Death_Claim_Amount"], agg["Amount_Exposed"])

    # A/E ratios: AE = Actual / Expected
    agg["AE_Count"]  = _safe_div(agg["Death_Count"],        agg["Exp_Count"])
    agg["AE_Amount"] = _safe_div(agg["Death_Claim_Amount"], agg["Exp_Amount"])

    logger.info(
        "A/E ratios computed for %d cells across dimensions: %s.",
        len(agg),
        groupby,
    )
    return agg


# ---------------------------------------------------------------------------
# 2. Confidence Intervals on A/E Ratios
# ---------------------------------------------------------------------------

def ae_confidence_intervals(
    ae_df: pd.DataFrame,
    ci_level: float = CI_LEVEL,
) -> pd.DataFrame:
    """
    Attach two-sided confidence intervals to A/E ratios using the
    normal approximation to the Poisson distribution.

    Methodology
    -----------
    Under the assumption that actual deaths ``D ~ Poisson(E·q)``,
    the standard error of the A/E ratio is:

        SE(AE) = AE / sqrt(D)

    yielding the ``(1 - alpha)`` CI:

        AE ± z_{alpha/2} · AE / sqrt(D)

    For cells with ``D == 0``, the upper bound uses the exact Poisson
    zero-event bound: ``-ln(alpha/2) / E``.

    Parameters
    ----------
    ae_df : pd.DataFrame
        Output of :func:`compute_ae_ratios`.
    ci_level : float
        Confidence level, default 0.95.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:

        * ``AE_Count_CI_Lo`` / ``AE_Count_CI_Hi``
        * ``AE_Amount_CI_Lo`` / ``AE_Amount_CI_Hi``
        * ``Credibility_Count``   – Z = min(1, sqrt(D / 1082))  (LLID full credibility)
        * ``Credibility_Amount``  – analogous for amount basis
    """
    alpha = 1.0 - ci_level
    z = stats.norm.ppf(1 - alpha / 2)  # e.g. 1.96 for 95 %

    out = ae_df.copy()

    for basis, death_col, exp_col, ae_col in [
        ("Count",  "Death_Count",        "Exp_Count",  "AE_Count"),
        ("Amount", "Death_Claim_Amount", "Exp_Amount", "AE_Amount"),
    ]:
        d  = out[death_col].astype(float)
        ae = out[ae_col].astype(float)

        # Standard-error based CI (normal approx)
        se = ae / np.sqrt(d.replace(0, np.nan))

        lo = (ae - z * se).clip(lower=0.0)
        hi = (ae + z * se)

        # Zero-death cells: use exact Poisson upper bound
        zero_mask = d == 0
        if zero_mask.any():
            # Upper bound: -ln(alpha/2) / Expected
            hi[zero_mask] = -np.log(alpha / 2) / out.loc[zero_mask, exp_col].replace(0, np.nan)
            lo[zero_mask] = 0.0

        out[f"AE_{basis}_CI_Lo"] = lo
        out[f"AE_{basis}_CI_Hi"] = hi

        # Limited fluctuation credibility (LLID standard: 1082 expected deaths for full)
        full_cred_threshold = 1082.0
        out[f"Credibility_{basis}"] = np.sqrt(d / full_cred_threshold).clip(upper=1.0)

    logger.info("Confidence intervals (%.0f%%) appended.", ci_level * 100)
    return out


# ---------------------------------------------------------------------------
# 3. PCA / Structural Distance – Juvenile vs Adult
# ---------------------------------------------------------------------------

def compare_juvenile_adult_pca(
    df_juvenile: pd.DataFrame,
    df_adult: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    n_components: int = 5,
) -> dict:
    """
    Quantify the structural difference between the Juvenile and Adult
    mortality datasets using PCA and Euclidean distance in feature space.

    Methodology
    -----------
    1. Select common numeric feature columns present in both datasets.
    2. Standardise (zero mean, unit variance) the pooled data.
    3. Fit PCA on the pooled standardised data.
    4. Project both subsets into PC space.
    5. Compute:

       * **Centroid distance** – Euclidean distance between the mean PC
         vectors of the juvenile and adult populations.
       * **Distribution overlap** – Bhattacharyya coefficient on the
         first two PCs (higher = more overlap = more similar).
       * **Explained variance** per component.

    Parameters
    ----------
    df_juvenile : pd.DataFrame
        Juvenile cohort (issue ages 0–17).
    df_adult : pd.DataFrame
        Adult cohort (issue ages 18+) or the full SOA adult table.
    feature_cols : list[str], optional
        Numeric columns to use.  Defaults to standard actuarial features.
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    dict with keys:
        ``pca``              – fitted sklearn PCA object
        ``scaler``           – fitted StandardScaler
        ``juvenile_pcs``     – PC-projected juvenile DataFrame
        ``adult_pcs``        – PC-projected adult DataFrame
        ``centroid_distance``– float, Euclidean distance between centroids
        ``explained_variance_ratio`` – array of per-component variance
        ``bhattacharyya_coeff``     – float, 0 (no overlap) to 1 (identical)
        ``summary``          – formatted string report
    """
    default_features = [
        "Attained_Age", "Duration", "Issue_Age",
        "Policies_Exposed", "Amount_Exposed",
        "Death_Count", "ExpDth_VBT2015_Cnt",
        "Death_Claim_Amount", "ExpDth_VBT2015_Amt",
    ]
    if feature_cols is None:
        feature_cols = [
            c for c in default_features
            if c in df_juvenile.columns and c in df_adult.columns
        ]

    if not feature_cols:
        raise ValueError("No common numeric feature columns found between the two datasets.")

    # Drop rows with NaN in feature columns
    juv = df_juvenile[feature_cols].dropna()
    adt = df_adult[feature_cols].dropna()

    # Standardise on the pooled distribution
    pooled = pd.concat([juv, adt], ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(pooled)

    juv_scaled = scaler.transform(juv)
    adt_scaled = scaler.transform(adt)
    pooled_scaled = scaler.transform(pooled)

    # PCA fit on pooled
    n_comp = min(n_components, len(feature_cols))
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(pooled_scaled)

    juv_pcs = pd.DataFrame(
        pca.transform(juv_scaled),
        columns=[f"PC{i+1}" for i in range(n_comp)],
    )
    adt_pcs = pd.DataFrame(
        pca.transform(adt_scaled),
        columns=[f"PC{i+1}" for i in range(n_comp)],
    )

    # Centroid Euclidean distance (in PC space)
    juv_centroid = juv_pcs.mean().values
    adt_centroid = adt_pcs.mean().values
    centroid_dist = float(np.linalg.norm(juv_centroid - adt_centroid))

    # Bhattacharyya coefficient on PC1 & PC2 (histogram-based)
    bhat = _bhattacharyya_coeff(juv_pcs["PC1"].values, adt_pcs["PC1"].values)

    summary = (
        f"PCA Structural Comparison – Juvenile vs Adult\n"
        f"{'─' * 50}\n"
        f"Features used          : {feature_cols}\n"
        f"Juvenile records       : {len(juv):,}\n"
        f"Adult records          : {len(adt):,}\n"
        f"Components retained    : {n_comp}\n"
        f"Explained var (cumul.) : {np.cumsum(pca.explained_variance_ratio_).round(4).tolist()}\n"
        f"Centroid distance (PC) : {centroid_dist:.4f}\n"
        f"Bhattacharyya coeff    : {bhat:.4f}  "
        f"({'high overlap' if bhat > 0.7 else 'moderate overlap' if bhat > 0.4 else 'low overlap'})\n"
    )

    logger.info(summary)

    return {
        "pca": pca,
        "scaler": scaler,
        "juvenile_pcs": juv_pcs,
        "adult_pcs": adt_pcs,
        "centroid_distance": centroid_dist,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "bhattacharyya_coeff": bhat,
        "feature_cols": feature_cols,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two Series, returning NaN where denominator is zero."""
    denom = denominator.replace(0, np.nan)
    result = numerator / denom
    return result.replace([np.inf, -np.inf], np.nan)


def _bhattacharyya_coeff(a: np.ndarray, b: np.ndarray, bins: int = 50) -> float:
    """
    Compute the Bhattacharyya coefficient between two 1-D distributions
    using a shared histogram.  Returns a value in [0, 1].
    """
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)

    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)

    # Normalise to proper probabilities
    bin_width = (hi - lo) / bins
    pa = ha * bin_width
    pb = hb * bin_width

    coeff = float(np.sum(np.sqrt(pa * pb)))
    return coeff
