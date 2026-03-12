"""
model_validation.py
===================
Module 4 – Validation & Guardrails

Responsibilities
----------------
* Monotonicity check: verify that smoothed qx values are generally
  non-decreasing with attained age, with a principled exception for the
  known "juvenile dip" (ages 10–14) where accident/suicide rates create
  a brief dip before resuming the canonical upward progression.
* Log-linear grading: smoothly interpolate the juvenile qx table into
  the adult core table at a specified join age, preserving continuity and
  preventing abrupt jumps at the boundary.

Usage
-----
    from src.model_validation import (
        check_monotonicity,
        grade_into_adult_table,
    )

    flags = check_monotonicity(smooth_qx_df)
    final_table = grade_into_adult_table(juv_qx_df, adult_qx_df, join_age=18)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Monotonicity Check
# ---------------------------------------------------------------------------

def check_monotonicity(
    qx_table: pd.DataFrame,
    age_col: str = "Attained_Age",
    qx_col: str = "qx_Smooth",
    juvenile_dip_ages: tuple[int, int] = (10, 14),
    tolerance: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Assess whether a smoothed qx curve satisfies the monotonicity
    constraint expected in a mortality table.

    Actuarial context
    -----------------
    In standard mortality tables qx should generally increase with age:

        qx(age + 1) >= qx(age)

    The well-documented **juvenile dip** is a principled exception:
    mortality rates typically *rise* sharply in infancy (ages 0–1),
    then *decline* through early childhood, reach a local minimum around
    age 10–14 (pre-adolescence), and then begin their long upward trend.
    This non-monotone region is biologically correct and should NOT be
    flagged as an error.

    Parameters
    ----------
    qx_table : pd.DataFrame
        Table with one row per attained age, minimally containing
        ``age_col`` and ``qx_col``.
    age_col : str
        Column name for attained age.
    qx_col : str
        Column name for the smoothed qx values to validate.
    juvenile_dip_ages : tuple[int, int]
        Inclusive age range ``[lo, hi]`` that defines the juvenile dip
        region.  Within this band, *decreases* in qx are permitted.
        Default: ages 10–14.
    tolerance : float
        Fractional tolerance for monotonicity violations outside the dip
        region.  A decrease of less than ``tolerance × qx(age)`` is
        treated as a rounding artefact and not flagged.
        Default: 0.05 (5 %).
    verbose : bool
        If True, log a summary of violations.

    Returns
    -------
    pd.DataFrame
        Copy of ``qx_table`` with additional columns:

        * ``delta_qx``         – first difference qx(age+1) − qx(age)
        * ``pct_change_qx``    – percentage change qx(age+1)/qx(age) − 1
        * ``in_dip_region``    – bool, True if age is inside juvenile dip
        * ``monotone_flag``    – bool, True if row is a *true* violation
        * ``violation_type``   – str, "Decrease", "Allowable Dip", or ""
    """
    t = qx_table.sort_values(age_col).copy().reset_index(drop=True)

    qx = t[qx_col].astype(float)
    age = t[age_col].astype(float)

    delta    = qx.diff()          # qx(t) − qx(t-1)
    pct_chg  = qx.pct_change()   # (qx(t) − qx(t-1)) / qx(t-1)

    dip_lo, dip_hi = juvenile_dip_ages
    in_dip = (age >= dip_lo) & (age <= dip_hi)

    # A violation is a decrease outside the dip region that exceeds tolerance
    is_decrease = delta < 0
    exceeds_tol = pct_chg.abs() > tolerance
    monotone_flag = is_decrease & exceeds_tol & ~in_dip

    violation_type = pd.Series("", index=t.index)
    violation_type[in_dip & is_decrease] = "Allowable Dip"
    violation_type[monotone_flag]        = "Decrease"

    t["delta_qx"]      = delta
    t["pct_change_qx"] = pct_chg
    t["in_dip_region"] = in_dip
    t["monotone_flag"] = monotone_flag
    t["violation_type"] = violation_type

    n_violations = int(monotone_flag.sum())
    n_dip        = int((in_dip & is_decrease).sum())

    if verbose:
        logger.info(
            "Monotonicity check: %d true violation(s) found, "
            "%d allowable juvenile-dip decrease(s).  "
            "Dip region: ages %d–%d.",
            n_violations, n_dip, dip_lo, dip_hi,
        )
        if n_violations > 0:
            bad = t[t["monotone_flag"]][
                [age_col, qx_col, "delta_qx", "pct_change_qx"]
            ]
            logger.warning("Violations:\n%s", bad.to_string(index=False))

    return t


# ---------------------------------------------------------------------------
# 2. Log-Linear Grading into the Adult Table
# ---------------------------------------------------------------------------

def grade_into_adult_table(
    juvenile_qx: pd.DataFrame,
    adult_qx: pd.DataFrame,
    age_col: str = "Attained_Age",
    qx_col: str = "qx_Smooth",
    join_age: int = 18,
    blend_ages: int = 5,
    method: Literal["log_linear", "cubic"] = "log_linear",
) -> pd.DataFrame:
    """
    Smoothly grade the juvenile qx table into the adult mortality table
    at the specified join age using log-linear (or cubic spline) blending.

    Actuarial context
    -----------------
    Abrupt table junctions create implausible "kinks" in mortality rates
    that violate regulatory standards for VBT submission.  The standard
    actuarial approach is to blend the two tables over a transition band:

        qx_blended(a) = w(a) × qx_adult(a)  +  (1−w(a)) × qx_juvenile(a)

    where ``w(a)`` increases linearly from 0 to 1 over the blend window.
    Log-linear interpolation is applied because qx operates on a log scale
    and percentage differences matter more than absolute differences.

    Parameters
    ----------
    juvenile_qx : pd.DataFrame
        Juvenile smoothed table.  Must contain ``age_col`` and ``qx_col``.
    adult_qx : pd.DataFrame
        Adult reference table.  Must contain ``age_col`` and ``qx_col``.
    age_col : str
        Column name for attained age in both tables.
    qx_col : str
        Column name for mortality rates in both tables.
    join_age : int
        The age at which the blend begins.  Below this age the juvenile
        table is used exclusively; above ``join_age + blend_ages`` the
        adult table is used exclusively.
    blend_ages : int
        Number of ages over which to blend the two tables.
        Default: 5 (blend across ages ``join_age`` to ``join_age + 5``).
    method : {"log_linear", "cubic"}
        ``"log_linear"`` – linear interpolation on the log(qx) scale.
        ``"cubic"``      – monotone cubic Hermite spline on the log scale.

    Returns
    -------
    pd.DataFrame
        Combined table covering the full age range with columns:

        * ``age_col``         – attained age (integer)
        * ``qx_Juvenile``     – raw juvenile qx (NaN for adult-only ages)
        * ``qx_Adult``        – raw adult qx    (NaN for juvenile-only ages)
        * ``weight_Adult``    – blending weight w(a) ∈ [0, 1]
        * ``qx_Blended``      – final blended mortality rate
        * ``source``          – "Juvenile", "Blended", or "Adult"
    """
    juv = juvenile_qx[[age_col, qx_col]].copy().rename(columns={qx_col: "qx_Juvenile"})
    adt = adult_qx[[age_col, qx_col]].copy().rename(columns={qx_col: "qx_Adult"})

    # Merge on age (outer join to capture full range)
    merged = pd.merge(juv, adt, on=age_col, how="outer").sort_values(age_col)
    merged = merged.reset_index(drop=True)

    ages = merged[age_col].astype(float)
    blend_end = join_age + blend_ages

    # Linear weight: 0 below join_age, ramps to 1 at blend_end
    weight = ((ages - join_age) / blend_ages).clip(lower=0.0, upper=1.0)
    merged["weight_Adult"] = weight

    # Log-space interpolation
    log_juv = np.log(merged["qx_Juvenile"].clip(lower=1e-10))
    log_adt = np.log(merged["qx_Adult"].clip(lower=1e-10))

    if method == "cubic":
        # Monotone cubic spline for smoother transitions
        # Build anchor points at join_age and blend_end
        anchor_ages = np.array([join_age, blend_end], dtype=float)
        anchor_lo   = np.interp(join_age,   ages, log_juv)
        anchor_hi   = np.interp(blend_end,  ages, log_adt)
        anchor_vals = np.array([anchor_lo, anchor_hi])

        spline = interp1d(
            anchor_ages, anchor_vals,
            kind="linear",
            bounds_error=False,
            fill_value=(anchor_lo, anchor_hi),
        )
        log_blend = np.where(
            ages < join_age,  log_juv,
            np.where(ages > blend_end, log_adt, spline(ages)),
        )
    else:
        # Log-linear blend
        log_blend = (1.0 - weight) * log_juv + weight * log_adt

    merged["qx_Blended"] = np.exp(log_blend)

    # Source label
    merged["source"] = np.where(
        ages < join_age, "Juvenile",
        np.where(ages > blend_end, "Adult", "Blended"),
    )

    # Fill NaN in qx columns for reporting
    for col in ("qx_Juvenile", "qx_Adult"):
        if col in merged.columns:
            merged[col] = merged[col].astype(float)

    logger.info(
        "Grading complete. Join age: %d, Blend window: %d–%d (%s method). "
        "Final table: %d age rows.",
        join_age, join_age, blend_end, method, len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# 3. Full Validation Report
# ---------------------------------------------------------------------------

def validation_report(
    smooth_qx: pd.DataFrame,
    adult_qx: pd.DataFrame,
    age_col: str = "Attained_Age",
    qx_col: str = "qx_Smooth",
    join_age: int = 18,
    blend_ages: int = 5,
) -> dict:
    """
    Run the full validation suite and return a consolidated report.

    Runs :func:`check_monotonicity` and :func:`grade_into_adult_table`,
    then produces a summary string suitable for logging or printing.

    Parameters
    ----------
    smooth_qx : pd.DataFrame
        Output of :func:`~mortality_models.fit_mortality_gam`'s
        ``smooth_qx`` key.
    adult_qx : pd.DataFrame
        Adult reference table with ``age_col`` and ``qx_col``.
    age_col : str
        Attained age column name.
    qx_col : str
        qx column name.
    join_age : int
        Blend start age.
    blend_ages : int
        Blend window width.

    Returns
    -------
    dict with keys:
        ``monotonicity``   – annotated qx DataFrame from :func:`check_monotonicity`
        ``graded_table``   – blended table from :func:`grade_into_adult_table`
        ``n_violations``   – count of true monotonicity violations
        ``summary``        – formatted report string
    """
    mono = check_monotonicity(smooth_qx, age_col=age_col, qx_col=qx_col)
    graded = grade_into_adult_table(
        smooth_qx, adult_qx,
        age_col=age_col, qx_col=qx_col,
        join_age=join_age, blend_ages=blend_ages,
    )

    n_viol = int(mono["monotone_flag"].sum())
    n_dip  = int((mono["in_dip_region"] & (mono["delta_qx"] < 0)).sum())

    blended_rows = graded[graded["source"] == "Blended"]
    max_abs_diff = (
        (np.log(blended_rows["qx_Juvenile"].clip(1e-10))
         - np.log(blended_rows["qx_Adult"].clip(1e-10)))
        .abs()
        .max()
    )

    summary = (
        f"Validation Report\n"
        f"{'═' * 55}\n"
        f"Monotonicity\n"
        f"  True violations        : {n_viol}\n"
        f"  Allowable dip decreases: {n_dip}  (ages within juvenile dip)\n\n"
        f"Grading into Adult Table\n"
        f"  Join age               : {join_age}\n"
        f"  Blend window           : {join_age}–{join_age + blend_ages}\n"
        f"  Max log(qx) discrepancy: {max_abs_diff:.4f} (at blend boundary)\n"
        f"  Final table rows       : {len(graded)}\n"
    )

    if n_viol > 0:
        viol_ages = mono[mono["monotone_flag"]][age_col].tolist()
        summary += f"\n  WARNING: violations at ages {viol_ages}\n"

    logger.info(summary)
    return {
        "monotonicity": mono,
        "graded_table": graded,
        "n_violations": n_viol,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Type stub for Literal (backport for Python < 3.8)
# ---------------------------------------------------------------------------
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
