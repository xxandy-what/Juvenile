"""
data_loader.py
==============
Module 1 – Data Ingestion & Actuarial Preprocessing

Responsibilities
----------------
* Load TSV / CSV / Parquet source data.
* Apply study-scope exclusions (SOA_Post_Lvl_Ind == "PLT").
* Convert Age Nearest Birthday (ANB) → Age Last Birthday (ALB).
* Engineer standard actuarial grouping features.
* Return a clean, typed DataFrame ready for analysis.

Usage
-----
    from src.data_loader import JuvenileDataLoader

    loader = JuvenileDataLoader(r"D:\...\Juvenile_cleaned.txt")
    df = loader.load()          # full pipeline
    print(loader.summary())
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    CAT_COLUMNS,
    NUM_COLUMNS,
    FACE_BAND_MAP,
    JUVENILE_MAX_ISSUE_AGE,
    EXPOSURE_CNT,
    EXPOSURE_AMT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: type coercion
# ---------------------------------------------------------------------------

def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce known columns to their correct dtypes in place.
    Preserves literal 'N/A' strings as category members (not NaN).
    """
    for col in NUM_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CAT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").astype("category")

    return df


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class JuvenileDataLoader:
    """
    End-to-end loader for the ILEC Juvenile Mortality study file.

    Parameters
    ----------
    path : str | Path
        Absolute path to the source TSV, CSV, or Parquet file.
    sep : str, optional
        Field delimiter for text files.  Defaults to ``\\t`` (tab).
    parquet_cache : str | Path | None
        Path where the processed DataFrame will be written as Parquet
        after the first load.  On subsequent runs the Parquet file is
        read directly, bypassing the (slower) CSV parsing step and
        freeing the raw CSV memory immediately after conversion.
        Pass ``None`` to disable caching.
    verbose : bool
        If True, log progress messages.

    Examples
    --------
    >>> loader = JuvenileDataLoader("Juvenile_cleaned.txt",
    ...                             parquet_cache="data/cache.parquet")
    >>> df = loader.load()
    >>> print(loader.summary())
    """

    def __init__(
        self,
        path: str | Path,
        sep: str = "\t",
        parquet_cache: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> None:
        self.path = Path(path)
        self.sep = sep
        self.parquet_cache: Optional[Path] = (
            Path(parquet_cache) if parquet_cache is not None else None
        )
        self.verbose = verbose

        self._raw: Optional[pd.DataFrame] = None
        self._processed: Optional[pd.DataFrame] = None
        self._exclusion_counts: dict[str, int] = {}
        self._loaded_from_cache: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Run the full ingestion pipeline and return the processed DataFrame.

        Pipeline steps
        --------------
        1. If a valid Parquet cache exists, load from it directly (fast path).
        2. Otherwise: read source file → coerce dtypes → apply exclusions →
           ANB→ALB conversion → feature engineering → write Parquet cache →
           release raw CSV memory.

        Returns
        -------
        pd.DataFrame
            Clean, feature-enriched study DataFrame.
        """
        # ── Fast path: Parquet cache already exists ──────────────────────
        if self.parquet_cache is not None and self.parquet_cache.exists():
            if self.verbose:
                logger.info(
                    "Parquet cache found at '%s'. Loading directly (skipping CSV).",
                    self.parquet_cache,
                )
            df = pd.read_parquet(self.parquet_cache)
            self._processed = df
            self._loaded_from_cache = True
            # Populate exclusion counts stub so summary() doesn't crash
            self._exclusion_counts.setdefault("(loaded from Parquet cache)", 0)
            if self.verbose:
                logger.info(
                    "Cache load complete: %d rows × %d columns.",
                    df.shape[0], df.shape[1],
                )
            return df

        # ── Slow path: build from source CSV / TSV ───────────────────────
        self._raw = self._read_file()
        df = _coerce_dtypes(self._raw.copy())
        df = self._apply_exclusions(df)
        df = self._convert_anb_to_alb(df)
        df = self._engineer_features(df)
        self._processed = df

        # ── Write Parquet cache then free raw memory ──────────────────────
        if self.parquet_cache is not None:
            self.parquet_cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.parquet_cache, index=False)
            if self.verbose:
                logger.info("Parquet cache written to '%s'.", self.parquet_cache)
            # Release the raw CSV DataFrame — the cache is the source of truth now
            self._raw = None
            import gc
            gc.collect()
            if self.verbose:
                logger.info("Raw CSV DataFrame released from memory.")

        if self.verbose:
            logger.info(
                "Load complete: %d rows × %d columns after preprocessing.",
                df.shape[0], df.shape[1],
            )
        return df

    @property
    def raw(self) -> pd.DataFrame:
        """The unprocessed DataFrame as read from disk."""
        if self._raw is None:
            raise RuntimeError("Call .load() first.")
        return self._raw

    @property
    def processed(self) -> pd.DataFrame:
        """The fully processed DataFrame."""
        if self._processed is None:
            raise RuntimeError("Call .load() first.")
        return self._processed

    def summary(self) -> str:
        """Return a human-readable summary of load statistics."""
        if self._processed is None:
            return "No data loaded yet. Call .load() first."

        if self._loaded_from_cache:
            raw_info = f"(loaded from Parquet cache: {self.parquet_cache})"
        elif self._raw is not None:
            raw_info = f"{len(self._raw):,}"
        else:
            raw_info = "(released after Parquet cache written)"

        lines = [
            f"Source     : {self.path.name}",
            f"Raw rows   : {raw_info}",
            f"Final rows : {len(self._processed):,}",
            "Exclusions :",
        ]
        for reason, n in self._exclusion_counts.items():
            if n > 0:
                lines.append(f"  {reason:<35} {n:>8,} rows removed")

        lines += [
            f"Columns    : {self._processed.shape[1]}",
            f"Exposure   : {self._processed[EXPOSURE_CNT].sum():,.0f} policies",
            f"Deaths     : {self._processed['Death_Count'].sum():,.0f}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Step 1: Read file
    # ------------------------------------------------------------------

    def _read_file(self) -> pd.DataFrame:
        """Read TSV/CSV/Parquet, preserving literal 'N/A' as a category."""
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")

        read_kw: dict = dict(low_memory=False, keep_default_na=False, na_filter=False)

        suffix = self.path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(self.path)
        elif suffix == ".csv":
            df = pd.read_csv(self.path, **read_kw)
        else:
            # Default: tab-delimited text
            df = pd.read_csv(self.path, sep=self.sep, **read_kw)

        # Treat empty strings as proper NaN (but never touch the literal "N/A")
        df = df.replace({"": np.nan})

        if self.verbose:
            logger.info("Read %d rows from '%s'.", len(df), self.path.name)
        return df

    # ------------------------------------------------------------------
    # Step 2: Study-scope exclusions
    # ------------------------------------------------------------------

    def _apply_exclusions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove records outside the study scope.

        Rules applied (in order)
        -------------------------
        1. ``SOA_Post_Lvl_Ind == 'PLT'`` – Post-level-term shock lapse
           period; distorts juvenile mortality trends.
        2. ``Issue_Age > JUVENILE_MAX_ISSUE_AGE`` – Restrict to issue
           ages 0–17 (juvenile study scope).
        3. ``Attained_Age > 94`` – Cap at age 94; age 95+ rows excluded.
        4. ``Smoker_Status != 'U'`` – Retain only unknown smoker status
           (standard for juvenile business where smoking is not collected).
        5. ``Insurance_Plan == 'Other'`` – Exclude miscellaneous plan type.
        """
        initial = len(df)

        # Rule 1: Exclude PLT post-level-term records
        if "SOA_Post_Lvl_Ind" in df.columns:
            mask_plt = df["SOA_Post_Lvl_Ind"].astype("string") == "PLT"
            n_plt = mask_plt.sum()
            df = df[~mask_plt].copy()
            self._exclusion_counts["SOA_Post_Lvl_Ind == 'PLT'"] = int(n_plt)

        # Rule 2: Keep only juvenile issue ages
        if "Issue_Age" in df.columns:
            mask_adult = df["Issue_Age"] > JUVENILE_MAX_ISSUE_AGE
            n_adult = mask_adult.sum()
            df = df[~mask_adult].copy()
            self._exclusion_counts[f"Issue_Age > {JUVENILE_MAX_ISSUE_AGE}"] = int(n_adult)

        # Rule 3: Exclude attained age > 94
        if "Attained_Age" in df.columns:
            mask_old = df["Attained_Age"] > 94
            n_old = mask_old.sum()
            df = df[~mask_old].copy()
            self._exclusion_counts["Attained_Age > 94"] = int(n_old)

        # Rule 4: Retain only unknown smoker status
        if "Smoker_Status" in df.columns:
            mask_smoker = df["Smoker_Status"].astype("string") != "U"
            n_smoker = mask_smoker.sum()
            df = df[~mask_smoker].copy()
            self._exclusion_counts["Smoker_Status != 'U'"] = int(n_smoker)

        # Rule 5: Exclude 'Other' insurance plan
        if "Insurance_Plan" in df.columns:
            mask_other = df["Insurance_Plan"].astype("string") == "Other"
            n_other = mask_other.sum()
            df = df[~mask_other].copy()
            self._exclusion_counts["Insurance_Plan == 'Other'"] = int(n_other)

        total_removed = initial - len(df)
        if self.verbose:
            logger.info(
                "Exclusions: %d rows removed (%d remaining).", total_removed, len(df)
            )
        return df

    # ------------------------------------------------------------------
    # Step 3: ANB → ALB conversion
    # ------------------------------------------------------------------

    def _convert_anb_to_alb(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Age Nearest Birthday (ANB) to Age Last Birthday (ALB).

        Actuarial convention
        --------------------
        The standard assumption is that, on average, individuals are
        observed at the midpoint of their age-nearest-birthday year:

            Age_ALB ≈ Age_ANB − 0.5

        This is applied to both ``Issue_Age`` and ``Attained_Age`` to
        produce continuous ALB equivalents used in GAM smoothing.

        New ``_mod`` columns hold the fractional equivalents (original − 0.5).
        """
        for col, new_col in [("Issue_Age", "Issue_Age_mod"), ("Attained_Age", "Attained_Age_mod")]:
            if col in df.columns:
                df[new_col] = df[col] - 0.5

        if self.verbose:
            logger.info("ANB → ALB conversion applied to Issue_Age and Attained_Age.")
        return df

    # ------------------------------------------------------------------
    # Step 4: Feature engineering
    # ------------------------------------------------------------------

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build derived actuarial grouping features.

        New columns
        -----------
        ``Attained_Age_Group``
            Quinquennial age bands: '0–4', '5–9', '10–14', '15–19', …
        ``Policy_Year_Group``
            Duration buckets: '1–5', '6–10', '11–15', '16+'.
        ``Is_Select``
            Boolean: True if Slct_Ult_Ind == 'S' (select period).
        """
        # --- Attained age quinquennial bands ---------------------------------
        if "Attained_Age" in df.columns:
            bins   = list(range(0, 95, 5)) + [95, np.inf]
            labels = (
                [f"{a}–{a+4}" for a in range(0, 95, 5)] + ["95+"]
            )
            df["Attained_Age_Group"] = pd.cut(
                df["Attained_Age"],
                bins=bins,
                labels=labels,
                right=False,
            )

        # --- Duration / policy-year groups -----------------------------------
        if "Duration" in df.columns:
            dur_bins   = list(range(0, 25, 5)) + [25, np.inf]
            dur_labels = [f"{a+1}–{a+5}" for a in range(0, 25, 5)] + ["26+"]
            df["Policy_Year_Group"] = pd.cut(
                df["Duration"],
                bins=dur_bins,
                labels=dur_labels,
                right=True,
            )

        # --- Select / Ultimate indicator -------------------------------------
        if "Slct_Ult_Ind" in df.columns:
            df["Is_Select"] = df["Slct_Ult_Ind"].astype("string") == "S"

        if self.verbose:
            logger.info(
                "Feature engineering complete. DataFrame now has %d columns.",
                df.shape[1],
            )
        return df
