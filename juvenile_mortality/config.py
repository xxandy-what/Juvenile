"""
config.py
=========
Shared constants, column specifications, and paths for the
Next-Generation SOA Juvenile Mortality Table pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).resolve().parent
DATA_DIR    = ROOT_DIR / "data"
OUTPUT_DIR  = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR  = OUTPUT_DIR / "tables"

DEFAULT_DATA_PATH = Path(r"C:\Users\haofe\PycharmProjects\Juvenile\Juvenile_cleaned.txt")

# ---------------------------------------------------------------------------
# Column specifications (sourced from ILEC 2012-2019 layout)
# ---------------------------------------------------------------------------
CAT_COLUMNS: list[str] = [
    "Age_Ind",
    "Sex",
    "Smoker_Status",
    "Insurance_Plan",
    "Face_Amount_Band",
    "SOA_Antp_Lvl_TP",
    "SOA_Guar_Lvl_TP",
    "SOA_Post_Lvl_Ind",
    "Slct_Ult_Ind",
    "Preferred_Indicator",
    "Number_of_Pfd_Classes",
    "Preferred_Class",
]

NUM_COLUMNS: list[str] = [
    "Observation_Year",
    "Issue_Age",
    "Duration",
    "Issue_Year",
    "Attained_Age",
    "Amount_Exposed",
    "Policies_Exposed",
    "Death_Claim_Amount",
    "Death_Count",
    "ExpDth_VBT2015_Cnt",
    "ExpDth_VBT2015_Amt",
]

# Columns required to compute A/E ratios
AE_REQUIRED: list[str] = [
    "Death_Count",
    "ExpDth_VBT2015_Cnt",
    "Death_Claim_Amount",
    "ExpDth_VBT2015_Amt",
]

# Exposure columns
EXPOSURE_CNT: str = "Policies_Exposed"
EXPOSURE_AMT: str = "Amount_Exposed"

# Face amount band lookup (code → human-readable label)
FACE_BAND_MAP: dict[str, str] = {
    "01": "01: 0–9,999",
    "02": "02: 10,000–24,999",
    "03": "03: 25,000–49,999",
    "04": "04: 50,000–99,999",
    "05": "05: 100,000–249,999",
    "06": "06: 250,000–499,999",
    "07": "07: 500,000–999,999",
    "08": "08: 1,000,000–2,499,999",
    "09": "09: 2,500,000–4,999,999",
    "10": "10: 5,000,000–9,999,999",
    "11": "11: 10,000,000+",
}

# Actuarial study scope
JUVENILE_MAX_ISSUE_AGE: int = 17   # issue ages 0–17
ADULT_MIN_ATTAINED_AGE: int = 18   # grading starts here

# VBT 2015 is the benchmark expected table
EXPECTED_CNT_COL: str = "ExpDth_VBT2015_Cnt"
EXPECTED_AMT_COL: str = "ExpDth_VBT2015_Amt"
ACTUAL_CNT_COL:   str = "Death_Count"
ACTUAL_AMT_COL:   str = "Death_Claim_Amount"

# Confidence interval level (two-sided)
CI_LEVEL: float = 0.95
