
import os
import re
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="SOA-style Mortality Explorer (DuckDB)", layout="wide")

# ---------------------------
# Simple styling
# ---------------------------
st.markdown(
    """
    <style>
    .app-header-band {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 55pt;
        background: white;
        z-index: 999;
    }

    .block-container {
        padding-top: calc(55pt + 1.2rem);
        padding-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {gap: 0.5rem;}
    .stTabs [data-baseweb="tab"] {padding-left: 0.7rem; padding-right: 0.7rem;}
    </style>

    <div class="app-header-band"></div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Dataset spec / dictionaries
# ---------------------------
CAT_COLUMNS = [
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

NUM_COLUMNS = [
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

ALL_KNOWN = set(CAT_COLUMNS + NUM_COLUMNS)

FRIENDLY = {
    "Observation_Year": "Observation Year",
    "Age_Ind": "Age Basis",
    "Sex": "Sex",
    "Smoker_Status": "Smoker Status",
    "Insurance_Plan": "Insurance Plan",
    "Issue_Age": "Issue Age",
    "Issue_Age_Group": "Issue Age Group",
    "Duration": "Duration",
    "Duration_Group": "Duration Group",
    "Face_Amount_Band": "Face Amount Band",
    "Face_Amount_Band_Label": "Face Amount Band",
    "Issue_Year": "Issue Year",
    "Attained_Age": "Attained Age",
    "Attained_Age_Group": "Attained Age Group",
    "SOA_Antp_Lvl_TP": "Anticipated Level Term Period",
    "SOA_Guar_Lvl_TP": "Guaranteed Level Term Period",
    "SOA_Post_Lvl_Ind": "Post Level Term Indicator",
    "Slct_Ult_Ind": "Select / Ultimate",
    "Preferred_Indicator": "Preferred Indicator",
    "Number_of_Pfd_Classes": "Number of Preferred Classes",
    "Preferred_Class": "Preferred Class",
    "Amount_Exposed": "Exposure - Amount",
    "Policies_Exposed": "Exposure - Policies",
    "Death_Claim_Amount": "Actual Deaths - Claim Amount",
    "Death_Count": "Actual Deaths - Count",
    "ExpDth_VBT2015_Cnt": "Expected Deaths (VBT2015) - Count",
    "ExpDth_VBT2015_Amt": "Expected Deaths (VBT2015) - Amount",
}

FIELD_NOTES = {
    "Observation_Year": "Calendar year of observation (2012-2019).",
    "Age_Ind": "0 = Age Nearest Birthday, 1 = Age Last Birthday, 2 = Age Next Birthday.",
    "Sex": "F = Female, M = Male.",
    "Smoker_Status": "N = Nonsmoker, S = Smoker, U = Uni-smoke. For Issue Year before 1981 the study sets smoker status to Uni-smoke.",
    "Insurance_Plan": "Perm, Term, UL, ULSG, VL, VLSG, Other.",
    "Issue_Age": "Age at issue.",
    "Duration": "Policy duration.",
    "Face_Amount_Band": "Band 01 to 11. For Observation Year before 2018 it uses base segment plus riders; for 2018 and later it is segment-level.",
    "Issue_Year": "Policy issue year.",
    "Attained_Age": "Current age at observation.",
    "SOA_Antp_Lvl_TP": "Anticipated level term period.",
    "SOA_Guar_Lvl_TP": "Guaranteed level term period.",
    "SOA_Post_Lvl_Ind": "N/A, NLT, PLT, ULT, WLT.",
    "Slct_Ult_Ind": "S = Select, U = Ultimate.",
    "Preferred_Indicator": "0 = Not Preferred, 1 = Preferred, U = Unknown.",
    "Number_of_Pfd_Classes": "Preferred class count by smoker grouping.",
    "Preferred_Class": "1, 2, 3, 4, NA, U.",
    "Amount_Exposed": "Exposure on an amount basis.",
    "Policies_Exposed": "Exposure on a policy-count basis.",
    "Death_Claim_Amount": "Actual death claim amount.",
    "Death_Count": "Actual death count.",
    "ExpDth_VBT2015_Cnt": "Expected deaths on count basis = XPO_C * q.",
    "ExpDth_VBT2015_Amt": "Expected deaths on amount basis = XPO_C * q * FA.",
}

AGE_IND_MAP = {
    "0": "0 = ANB",
    "1": "1 = ALB",
    "2": "2 = Age Next Birthday",
}
REVERSE_AGE_IND_MAP = {v: k for k, v in AGE_IND_MAP.items()}

FACE_BAND_MAP = {
    "01": "01: 0 - 9,999",
    "02": "02: 10,000 - 24,999",
    "03": "03: 25,000 - 49,999",
    "04": "04: 50,000 - 99,999",
    "05": "05: 100,000 - 249,999",
    "06": "06: 250,000 - 499,999",
    "07": "07: 500,000 - 999,999",
    "08": "08: 1,000,000 - 2,499,999",
    "09": "09: 2,500,000 - 4,999,999",
    "10": "10: 5,000,000 - 9,999,999",
    "11": "11: 10,000,000+",
}
REVERSE_FACE_BAND_MAP = {v: k for k, v in FACE_BAND_MAP.items()}
FACE_BAND_ORDER = list(FACE_BAND_MAP.values())

AE_OPTIONS = [
    "A/E (Count)",
    "A/E (Amount)",
    "A/E (Count, w/MI)",
    "A/E (Amount, w/MI)",
]

INT_FIELDS = {"Observation_Year", "Issue_Age", "Attained_Age", "Duration", "Issue_Year", "Death_Count"}

PAGE_SPECS = [
    ("Observation Year", "Observation_Year", "Observation_Year"),
    ("Issue Age", "Issue_Age", "Issue_Age"),
    ("Duration", "Duration", "Duration"),
    ("Attained Age", "Attained_Age", "Attained_Age"),
    ("Face Amount", "Face_Amount_Band_Label", "Face_Amount_Band"),
]

DEFAULT_FILTER_FIELDS = ["Smoker_Status", "SOA_Post_Lvl_Ind", "Insurance_Plan", "Attained_Age"]

GRAPH_FILTER_FIELDS = [
    "Observation_Year",
    "Sex",
    "Issue_Age_Group",
    "Duration_Group",
    "Attained_Age_Group",
    "Face_Amount_Band",
    "Preferred_Class",
    "Insurance_Plan",
]

DURATION_GROUP_OPTIONS = ["1", "2", "3", "4-5", "6-10", "11-15", "16-20", "21-25", "26+"]
DURATION_GROUP_ORDER = {v: i for i, v in enumerate(DURATION_GROUP_OPTIONS)}

ATTAINED_AGE_GROUP_OPTIONS = ["0"] + [f"{i}-{i+4}" for i in range(1, 95, 5)] + ["95+"]

# ---------------------------
# Paths
# ---------------------------
def builtin_data_candidates() -> List[str]:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    return [
        os.path.join(app_dir, "script", "data", "parquet", "full.parquet"),
        os.path.join(app_dir, "data", "parquet", "full.parquet"),
        os.path.join(cwd, "script", "data", "parquet", "full.parquet"),
        os.path.join(cwd, "data", "parquet", "full.parquet"),
    ]


def resolve_builtin_data_path() -> str:
    candidates = builtin_data_candidates()
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


# ---------------------------
# Helpers
# ---------------------------
def human(col: str) -> str:
    return FRIENDLY.get(col, col)


def safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    d2 = d.replace(0, np.nan)
    out = n / d2
    return out.replace([np.inf, -np.inf], np.nan)


def safe_filename(text: str) -> str:
    text = re.sub(r"[^\w\-.]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def sql_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return "NULL"
        return repr(float(value))
    return "'" + str(value).replace("'", "''") + "'"


def normalize_path_for_sql(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/").replace("'", "''")


def source_sql(path: str, nrows: Optional[int] = None) -> str:
    p = normalize_path_for_sql(path)
    lower = path.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        base = f"read_parquet('{p}')"
    elif lower.endswith(".csv"):
        base = f"read_csv_auto('{p}', header=true, sample_size=-1)"
    else:
        base = f"read_csv_auto('{p}', delim='\\t', header=true, sample_size=-1)"

    if nrows is not None and nrows > 0:
        return f"(SELECT * FROM {base} LIMIT {int(nrows)})"
    return base


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def label_display_value(col: str, value: Optional[str]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    value = str(value)
    if col == "Age_Ind":
        return AGE_IND_MAP.get(value, value)
    if col in {"Face_Amount_Band", "Face_Amount_Band_Label"}:
        return FACE_BAND_MAP.get(value, value)
    return value


def selected_display_to_raw_values(col: str, selected: List[str]) -> Tuple[bool, List[str]]:
    selected = [str(x) for x in selected]
    want_missing = "(Missing)" in selected
    raw_vals: List[str] = []
    for x in selected:
        if x == "(Missing)":
            continue
        if col == "Age_Ind":
            raw_vals.append(REVERSE_AGE_IND_MAP.get(x, x))
        elif col in {"Face_Amount_Band", "Face_Amount_Band_Label"}:
            raw_vals.append(REVERSE_FACE_BAND_MAP.get(x, x))
        else:
            raw_vals.append(x)
    raw_vals = list(dict.fromkeys(raw_vals))
    return want_missing, raw_vals


def freeze_cat_filters(cat_filters: Dict[str, List[str]]) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    items = []
    for col in sorted(cat_filters.keys()):
        items.append((col, tuple(cat_filters[col])))
    return tuple(items)


def freeze_num_ranges(num_ranges: Dict[str, Tuple[float, float]]) -> Tuple[Tuple[str, float, float], ...]:
    items = []
    for col in sorted(num_ranges.keys()):
        lo, hi = num_ranges[col]
        items.append((col, float(lo), float(hi)))
    return tuple(items)


def combine_category_filters(*filter_dicts: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for fd in filter_dicts:
        for col, vals in fd.items():
            if not vals:
                continue
            vals = list(dict.fromkeys([str(v) for v in vals]))
            if col not in merged:
                merged[col] = vals
            else:
                merged[col] = [v for v in merged[col] if v in set(vals)]
    return merged


def filter_value_expr_sql(col: str) -> str:
    if col == "Observation_Year":
        return f"CAST(TRY_CAST({sql_ident('Observation_Year')} AS BIGINT) AS VARCHAR)"
    if col == "Issue_Age_Group":
        ident = sql_ident("Issue_Age")
        num_expr = f"TRY_CAST({ident} AS BIGINT)"
        return (
            f"CASE "
            f"WHEN {num_expr} IS NULL THEN NULL "
            f"WHEN {num_expr} = 0 THEN '0' "
            f"WHEN {num_expr} >= 95 THEN '95+' "
            f"ELSE CAST(((CAST(FLOOR(({num_expr}-1) / 5.0) AS BIGINT) * 5) + 1) AS VARCHAR) "
            f"|| '-' || "
            f"CAST(((CAST(FLOOR(({num_expr}-1) / 5.0) AS BIGINT) * 5) + 5) AS VARCHAR) "
            f"END"
        )
    if col == "Duration_Group":
        ident = sql_ident("Duration")
        num_expr = f"TRY_CAST({ident} AS BIGINT)"
        return (
            f"CASE "
            f"WHEN {num_expr} IS NULL THEN NULL "
            f"WHEN {num_expr} = 1 THEN '1' "
            f"WHEN {num_expr} = 2 THEN '2' "
            f"WHEN {num_expr} = 3 THEN '3' "
            f"WHEN {num_expr} BETWEEN 4 AND 5 THEN '4-5' "
            f"WHEN {num_expr} BETWEEN 6 AND 10 THEN '6-10' "
            f"WHEN {num_expr} BETWEEN 11 AND 15 THEN '11-15' "
            f"WHEN {num_expr} BETWEEN 16 AND 20 THEN '16-20' "
            f"WHEN {num_expr} BETWEEN 21 AND 25 THEN '21-25' "
            f"WHEN {num_expr} >= 26 THEN '26+' "
            f"END"
        )
    if col == "Attained_Age_Group":
        ident = sql_ident("Attained_Age")
        num_expr = f"TRY_CAST({ident} AS BIGINT)"
        return (
            f"CASE "
            f"WHEN {num_expr} IS NULL THEN NULL "
            f"WHEN {num_expr} = 0 THEN '0' "
            f"WHEN {num_expr} >= 95 THEN '95+' "
            f"ELSE CAST(((CAST(FLOOR(({num_expr}-1) / 5.0) AS BIGINT) * 5) + 1) AS VARCHAR) "
            f"|| '-' || "
            f"CAST(((CAST(FLOOR(({num_expr}-1) / 5.0) AS BIGINT) * 5) + 5) AS VARCHAR) "
            f"END"
        )
    if col == "Face_Amount_Band_Label":
        return f"CAST({sql_ident('Face_Amount_Band')} AS VARCHAR)"
    raw_col = "Face_Amount_Band" if col == "Face_Amount_Band_Label" else col
    return f"CAST({sql_ident(raw_col)} AS VARCHAR)"


def build_where_sql(
    cat_filters_frozen: Tuple[Tuple[str, Tuple[str, ...]], ...],
    num_ranges_frozen: Tuple[Tuple[str, float, float], ...],
) -> str:
    clauses: List[str] = []

    for col, selected in cat_filters_frozen:
        want_missing, raw_values = selected_display_to_raw_values(col, list(selected))
        expr = filter_value_expr_sql(col)
        if raw_values and want_missing:
            values_sql = ", ".join(sql_literal(v) for v in raw_values)
            clauses.append(f"(({expr}) IS NULL OR ({expr}) IN ({values_sql}))")
        elif raw_values:
            values_sql = ", ".join(sql_literal(v) for v in raw_values)
            clauses.append(f"({expr}) IN ({values_sql})")
        elif want_missing:
            clauses.append(f"({expr}) IS NULL")

    for col, lo, hi in num_ranges_frozen:
        ident = sql_ident(col)
        clauses.append(f"TRY_CAST({ident} AS DOUBLE) BETWEEN {float(lo)} AND {float(hi)}")

    return (" WHERE " + " AND ".join(clauses)) if clauses else ""


def build_filtered_filename(source_name: str, row_count: int, num_ranges: Dict[str, Tuple[float, float]]) -> str:
    base = os.path.splitext(source_name)[0]
    parts = [base, "filtered"]
    if "Observation_Year" in num_ranges:
        lo, hi = num_ranges["Observation_Year"]
        if int(lo) == int(hi):
            parts.append(f"year_{int(lo)}")
        else:
            parts.append(f"years_{int(lo)}_{int(hi)}")
    parts.append(f"rows_{row_count}")
    return safe_filename("_".join(parts) + ".csv")


def numeric_fields_present(available_columns: List[str]) -> List[str]:
    return [c for c in NUM_COLUMNS if c in available_columns]


def category_fields_present(available_columns: List[str]) -> List[str]:
    return [c for c in CAT_COLUMNS if c in available_columns]


def dim_select_expr(col: str) -> str:
    ident = sql_ident(col if col != "Face_Amount_Band_Label" else "Face_Amount_Band")
    alias = sql_ident(col)
    if col == "Age_Ind":
        expr = (
            f"CASE CAST({ident} AS VARCHAR) "
            f"WHEN '0' THEN '0 = ANB' "
            f"WHEN '1' THEN '1 = ALB' "
            f"WHEN '2' THEN '2 = Age Next Birthday' "
            f"ELSE CAST({ident} AS VARCHAR) END"
        )
        return f"{expr} AS {alias}"
    if col in {"Face_Amount_Band", "Face_Amount_Band_Label"}:
        parts = [f"WHEN '{k}' THEN {sql_literal(v)}" for k, v in FACE_BAND_MAP.items()]
        expr = f"CASE CAST({ident} AS VARCHAR) {' '.join(parts)} ELSE CAST({ident} AS VARCHAR) END"
        return f"{expr} AS {alias}"
    return f"CAST({ident} AS VARCHAR) AS {alias}" if col in CAT_COLUMNS else f"{ident} AS {alias}"


def axis_exprs(base_x_field: str, bin_size: int) -> Tuple[str, str]:
    raw_field = "Face_Amount_Band" if base_x_field == "Face_Amount_Band_Label" else base_x_field
    ident = sql_ident(raw_field)

    if base_x_field == "Face_Amount_Band_Label" or base_x_field == "Face_Amount_Band":
        sort_parts = [f"WHEN '{k}' THEN {i}" for i, k in enumerate(FACE_BAND_MAP.keys())]
        label_parts = [f"WHEN '{k}' THEN {sql_literal(v)}" for k, v in FACE_BAND_MAP.items()]
        sort_expr = f"CASE CAST({ident} AS VARCHAR) {' '.join(sort_parts)} ELSE 10000 END"
        label_expr = f"CASE CAST({ident} AS VARCHAR) {' '.join(label_parts)} ELSE CAST({ident} AS VARCHAR) END"
        return label_expr, sort_expr

    if base_x_field in NUM_COLUMNS:
        num_expr = f"TRY_CAST({ident} AS DOUBLE)"
        if bin_size <= 1:
            if base_x_field in INT_FIELDS:
                label_expr = f"CASE WHEN {ident} IS NULL THEN NULL ELSE CAST(TRY_CAST({ident} AS BIGINT) AS VARCHAR) END"
            else:
                label_expr = f"CASE WHEN {ident} IS NULL THEN NULL ELSE CAST({num_expr} AS VARCHAR) END"
            sort_expr = num_expr
            return label_expr, sort_expr

        start_expr = f"CAST(FLOOR({num_expr} / {int(bin_size)}) * {int(bin_size)} AS BIGINT)"
        label_expr = (
            f"CASE WHEN {ident} IS NULL THEN NULL "
            f"ELSE CAST({start_expr} AS VARCHAR) || '-' || CAST(({start_expr} + {int(bin_size) - 1}) AS VARCHAR) END"
        )
        sort_expr = start_expr
        return label_expr, sort_expr

    return dim_select_expr(base_x_field).rsplit(" AS ", 1)[0], dim_select_expr(base_x_field).rsplit(" AS ", 1)[0]


def split_expr(col: Optional[str], alias: str) -> str:
    if not col:
        return f"CAST(NULL AS VARCHAR) AS {sql_ident(alias)}"
    expr = dim_select_expr(col)
    base_expr = expr.rsplit(" AS ", 1)[0]
    return f"{base_expr} AS {sql_ident(alias)}"


def ae_requirements(metric: str) -> Tuple[str, str]:
    if metric == "A/E (Count)":
        return "Death_Count", "ExpDth_VBT2015_Cnt"
    if metric == "A/E (Amount)":
        return "Death_Claim_Amount", "ExpDth_VBT2015_Amt"
    raise KeyError(metric)


def natural_text_sort_key(value: str) -> Tuple:
    text = str(value)
    if text == "(Missing)":
        return (999999, text)
    try:
        return (int(text), "")
    except Exception:
        pass
    pieces = re.split(r"(\d+)", text)
    out = []
    for piece in pieces:
        if piece.isdigit():
            out.append(int(piece))
        else:
            out.append(piece.lower())
    return (0, *out)


def age_group_sort_key(value: str) -> Tuple[int, int]:
    text = str(value)
    if text == "(Missing)":
        return (999, 0)
    if text == "0":
        return (0, 0)
    if text.endswith("+"):
        try:
            return (int(text[:-1]), 1)
        except Exception:
            return (998, 0)
    if "-" in text:
        try:
            start = int(text.split("-")[0])
            return (start, 0)
        except Exception:
            return (998, 0)
    return (998, 0)


def sort_display_values(col: str, values: List[str]) -> List[str]:
    vals = [str(v) for v in values]
    if col in {"Face_Amount_Band", "Face_Amount_Band_Label"}:
        order = {v: i for i, v in enumerate(FACE_BAND_ORDER)}
        return sorted(vals, key=lambda x: (999 if x == "(Missing)" else order.get(x, 998), x))
    if col == "Observation_Year":
        return sorted(vals, key=natural_text_sort_key)
    if col in {"Issue_Age_Group", "Attained_Age_Group"}:
        return sorted(vals, key=age_group_sort_key)
    if col == "Duration_Group":
        return sorted(vals, key=lambda x: (999 if x == "(Missing)" else DURATION_GROUP_ORDER.get(x, 998), x))
    if col == "Preferred_Class":
        pref_order = {"1": 1, "2": 2, "3": 3, "4": 4, "NA": 98, "U": 99}
        return sorted(vals, key=lambda x: (999 if x == "(Missing)" else pref_order.get(x, 97), x))
    return sorted(vals, key=natural_text_sort_key)


def ensure_multiselect_state(key: str, options: List[str], default_values: Optional[List[str]] = None) -> None:
    valid_options = [str(x) for x in options]
    default_values = valid_options if default_values is None else [str(x) for x in default_values if str(x) in valid_options]
    current = st.session_state.get(key)
    if current is None:
        st.session_state[key] = default_values
        return
    current_valid = [str(x) for x in current if str(x) in valid_options]
    st.session_state[key] = current_valid if current_valid else default_values


def build_trace_labels(df: pd.DataFrame, trace_cols: List[str]) -> pd.Series:
    trace_cols = unique_preserve_order([c for c in trace_cols if c in df.columns])
    if not trace_cols:
        return pd.Series(["Series"] * len(df), index=df.index, dtype="string")

    trace_df = df.loc[:, trace_cols].copy()
    for col in trace_df.columns:
        trace_df[col] = trace_df[col].fillna("(Missing)").astype(str)

    if trace_df.shape[1] == 1:
        return trace_df.iloc[:, 0].astype("string")

    return trace_df.agg(lambda row: " | ".join(row.astype(str)), axis=1).astype("string")


@st.cache_data(show_spinner=False)
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------
# DuckDB queries
# ---------------------------
@st.cache_data(show_spinner=False)
def describe_source(path: str, nrows: Optional[int]) -> pd.DataFrame:
    con = duckdb.connect()
    src = source_sql(path, nrows)
    return con.execute(f"DESCRIBE SELECT * FROM {src}").df()


@st.cache_data(show_spinner=False)
def row_count_source(path: str, nrows: Optional[int]) -> int:
    con = duckdb.connect()
    src = source_sql(path, nrows)
    return int(con.execute(f"SELECT COUNT(*) AS n FROM {src}").fetchone()[0])


@st.cache_data(show_spinner=False)
def category_options_query(path: str, nrows: Optional[int], col: str) -> List[str]:
    con = duckdb.connect()
    src = source_sql(path, nrows)
    expr = filter_value_expr_sql(col)
    sql = f"SELECT DISTINCT {expr} AS v FROM {src} ORDER BY 1"
    vals = [row[0] for row in con.execute(sql).fetchall() if row[0] is not None]
    display = [label_display_value(col, v) if col in {'Age_Ind', 'Face_Amount_Band', 'Face_Amount_Band_Label'} else str(v) for v in vals]
    display = sort_display_values(col, list(dict.fromkeys(display)))
    null_count = con.execute(f"SELECT COUNT(*) FROM {src} WHERE ({expr}) IS NULL").fetchone()[0]
    if null_count > 0:
        return ["(Missing)"] + display
    return display


@st.cache_data(show_spinner=False)
def category_options_filtered_query(
    path: str,
    nrows: Optional[int],
    col: str,
    cat_filters_frozen: Tuple[Tuple[str, Tuple[str, ...]], ...],
    num_ranges_frozen: Tuple[Tuple[str, float, float], ...],
) -> List[str]:
    con = duckdb.connect()
    src = source_sql(path, nrows)
    where_sql = build_where_sql(cat_filters_frozen, num_ranges_frozen)
    expr = filter_value_expr_sql(col)
    sql = f"SELECT DISTINCT {expr} AS v FROM {src}{where_sql} ORDER BY 1"
    vals = [row[0] for row in con.execute(sql).fetchall() if row[0] is not None]
    display = [label_display_value(col, v) if col in {'Age_Ind', 'Face_Amount_Band', 'Face_Amount_Band_Label'} else str(v) for v in vals]
    display = sort_display_values(col, list(dict.fromkeys(display)))
    null_count = con.execute(f"SELECT COUNT(*) FROM {src}{where_sql} AND ({expr}) IS NULL" if where_sql else f"SELECT COUNT(*) FROM {src} WHERE ({expr}) IS NULL").fetchone()[0]
    if null_count > 0:
        return ["(Missing)"] + display
    return display


@st.cache_data(show_spinner=False)
def numeric_bounds_query(path: str, nrows: Optional[int], col: str) -> Tuple[Optional[float], Optional[float]]:
    con = duckdb.connect()
    src = source_sql(path, nrows)
    ident = sql_ident(col)
    sql = f"SELECT MIN(TRY_CAST({ident} AS DOUBLE)), MAX(TRY_CAST({ident} AS DOUBLE)) FROM {src} WHERE TRY_CAST({ident} AS DOUBLE) IS NOT NULL"
    lo, hi = con.execute(sql).fetchone()
    return (None if lo is None else float(lo), None if hi is None else float(hi))


@st.cache_data(show_spinner=False)
def filtered_row_count_query(
    path: str,
    nrows: Optional[int],
    cat_filters_frozen: Tuple[Tuple[str, Tuple[str, ...]], ...],
    num_ranges_frozen: Tuple[Tuple[str, float, float], ...],
) -> int:
    con = duckdb.connect()
    src = source_sql(path, nrows)
    where_sql = build_where_sql(cat_filters_frozen, num_ranges_frozen)
    return int(con.execute(f"SELECT COUNT(*) AS n FROM {src}{where_sql}").fetchone()[0])


@st.cache_data(show_spinner=False)
def filtered_preview_query(
    path: str,
    nrows: Optional[int],
    preview_cols: Tuple[str, ...],
    limit: int,
    cat_filters_frozen: Tuple[Tuple[str, Tuple[str, ...]], ...],
    num_ranges_frozen: Tuple[Tuple[str, float, float], ...],
) -> pd.DataFrame:
    con = duckdb.connect()
    src = source_sql(path, nrows)
    cols = list(preview_cols) if preview_cols else ["*"]
    select_sql = ", ".join(sql_ident(c) for c in cols) if cols != ["*"] else "*"
    where_sql = build_where_sql(cat_filters_frozen, num_ranges_frozen)
    df = con.execute(f"SELECT {select_sql} FROM {src}{where_sql} LIMIT {int(limit)}").df()

    for col in ["Age_Ind", "Face_Amount_Band"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: label_display_value(col, x) if pd.notna(x) else x)
    return df


@st.cache_data(show_spinner=False)
def filtered_download_query(
    path: str,
    nrows: Optional[int],
    limit: int,
    cat_filters_frozen: Tuple[Tuple[str, Tuple[str, ...]], ...],
    num_ranges_frozen: Tuple[Tuple[str, float, float], ...],
) -> pd.DataFrame:
    con = duckdb.connect()
    src = source_sql(path, nrows)
    where_sql = build_where_sql(cat_filters_frozen, num_ranges_frozen)
    df = con.execute(f"SELECT * FROM {src}{where_sql} LIMIT {int(limit)}").df()
    for col in ["Age_Ind", "Face_Amount_Band"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: label_display_value(col, x) if pd.notna(x) else x)
    return df


@st.cache_data(show_spinner=False)
def build_pivot_base_duckdb(
    path: str,
    nrows: Optional[int],
    dims: Tuple[str, ...],
    values: Tuple[str, ...],
    aggfunc: str,
    add_measures: bool,
    cat_filters_frozen: Tuple[Tuple[str, Tuple[str, ...]], ...],
    num_ranges_frozen: Tuple[Tuple[str, float, float], ...],
) -> pd.DataFrame:
    con = duckdb.connect()
    src = source_sql(path, nrows)
    where_sql = build_where_sql(cat_filters_frozen, num_ranges_frozen)

    dims_list = list(dims)
    if dims_list:
        dim_selects = [dim_select_expr(c) for c in dims_list]
        group_by = ", ".join(str(i + 1) for i in range(len(dim_selects)))
    else:
        dim_selects = ["'All' AS \"_ALL_\""]
        dims_list = ["_ALL_"]
        group_by = "1"

    agg_map = {
        "sum": "SUM",
        "mean": "AVG",
        "count": "COUNT",
        "min": "MIN",
        "max": "MAX",
    }
    chosen = agg_map.get(aggfunc, "SUM")

    metric_selects: List[str] = []
    value_fields = unique_preserve_order(list(values))
    for v in value_fields:
        ident = sql_ident(v)
        alias = sql_ident(v)
        if aggfunc == "nunique":
            metric_selects.append(f"COUNT(DISTINCT {ident}) AS {alias}")
        else:
            metric_selects.append(f"{chosen}({ident}) AS {alias}")

    helper_aliases: Dict[str, str] = {}
    if add_measures:
        raw_measure_cols = [
            "Death_Count",
            "Death_Claim_Amount",
            "ExpDth_VBT2015_Cnt",
            "ExpDth_VBT2015_Amt",
        ]
        for v in raw_measure_cols:
            helper_alias = f"__ae__{v}"
            helper_aliases[v] = helper_alias
            metric_selects.append(f"SUM({sql_ident(v)}) AS {sql_ident(helper_alias)}")

    if not metric_selects:
        return pd.DataFrame()

    sql = (
        "SELECT " + ", ".join(dim_selects + metric_selects) +
        f" FROM {src}{where_sql} GROUP BY {group_by}"
    )
    g = con.execute(sql).df()

    if add_measures:
        if {"Death_Count", "ExpDth_VBT2015_Cnt"}.issubset(helper_aliases):
            g["A/E (Count)"] = safe_div(g[helper_aliases["Death_Count"]], g[helper_aliases["ExpDth_VBT2015_Cnt"]])
        if {"Death_Claim_Amount", "ExpDth_VBT2015_Amt"}.issubset(helper_aliases):
            g["A/E (Amount)"] = safe_div(g[helper_aliases["Death_Claim_Amount"]], g[helper_aliases["ExpDth_VBT2015_Amt"]])
        helper_cols = [alias for alias in helper_aliases.values() if alias in g.columns]
        if helper_cols:
            g = g.drop(columns=helper_cols)

    return g


@st.cache_data(show_spinner=False)
def summarize_for_chart_duckdb(
    path: str,
    nrows: Optional[int],
    base_x_field: str,
    bin_size: int,
    bar_metrics: Tuple[str, ...],
    line_metrics: Tuple[str, ...],
    split1: Optional[str],
    split2: Optional[str],
    keep_split1: Tuple[str, ...],
    keep_split2: Tuple[str, ...],
    non_ae_agg: str,
    top_n_traces: int,
    cat_filters_frozen: Tuple[Tuple[str, Tuple[str, ...]], ...],
    num_ranges_frozen: Tuple[Tuple[str, float, float], ...],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    con = duckdb.connect()
    src = source_sql(path, nrows)

    merged_cat = {k: list(v) for k, v in cat_filters_frozen}
    if split1 and keep_split1:
        merged_cat = combine_category_filters(merged_cat, {split1: list(keep_split1)})
    if split2 and keep_split2:
        merged_cat = combine_category_filters(merged_cat, {split2: list(keep_split2)})

    where_sql = build_where_sql(freeze_cat_filters(merged_cat), num_ranges_frozen)
    axis_label_expr, axis_sort_expr = axis_exprs(base_x_field, bin_size)
    split1_sql = split_expr(split1, "_split1")
    split2_sql = split_expr(split2, "_split2")

    # Bars
    bars = pd.DataFrame(columns=["_axis_value", "_axis_sort"])
    bar_metrics_list = [m for m in unique_preserve_order(list(bar_metrics)) if m in NUM_COLUMNS]
    if bar_metrics_list:
        bar_selects = [f"SUM({sql_ident(m)}) AS {sql_ident(m)}" for m in bar_metrics_list]
        bar_sql = (
            f"SELECT {axis_label_expr} AS \"_axis_value\", {axis_sort_expr} AS \"_axis_sort\", "
            + ", ".join(bar_selects)
            + f" FROM {src}{where_sql} GROUP BY 1, 2 ORDER BY 2, 1"
        )
        bars = con.execute(bar_sql).df()

    # Lines grouped in DuckDB, reshaped in pandas
    line_metrics_list = unique_preserve_order(list(line_metrics))
    ae_metrics = [m for m in line_metrics_list if m in AE_OPTIONS]
    other_metrics = [m for m in line_metrics_list if m not in AE_OPTIONS and m in NUM_COLUMNS]

    line_needed_raw: List[str] = []
    for m in ae_metrics:
        ncol, dcol = ae_requirements(m)
        line_needed_raw.extend([ncol, dcol])
    line_needed_raw = sorted(set(line_needed_raw))

    agg_func = "AVG" if non_ae_agg == "mean" else "SUM"
    line_selects: List[str] = []
    for raw_col in line_needed_raw:
        line_selects.append(f"SUM({sql_ident(raw_col)}) AS {sql_ident('raw__' + raw_col)}")
    for m in other_metrics:
        line_selects.append(f"{agg_func}({sql_ident(m)}) AS {sql_ident('line__' + m)}")

    lines = pd.DataFrame(columns=["_axis_value", "_axis_sort", "_split1", "_split2", "_line_metric", "_value", "_support", "_trace_label"])
    if line_selects:
        line_sql = (
            f"SELECT {axis_label_expr} AS \"_axis_value\", {axis_sort_expr} AS \"_axis_sort\", {split1_sql}, {split2_sql}, "
            + ", ".join(line_selects)
            + f" FROM {src}{where_sql} GROUP BY 1, 2, 3, 4"
        )
        g = con.execute(line_sql).df()
        long_parts: List[pd.DataFrame] = []

        for metric in ae_metrics:
            ncol, dcol = ae_requirements(metric)
            num_name = f"raw__{ncol}"
            den_name = f"raw__{dcol}"
            if num_name not in g.columns or den_name not in g.columns:
                continue
            tmp = g[["_axis_value", "_axis_sort", "_split1", "_split2"]].copy()
            tmp["_line_metric"] = metric
            tmp["_value"] = safe_div(g[num_name], g[den_name])
            tmp["_support"] = g[num_name].fillna(0)
            long_parts.append(tmp)

        for metric in other_metrics:
            val_name = f"line__{metric}"
            if val_name not in g.columns:
                continue
            tmp = g[["_axis_value", "_axis_sort", "_split1", "_split2"]].copy()
            tmp["_line_metric"] = metric
            tmp["_value"] = g[val_name]
            tmp["_support"] = g[val_name].abs().fillna(0)
            long_parts.append(tmp)

        if long_parts:
            lines = pd.concat(long_parts, ignore_index=True)
            trace_cols = ["_line_metric"]
            if split1:
                trace_cols.append("_split1")
            if split2:
                trace_cols.append("_split2")
            lines["_trace_label"] = build_trace_labels(lines, trace_cols)
            support = lines.groupby("_trace_label", observed=True)["_support"].sum().sort_values(ascending=False)
            keep_traces = support.head(max(int(top_n_traces), 1)).index.tolist()
            lines = lines[lines["_trace_label"].isin(keep_traces)].copy()
            lines = lines.sort_values(["_axis_sort", "_axis_value", "_trace_label"]).reset_index(drop=True)

    axis_order: List[str]
    if not bars.empty:
        axis_order = bars["_axis_value"].astype(str).tolist()
    elif not lines.empty:
        axis_order = lines[["_axis_value", "_axis_sort"]].drop_duplicates().sort_values(["_axis_sort", "_axis_value"])["_axis_value"].astype(str).tolist()
    else:
        axis_order = []

    return bars, lines, axis_order


# ---------------------------
# Plotting / pivot shape helpers
# ---------------------------
def infer_bar_axis_label(bar_cols: List[str]) -> str:
    if any(c in ("Death_Claim_Amount", "ExpDth_VBT2015_Amt", "Amount_Exposed") for c in bar_cols):
        return "Bars (amount totals)"
    if any(c in ("Death_Count", "ExpDth_VBT2015_Cnt", "Policies_Exposed") for c in bar_cols):
        return "Bars (count / exposure totals)"
    return "Bars"


def infer_line_axis_label(line_metrics: List[str]) -> str:
    if line_metrics and all(m.startswith("A/E") for m in line_metrics):
        return "Lines (A/E ratio)"
    return "Lines"


def make_visual_figure(
    bars: pd.DataFrame,
    lines: pd.DataFrame,
    axis_order: List[str],
    bar_metrics: List[str],
    line_metrics: List[str],
    chart_mode: str,
    line_mode: str,
    title: str,
    x_title: str,
    height: int,
) -> go.Figure:
    use_secondary = chart_mode == "Combo"
    fig = make_subplots(specs=[[{"secondary_y": use_secondary}]])

    if chart_mode in ("Combo", "Bars only") and not bars.empty:
        for metric in bar_metrics:
            if metric not in bars.columns:
                continue
            fig.add_trace(
                go.Bar(
                    x=bars["_axis_value"].astype(str),
                    y=bars[metric],
                    name=human(metric),
                ),
                secondary_y=False,
            )

    if chart_mode in ("Combo", "Lines only") and not lines.empty:
        mode_map = {
            "Lines + Markers": "lines+markers",
            "Lines": "lines",
            "Markers": "markers",
        }
        plot_mode = mode_map.get(line_mode, "lines+markers")
        for trace_name in lines["_trace_label"].dropna().astype(str).unique().tolist():
            sub = lines[lines["_trace_label"].astype(str) == trace_name].copy()
            sub = sub.sort_values(["_axis_sort", "_axis_value"])
            fig.add_trace(
                go.Scatter(
                    x=sub["_axis_value"].astype(str),
                    y=sub["_value"],
                    mode=plot_mode,
                    name=trace_name,
                ),
                secondary_y=use_secondary,
            )

    fig.update_layout(
        title=title,
        hovermode="x unified",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
    )
    fig.update_xaxes(title_text=x_title, categoryorder="array", categoryarray=axis_order, tickangle=-25)

    if chart_mode == "Combo":
        fig.update_yaxes(title_text=infer_bar_axis_label(bar_metrics), secondary_y=False, tickformat=",.4g")
        fig.update_yaxes(title_text=infer_line_axis_label(line_metrics), secondary_y=True, tickformat=".4f")
    elif chart_mode == "Bars only":
        fig.update_yaxes(title_text=infer_bar_axis_label(bar_metrics), tickformat=",.4g")
    else:
        fig.update_yaxes(title_text=infer_line_axis_label(line_metrics), tickformat=".4f")

    return fig


def pivot_table_from_base(base: pd.DataFrame, row_fields: List[str], col_fields: List[str], metric_fields: List[str]) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    if not row_fields:
        base = base.copy()
        base["_Rows"] = "All"
        row_fields = ["_Rows"]
    if not col_fields:
        return base[row_fields + metric_fields].set_index(row_fields)
    return pd.pivot_table(
        base,
        index=row_fields,
        columns=col_fields,
        values=metric_fields,
        aggfunc="first",
        dropna=False,
        observed=True,
    )


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [" | ".join([str(x) for x in tup if str(x) != ""]) for tup in out.columns.to_flat_index()]
    else:
        out.columns = [str(c) for c in out.columns]
    return out


def default_filter_config(path: str, nrows: Optional[int], available_columns: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, Tuple[float, float]]]:
    default_cat_filters: Dict[str, List[str]] = {}
    default_num_ranges: Dict[str, Tuple[float, float]] = {}

    if "Smoker_Status" in available_columns:
        opts = category_options_query(path, nrows, "Smoker_Status")
        if "U" in opts:
            default_cat_filters["Smoker_Status"] = ["U"]

    if "SOA_Post_Lvl_Ind" in available_columns:
        opts = category_options_query(path, nrows, "SOA_Post_Lvl_Ind")
        default_cat_filters["SOA_Post_Lvl_Ind"] = [x for x in opts if x not in {"PLT", "(Missing)"}] or [x for x in opts if x != "PLT"]

    if "Insurance_Plan" in available_columns:
        opts = category_options_query(path, nrows, "Insurance_Plan")
        default_cat_filters["Insurance_Plan"] = [x for x in opts if x != "Other"]

    if "Attained_Age" in available_columns:
        lo, hi = numeric_bounds_query(path, nrows, "Attained_Age")
        if lo is not None and hi is not None:
            default_num_ranges["Attained_Age"] = (float(lo), float(min(hi, 94)))

    return default_cat_filters, default_num_ranges


# ---------------------------
# UI renderers
# ---------------------------
def render_welcome_tab(source_name: str, data_path: str, available_columns: List[str]) -> None:
    st.title("SOA-style Mortality Experience Explorer")

    st.markdown(
        """
        **Basic flow**
        1. Use **Preliminary filters** to define the working data.
        2. Use **Pivot table** for cross-tab summaries and optional A/E output.
        3. Use the five study tabs to build charts.
        4. In each study tab, use the right-side dropdown filters to refine the chart further without changing the preliminary working set.
        """
    )

    st.markdown(
        """
        **Default working filters**
        - `Smoker_Status = U`
        - `SOA_Post_Lvl_Ind` excludes `PLT`
        - `Insurance_Plan` excludes `Other`
        - `Attained_Age <= 94`
        """
    )

    st.markdown(
        """
        **Notes**
        - This version queries the built-in file on demand with DuckDB instead of loading the full dataset into pandas at startup.
        - Parquet is the preferred format for stability and speed in public deployment.
        - The graph tabs now support additional dropdown filters layered on top of the preliminary filters.
        """
    )

    st.write(f"**Built-in file:** `{source_name}`")
    st.write(f"**Resolved path:** `{data_path}`")

    with st.expander("Field reference", expanded=False):
        dict_rows = [{"Field": c, "Friendly": human(c), "Note": FIELD_NOTES.get(c, "")} for c in available_columns]
        st.dataframe(pd.DataFrame(dict_rows), use_container_width=True, hide_index=True, height=420)


def render_preliminary_filters_tab(
    path: str,
    nrows: Optional[int],
    available_columns: List[str],
    source_name: str,
    default_cat_filters: Dict[str, List[str]],
    default_num_ranges: Dict[str, Tuple[float, float]],
) -> None:
    st.caption("These filters define the working dataset used by the pivot table and the five study tabs.")

    candidate_filter_cols = [c for c in available_columns if c in ALL_KNOWN]
    default_filter_cols = [c for c in DEFAULT_FILTER_FIELDS if c in candidate_filter_cols]

    if "working_filter_cols" not in st.session_state:
        st.session_state["working_filter_cols"] = default_filter_cols

    filter_cols = st.multiselect(
        "Fields to filter",
        options=candidate_filter_cols,
        default=st.session_state.get("working_filter_cols", default_filter_cols),
        key="working_filter_cols",
    )

    with st.form("preliminary_filters_form"):
        cat_filters: Dict[str, List[str]] = {}
        num_ranges: Dict[str, Tuple[float, float]] = {}

        current_cat = st.session_state.get("active_cat_filters", default_cat_filters)
        current_num = st.session_state.get("active_num_ranges", default_num_ranges)

        for col in filter_cols:
            with st.expander(human(col), expanded=False):
                if col in NUM_COLUMNS:
                    lo, hi = numeric_bounds_query(path, nrows, col)
                    if lo is None or hi is None:
                        st.caption("No numeric values available.")
                        continue

                    current_val = current_num.get(col, (float(lo), float(hi)))
                    current_lo = max(float(lo), float(current_val[0]))
                    current_hi = min(float(hi), float(current_val[1]))
                    if current_lo > current_hi:
                        current_lo, current_hi = float(lo), float(hi)

                    if col in INT_FIELDS:
                        lo_i, hi_i = int(lo), int(hi)
                        val = st.slider(
                            f"{human(col)} range",
                            min_value=lo_i,
                            max_value=hi_i,
                            value=(int(round(current_lo)), int(round(current_hi))),
                            step=1,
                            key=f"pre_num_{col}",
                        )
                        num_ranges[col] = (float(val[0]), float(val[1]))
                    else:
                        val = st.slider(
                            f"{human(col)} range",
                            min_value=float(lo),
                            max_value=float(hi),
                            value=(float(current_lo), float(current_hi)),
                            key=f"pre_num_{col}",
                        )
                        num_ranges[col] = (float(val[0]), float(val[1]))
                else:
                    opts = category_options_query(path, nrows, col)
                    default_selected = current_cat.get(col, opts)
                    default_selected = [x for x in default_selected if x in opts] or opts
                    cat_filters[col] = st.multiselect(
                        f"{human(col)} values",
                        options=opts,
                        default=default_selected,
                        key=f"pre_cat_{col}",
                    )

        st.markdown(
            """
            Default screen applied here:
            - Smoker Status = U
            - Post Level Term Indicator excludes PLT
            - Insurance Plan excludes Other
            - Attained Age capped at 94
            """
        )

        a, b = st.columns([1, 1])
        apply_clicked = a.form_submit_button("Apply filters", type="primary")
        reset_clicked = b.form_submit_button("Reset to default filters")

    if "active_cat_filters" not in st.session_state:
        st.session_state["active_cat_filters"] = default_cat_filters
        st.session_state["active_num_ranges"] = default_num_ranges

    if reset_clicked:
        st.session_state["working_filter_cols"] = default_filter_cols
        st.session_state["active_cat_filters"] = default_cat_filters
        st.session_state["active_num_ranges"] = default_num_ranges
        st.session_state.pop("working_download_df", None)
        st.session_state["pivot_table_duck"] = pd.DataFrame()
        for label, _, _ in PAGE_SPECS:
            st.session_state.pop(f"chart_output_{label}", None)
            st.session_state.pop(f"chart_error_{label}", None)
    elif apply_clicked:
        st.session_state["active_cat_filters"] = cat_filters
        st.session_state["active_num_ranges"] = num_ranges
        st.session_state.pop("working_download_df", None)
        st.session_state["pivot_table_duck"] = pd.DataFrame()
        for label, _, _ in PAGE_SPECS:
            st.session_state.pop(f"chart_output_{label}", None)
            st.session_state.pop(f"chart_error_{label}", None)

    active_cat_filters = st.session_state.get("active_cat_filters", {})
    active_num_ranges = st.session_state.get("active_num_ranges", {})

    cat_frozen = freeze_cat_filters(active_cat_filters)
    num_frozen = freeze_num_ranges(active_num_ranges)

    working_count = filtered_row_count_query(path, nrows, cat_frozen, num_frozen)

    summary_parts: List[str] = []
    for col, vals in active_cat_filters.items():
        if vals:
            summary_parts.append(f"{human(col)}: {len(vals)} selected" if len(vals) > 4 else f"{human(col)}: {', '.join(vals)}")
    for col, (lo, hi) in active_num_ranges.items():
        summary_parts.append(f"{human(col)}: {lo:g} to {hi:g}")

    if summary_parts:
        st.caption("Applied filters: " + "  •  ".join(summary_parts))

    if working_count == 0:
        st.error("No rows are left after the current preliminary filters.")
        return

    with st.expander("Preview working dataset", expanded=False):
        default_preview_cols = [c for c in ["Observation_Year", "Sex", "Insurance_Plan", "Issue_Age", "Duration", "Attained_Age", "Death_Count", "ExpDth_VBT2015_Cnt"] if c in available_columns]
        preview_cols = st.multiselect(
            "Preview columns",
            options=available_columns,
            default=default_preview_cols or available_columns[:10],
            key="preview_cols_working",
        )
        preview_n = st.slider("Preview rows", 10, 200, 50, 10, key="preview_rows_working")
        preview_df = filtered_preview_query(path, nrows, tuple(preview_cols), preview_n, cat_frozen, num_frozen)
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

    with st.expander("Download working dataset", expanded=False):
        st.caption("To keep the app stable, this download is capped. Increase the cap only when needed.")
        download_cap = st.slider("Download row cap", 1000, 200000, 50000, 1000, key="download_cap")
        if st.button("Prepare filtered download", key="prepare_filtered_download"):
            st.session_state["working_download_df"] = filtered_download_query(path, nrows, int(download_cap), cat_frozen, num_frozen)
        if "working_download_df" in st.session_state and isinstance(st.session_state["working_download_df"], pd.DataFrame):
            filtered_filename = build_filtered_filename(source_name, min(working_count, int(download_cap)), active_num_ranges)
            st.download_button(
                f"Download up to {int(download_cap):,} filtered rows (.csv)",
                data=df_to_csv_bytes(st.session_state["working_download_df"]),
                file_name=filtered_filename,
                mime="text/csv",
                key="download_working_data",
            )


def render_pivot_tab(
    path: str,
    nrows: Optional[int],
    available_columns: List[str],
    active_cat_filters: Dict[str, List[str]],
    active_num_ranges: Dict[str, Tuple[float, float]],
) -> None:
    st.caption("Build the pivot table from the current preliminary working data.")

    cat_frozen = freeze_cat_filters(active_cat_filters)
    num_frozen = freeze_num_ranges(active_num_ranges)
    working_count = filtered_row_count_query(path, nrows, cat_frozen, num_frozen)
    if working_count == 0:
        st.info("Apply broader preliminary filters to populate the pivot table.")
        return

    axis_fields = [c for c in available_columns if c in ALL_KNOWN]
    value_fields = [c for c in available_columns if c in NUM_COLUMNS]

    with st.form("pivot_form"):
        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.1, 0.8], gap="large")
        with c1:
            row_fields = st.multiselect(
                "Rows",
                axis_fields,
                default=[c for c in ["Sex", "Attained_Age"] if c in axis_fields],
                key="pivot_rows_duck",
            )
        with c2:
            col_fields = st.multiselect(
                "Columns",
                axis_fields,
                default=[c for c in ["Observation_Year"] if c in axis_fields],
                key="pivot_cols_duck",
            )
        with c3:
            value_sel = st.multiselect(
                "Values",
                value_fields,
                default=[c for c in ["Death_Count", "ExpDth_VBT2015_Cnt"] if c in value_fields],
                key="pivot_vals_duck",
            )
        with c4:
            agg_choice = st.selectbox("Aggregation", ["sum", "mean", "count", "min", "max", "nunique"], index=0, key="pivot_agg_duck")
            add_measures = st.checkbox("Add A/E columns", value=True, key="pivot_add_ae_duck")
        pivot_apply = st.form_submit_button("Build pivot", type="primary")

    if "pivot_table_duck" not in st.session_state:
        st.session_state["pivot_table_duck"] = pd.DataFrame()

    if pivot_apply:
        safe_rows = unique_preserve_order([c for c in row_fields if c in axis_fields])
        safe_cols = unique_preserve_order([c for c in col_fields if c in axis_fields and c not in safe_rows])
        safe_vals = unique_preserve_order([c for c in value_sel if c in value_fields])

        if not safe_vals and not add_measures:
            st.warning("Choose at least one value field, or keep Add A/E columns checked.")
        else:
            base = build_pivot_base_duckdb(
                path,
                nrows,
                tuple(safe_rows + safe_cols),
                tuple(safe_vals),
                agg_choice,
                add_measures,
                cat_frozen,
                num_frozen,
            )
            metric_fields = [c for c in safe_vals if c in base.columns] + [c for c in AE_OPTIONS if c in base.columns]
            pt = pivot_table_from_base(base, row_fields=safe_rows, col_fields=safe_cols, metric_fields=metric_fields)
            st.session_state["pivot_table_duck"] = flatten_columns(pt)

    pt_show = st.session_state.get("pivot_table_duck", pd.DataFrame())
    if pt_show.empty:
        st.info("Choose the pivot settings and click Build pivot.")
        return

    st.dataframe(pt_show, use_container_width=True, hide_index=False, height=520)
    st.download_button(
        "Download pivot (.csv)",
        data=df_to_csv_bytes(pt_show.reset_index(drop=False)),
        file_name="pivot_table.csv",
        mime="text/csv",
        key="download_pivot_duck",
    )


def render_graph_filter_block(
    path: str,
    nrows: Optional[int],
    tab_label: str,
    prelim_cat_filters: Dict[str, List[str]],
    prelim_num_ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, List[str]]:
    prelim_cat_frozen = freeze_cat_filters(prelim_cat_filters)
    prelim_num_frozen = freeze_num_ranges(prelim_num_ranges)

    graph_filters: Dict[str, List[str]] = {}
    st.markdown("**Chart filters**")

    for col in GRAPH_FILTER_FIELDS:
        opts = category_options_filtered_query(path, nrows, col, prelim_cat_frozen, prelim_num_frozen)
        key = f"graph_filter_{tab_label}_{col}"
        ensure_multiselect_state(key, opts, opts)
        graph_filters[col] = st.multiselect(
            human(col),
            options=opts,
            key=key,
            placeholder=f"Select {human(col)}",
        )

    return graph_filters


def render_analysis_tab(
    path: str,
    nrows: Optional[int],
    available_columns: List[str],
    active_cat_filters: Dict[str, List[str]],
    active_num_ranges: Dict[str, Tuple[float, float]],
    tab_label: str,
    display_field: str,
    source_field: str,
) -> None:
    prelim_cat_frozen = freeze_cat_filters(active_cat_filters)
    prelim_num_frozen = freeze_num_ranges(active_num_ranges)
    working_count = filtered_row_count_query(path, nrows, prelim_cat_frozen, prelim_num_frozen)
    if working_count == 0:
        st.info("Apply broader preliminary filters to populate this page.")
        return

    actual_x = source_field
    numeric_fields = [c for c in available_columns if c in NUM_COLUMNS]
    split_candidates = [c for c in available_columns if c in CAT_COLUMNS and c != source_field]

    state_key = f"chart_output_{tab_label}"
    error_key = f"chart_error_{tab_label}"

    plot_col, settings_col = st.columns([4.6, 1.8], gap="large")

    with settings_col:
        graph_filter_dict = render_graph_filter_block(
            path=path,
            nrows=nrows,
            tab_label=tab_label,
            prelim_cat_filters=active_cat_filters,
            prelim_num_ranges=active_num_ranges,
        )

        st.markdown("---")
        st.markdown("**Chart setup**")

        if source_field in NUM_COLUMNS:
            max_bin = 10 if source_field in {"Issue_Age", "Attained_Age", "Duration"} else 5
            bin_size = st.select_slider(
                "Bin size",
                options=list(range(1, max_bin + 1)),
                value=1,
                key=f"bin_{tab_label}",
            )
        else:
            bin_size = 1

        chart_mode = st.selectbox(
            "Chart mode",
            ["Combo", "Lines only", "Bars only"],
            index=0,
            key=f"mode_{tab_label}",
        )
        line_mode = st.selectbox(
            "Line style",
            ["Lines + Markers", "Lines", "Markers"],
            index=0,
            key=f"line_mode_{tab_label}",
        )
        height = st.slider(
            "Chart height",
            360,
            900,
            540,
            10,
            key=f"height_{tab_label}",
        )

        default_bars = [c for c in ["Death_Count", "ExpDth_VBT2015_Cnt"] if c in numeric_fields]
        line_metric_options = AE_OPTIONS + [
            c for c in numeric_fields
            if c not in {
                "Death_Count",
                "Death_Claim_Amount",
                "ExpDth_VBT2015_Cnt",
                "ExpDth_VBT2015_Amt",
                "ExpDth_VBT2015wMI_Cnt",
                "ExpDth_VBT2015wMI_Amt",
            }
        ]
        default_line_metrics = [m for m in ["A/E (Count)", "A/E (Amount)"] if m in AE_OPTIONS]

        ensure_multiselect_state(f"bars_{tab_label}", numeric_fields, default_bars)
        ensure_multiselect_state(f"line_metrics_{tab_label}", line_metric_options, default_line_metrics)

        bar_metrics = st.multiselect(
            "Bar metrics",
            options=numeric_fields,
            key=f"bars_{tab_label}",
        )
        line_metrics = st.multiselect(
            "Line metrics",
            options=line_metric_options,
            key=f"line_metrics_{tab_label}",
        )

        non_ae_agg = st.selectbox(
            "Aggregation for non-A/E line metrics",
            ["sum", "mean"],
            index=0,
            key=f"line_agg_{tab_label}",
        )

        split1_raw = st.selectbox(
            "Split 1",
            options=["(none)"] + split_candidates,
            index=0,
            key=f"split1_{tab_label}",
        )
        split1 = None if split1_raw == "(none)" else split1_raw

        split2_candidates = [c for c in split_candidates if c != split1]
        split2_raw = st.selectbox(
            "Split 2",
            options=["(none)"] + split2_candidates,
            index=0,
            key=f"split2_{tab_label}",
        )
        split2 = None if split2_raw == "(none)" else split2_raw

        # Strip empty local chart filters before using them anywhere
        active_graph_filters = {
            k: v for k, v in graph_filter_dict.items()
            if v is not None and len(v) > 0
        }

        merged_for_split_options = combine_category_filters(active_cat_filters, active_graph_filters)

        if split1:
            split1_opts = category_options_filtered_query(
                path,
                nrows,
                split1,
                freeze_cat_filters(merged_for_split_options),
                prelim_num_frozen,
            )
            ensure_multiselect_state(
                f"keep1_{tab_label}",
                split1_opts,
                split1_opts[:12] if len(split1_opts) > 12 else split1_opts,
            )
            keep_split1 = st.multiselect(
                "Keep Split 1 values",
                options=split1_opts,
                key=f"keep1_{tab_label}",
            )
        else:
            keep_split1 = []

        if split2:
            split2_filter_base = combine_category_filters(
                merged_for_split_options,
                {split1: keep_split1} if split1 and keep_split1 else {},
            )
            split2_opts = category_options_filtered_query(
                path,
                nrows,
                split2,
                freeze_cat_filters(split2_filter_base),
                prelim_num_frozen,
            )
            ensure_multiselect_state(
                f"keep2_{tab_label}",
                split2_opts,
                split2_opts[:8] if len(split2_opts) > 8 else split2_opts,
            )
            keep_split2 = st.multiselect(
                "Keep Split 2 values",
                options=split2_opts,
                key=f"keep2_{tab_label}",
            )
        else:
            keep_split2 = []

        top_n_traces = st.slider(
            "Max line traces",
            1,
            60,
            20,
            1,
            key=f"tracecap_{tab_label}",
        )

        apply_chart = st.button(
            "Apply chart",
            type="primary",
            key=f"apply_chart_{tab_label}",
        )

    if state_key not in st.session_state:
        st.session_state[state_key] = None
    if error_key not in st.session_state:
        st.session_state[error_key] = None

    if apply_chart:
        merged_cat_filters = combine_category_filters(active_cat_filters, active_graph_filters)

        if not bar_metrics:
            bar_metrics = default_bars
        if not line_metrics:
            line_metrics = default_line_metrics

        try:
            bars, lines, axis_order = summarize_for_chart_duckdb(
                path=path,
                nrows=nrows,
                base_x_field=actual_x,
                bin_size=bin_size,
                bar_metrics=tuple(bar_metrics),
                line_metrics=tuple(line_metrics),
                split1=split1,
                split2=split2,
                keep_split1=tuple(keep_split1),
                keep_split2=tuple(keep_split2),
                non_ae_agg=non_ae_agg,
                top_n_traces=top_n_traces,
                cat_filters_frozen=freeze_cat_filters(merged_cat_filters),
                num_ranges_frozen=prelim_num_frozen,
            )
            st.session_state[state_key] = {
                "bars": bars,
                "lines": lines,
                "axis_order": axis_order,
                "bar_metrics": bar_metrics,
                "line_metrics": line_metrics,
                "title": f"{tab_label}: {human(display_field)}",
                "x_title": human(display_field),
                "height": height,
                "chart_mode": chart_mode,
                "line_mode": line_mode,
            }
            st.session_state[error_key] = None
        except Exception as exc:
            st.session_state[error_key] = str(exc)

    payload = st.session_state.get(state_key)
    chart_error = st.session_state.get(error_key)

    with plot_col:
        if chart_error:
            st.error(f"This tab hit an error while applying the chart: {chart_error}")

        if not payload:
            st.info("Set the chart controls on the right and click Apply chart.")
            return

        bars = payload["bars"]
        lines = payload["lines"]
        axis_order = payload["axis_order"]

        if bars.empty and lines.empty:
            st.warning("No summary output could be built for this page with the current settings.")
            return

        try:
            fig = make_visual_figure(
                bars=bars,
                lines=lines,
                axis_order=axis_order,
                bar_metrics=payload["bar_metrics"],
                line_metrics=payload["line_metrics"],
                chart_mode=payload["chart_mode"],
                line_mode=payload["line_mode"],
                title=payload["title"],
                x_title=payload["x_title"],
                height=payload["height"],
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"This tab could not render the saved chart output: {exc}")
            return

        with st.expander("Download aggregated outputs", expanded=False):
            if not bars.empty:
                st.download_button(
                    f"Download {tab_label} bars (.csv)",
                    data=df_to_csv_bytes(bars),
                    file_name=safe_filename(f"{tab_label.lower().replace(' ', '_')}_bars.csv"),
                    mime="text/csv",
                    key=f"download_bars_{tab_label}",
                )
            if not lines.empty:
                st.download_button(
                    f"Download {tab_label} lines (.csv)",
                    data=df_to_csv_bytes(lines),
                    file_name=safe_filename(f"{tab_label.lower().replace(' ', '_')}_lines.csv"),
                    mime="text/csv",
                    key=f"download_lines_{tab_label}",
                )

def safe_render_tab(tab_name: str, render_fn) -> None:
    try:
        render_fn()
    except Exception as exc:
        st.error(f"{tab_name} encountered an error, but the other tabs remain usable.")
        st.exception(exc)


# ---------------------------
# Main app
# ---------------------------
BUILTIN_DATA_PATH = resolve_builtin_data_path()

if not os.path.exists(BUILTIN_DATA_PATH):
    checked_locations = "\n".join(builtin_data_candidates())
    st.error(f"Built-in data file was not found. Checked these locations:\n\n{checked_locations}")
    st.stop()

data_path = BUILTIN_DATA_PATH
data_nrows = None
source_name = os.path.basename(data_path)

try:
    schema_df = describe_source(data_path, data_nrows)
    available_columns_all = schema_df["column_name"].tolist() if "column_name" in schema_df.columns else []
    available_columns = [c for c in available_columns_all if c in ALL_KNOWN]
except Exception as exc:
    st.error(f"Failed to open built-in data file: {exc}")
    st.stop()

default_cat_filters, default_num_ranges = default_filter_config(data_path, data_nrows, available_columns)

if "data_path" not in st.session_state or st.session_state.get("data_path") != data_path:
    st.session_state["data_path"] = data_path
    st.session_state["data_nrows"] = data_nrows
    st.session_state["source_name"] = source_name
    st.session_state["schema_df"] = schema_df
    st.session_state["active_cat_filters"] = default_cat_filters
    st.session_state["active_num_ranges"] = default_num_ranges
    st.session_state["working_filter_cols"] = [c for c in DEFAULT_FILTER_FIELDS if c in available_columns]
    st.session_state.pop("working_download_df", None)
    for label, _, _ in PAGE_SPECS:
        st.session_state.pop(f"chart_output_{label}", None)
    st.session_state["pivot_table_duck"] = pd.DataFrame()

welcome_tab, tab_prelim, tab_pivot, tab_obs, tab_issue, tab_dur, tab_att, tab_face = st.tabs(
    [
        "Welcome",
        "Preliminary filters",
        "Pivot table",
        "Observation Year",
        "Issue Age",
        "Duration",
        "Attained Age",
        "Face Amount",
    ]
)

with welcome_tab:
    safe_render_tab("Welcome", lambda: render_welcome_tab(source_name, data_path, available_columns))

with tab_prelim:
    safe_render_tab(
        "Preliminary filters",
        lambda: render_preliminary_filters_tab(
            data_path,
            data_nrows,
            available_columns,
            source_name,
            default_cat_filters,
            default_num_ranges,
        ),
    )

active_cat_filters = st.session_state.get("active_cat_filters", {})
active_num_ranges = st.session_state.get("active_num_ranges", {})

with tab_pivot:
    safe_render_tab("Pivot table", lambda: render_pivot_tab(data_path, data_nrows, available_columns, active_cat_filters, active_num_ranges))

with tab_obs:
    safe_render_tab("Observation Year", lambda: render_analysis_tab(data_path, data_nrows, available_columns, active_cat_filters, active_num_ranges, "Observation Year", "Observation_Year", "Observation_Year"))
with tab_issue:
    safe_render_tab("Issue Age", lambda: render_analysis_tab(data_path, data_nrows, available_columns, active_cat_filters, active_num_ranges, "Issue Age", "Issue_Age", "Issue_Age"))
with tab_dur:
    safe_render_tab("Duration", lambda: render_analysis_tab(data_path, data_nrows, available_columns, active_cat_filters, active_num_ranges, "Duration", "Duration", "Duration"))
with tab_att:
    safe_render_tab("Attained Age", lambda: render_analysis_tab(data_path, data_nrows, available_columns, active_cat_filters, active_num_ranges, "Attained Age", "Attained_Age", "Attained_Age"))
with tab_face:
    safe_render_tab("Face Amount", lambda: render_analysis_tab(data_path, data_nrows, available_columns, active_cat_filters, active_num_ranges, "Face Amount", "Face_Amount_Band_Label", "Face_Amount_Band"))
