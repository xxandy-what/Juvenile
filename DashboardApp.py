# app.py
# Streamlit PivotTable + Multi-Line Combo Chart + Optional Line Grouping
# Local-path loading + "N/A" preserved as a real category in every filter
# ----------------------------------------------------------------------
# Requirements:
#   pip install streamlit pandas numpy plotly
# Run:
#   streamlit run app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Pivot + Multi-Line Combo Chart", layout="wide")


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
    "ExpDth_VBT2015_Cnt": "Expected Deaths (VBT2015) - Count",
    "ExpDth_VBT2015_Amt": "Expected Deaths (VBT2015) - Amount",
    "Death_Claim_Amount": "Actual Deaths - Claim Amount",
    "Death_Count": "Actual Deaths - Count",
    "Policies_Exposed": "Exposure - Policies",
    "Amount_Exposed": "Exposure - Amount",
}

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

AGG_MAP = {
    "Sum": "sum",
    "Mean": "mean",
    "Count": "count",
    "Min": "min",
    "Max": "max",
    "Distinct Count (nunique)": "nunique",
}

AE_INPUTS = ["Death_Count", "ExpDth_VBT2015_Cnt", "Death_Claim_Amount", "ExpDth_VBT2015_Amt"]


# ---------------------------
# Helpers (performance + safety)
# ---------------------------
def human(col: str) -> str:
    return FRIENDLY.get(col, col)

def safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    d2 = d.replace(0, np.nan)
    out = n / d2
    return out.replace([np.inf, -np.inf], np.nan)

def _as_categorical_if_reasonable(s: pd.Series, max_unique: int = 2000) -> pd.Series:
    try:
        nunq = s.nunique(dropna=True)
        if nunq <= max_unique:
            return s.astype("category")
    except Exception:
        pass
    return s

def coerce_types_fast(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # numeric coercion (won't affect string columns like "N/A")
    for c in NUM_COLUMNS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # categorical coercion
    for c in CAT_COLUMNS:
        if c in out.columns:
            out[c] = out[c].astype("string")
            out[c] = _as_categorical_if_reasonable(out[c])

    # Face amount band label
    if "Face_Amount_Band" in out.columns and "Face_Amount_Band_Label" not in out.columns:
        s = out["Face_Amount_Band"].astype("string")
        out["Face_Amount_Band_Label"] = s.map(FACE_BAND_MAP).astype("string")
        out["Face_Amount_Band_Label"] = _as_categorical_if_reasonable(out["Face_Amount_Band_Label"])

    # other object/string cols
    for c in out.columns:
        if c in ALL_KNOWN or c == "Face_Amount_Band_Label":
            continue
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].astype("string")
            out[c] = _as_categorical_if_reasonable(out[c])

    return out

@st.cache_data(show_spinner=False)
def load_local(path: str) -> pd.DataFrame:
    """
    Critical: preserve literal 'N/A' as a category.
    - keep_default_na=False and na_filter=False prevent pandas from converting 'N/A' to NaN.
    """
    read_kwargs = dict(
        low_memory=False,
        keep_default_na=False,  # do not convert N/A, NA, etc. to NaN
        na_filter=False,        # disable NA parsing entirely
    )
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, **read_kwargs)
    else:
        df = pd.read_csv(path, sep="\t", **read_kwargs)

    # Treat blank strings as missing (optional), but NEVER touch 'N/A'
    df = df.replace({"": np.nan})

    return coerce_types_fast(df)

def safe_filename(text: str) -> str:
    text = re.sub(r"[^\w\-\.]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text

@st.cache_data(show_spinner=False)
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def build_filtered_filename(source_name: str, df_filtered: pd.DataFrame, num_ranges: Dict[str, Tuple[float, float]]) -> str:
    base = os.path.splitext(source_name)[0]
    parts = [base, "filtered"]

    # nice year tag if year is one of the active numeric filters
    if "Observation_Year" in num_ranges:
        lo, hi = num_ranges["Observation_Year"]
        if int(lo) == int(hi):
            parts.append(f"year_{int(lo)}")
        else:
            parts.append(f"years_{int(lo)}_{int(hi)}")

    parts.append(f"rows_{len(df_filtered)}")
    return safe_filename("_".join(parts) + ".csv")

def _category_options_including_NA(df: pd.DataFrame, col: str) -> List[str]:
    """
    Build options list WITHOUT dropping 'N/A'.
    - 'N/A' stays if it exists as a literal string.
    - If there are true missing values (NaN), we expose them as '(Missing)'.
    """
    s = df[col].astype("string")  # keeps "N/A" as "N/A"
    has_missing = df[col].isna().any()
    # Do NOT dropna for options; keep actual strings, but remove <NA> tokens from options list
    uniq = s.unique().tolist()
    uniq = [u for u in uniq if u is not None and str(u) != "<NA>"]
    uniq = sorted([str(u) for u in uniq])

    if has_missing:
        uniq = ["(Missing)"] + uniq
    return uniq

def apply_report_filters(
    df: pd.DataFrame,
    cat_filters: Dict[str, List[str]],
    num_ranges: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    out = df
    for col, selected in cat_filters.items():
        if col not in out.columns or selected is None:
            continue

        selected = [str(x) for x in selected]
        want_missing = "(Missing)" in selected
        selected_no_missing = [v for v in selected if v != "(Missing)"]

        if want_missing and selected_no_missing:
            out = out[(out[col].isna()) | (out[col].astype("string").isin(pd.Series(selected_no_missing, dtype="string")))]
        elif want_missing and not selected_no_missing:
            out = out[out[col].isna()]
        else:
            out = out[out[col].astype("string").isin(pd.Series(selected_no_missing, dtype="string"))]

    for col, (lo, hi) in num_ranges.items():
        if col in out.columns:
            out = out[(out[col] >= lo) & (out[col] <= hi)]

    return out

def build_pivot_base(
    df: pd.DataFrame,
    dims: List[str],
    values: List[str],
    aggfunc: str,
    add_measures: bool,
) -> pd.DataFrame:
    if not dims:
        tmp = df.copy()
        tmp["_ALL_"] = "All"
        dims = ["_ALL_"]
        df = tmp

    agg_dict: Dict[str, str] = {}
    for v in values:
        if v in df.columns:
            agg_dict[v] = aggfunc

    if add_measures:
        for m in AE_INPUTS:
            if m in df.columns:
                agg_dict[m] = "sum"

    g = df.groupby(dims, dropna=False, observed=True).agg(agg_dict).reset_index()

    if add_measures:
        if "Death_Count" in g.columns and "ExpDth_VBT2015_Cnt" in g.columns:
            g["A/E (Count)"] = safe_div(g["Death_Count"], g["ExpDth_VBT2015_Cnt"])
        if "Death_Claim_Amount" in g.columns and "ExpDth_VBT2015_Amt" in g.columns:
            g["A/E (Amount)"] = safe_div(g["Death_Claim_Amount"], g["ExpDth_VBT2015_Amt"])
    return g

def pivot_table_from_base(
    base: pd.DataFrame,
    row_fields: List[str],
    col_fields: List[str],
    metric_fields: List[str],
) -> pd.DataFrame:
    if not row_fields:
        base = base.copy()
        base["_Rows"] = "All"
        row_fields = ["_Rows"]

    if not col_fields:
        return base[row_fields + metric_fields].set_index(row_fields)

    pt = pd.pivot_table(
        base,
        index=row_fields,
        columns=col_fields,
        values=metric_fields,
        aggfunc="first",
        dropna=False,
        observed=True,
    )
    return pt

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [" | ".join([str(x) for x in tup if str(x) != ""]) for tup in out.columns.to_flat_index()]
    else:
        out.columns = [str(c) for c in out.columns]
    return out

def sort_x(g: pd.DataFrame, x_dim: str) -> pd.DataFrame:
    try:
        x_num = pd.to_numeric(g[x_dim], errors="coerce")
        if x_num.notna().any():
            g = g.assign(_x_num=x_num).sort_values(["_x_num", x_dim]).drop(columns=["_x_num"])
        else:
            g = g.sort_values(x_dim)
    except Exception:
        g = g.sort_values(x_dim)
    return g

def infer_bar_axis_label(bar_cols: List[str]) -> str:
    if any(c in ("Death_Claim_Amount", "ExpDth_VBT2015_Amt", "Amount_Exposed") for c in bar_cols):
        return "Bars (Amount Totals)"
    if any(c in ("Death_Count", "ExpDth_VBT2015_Cnt", "Policies_Exposed") for c in bar_cols):
        return "Bars (Count / Exposure Totals)"
    return "Bars (Totals)"

def infer_line_axis_label(line_metric: str) -> str:
    if line_metric.startswith("A/E"):
        return "Lines (A/E Ratio)"
    return f"Lines ({human(line_metric)})"

def build_line_group_column(
    df: pd.DataFrame,
    category_col: str,
    grouping_enabled: bool,
    groups_map: Dict[str, List[str]],
    include_other: bool,
    other_label: str = "Other",
) -> Tuple[pd.DataFrame, str]:
    if category_col is None:
        return df, category_col

    d = df.copy()
    base = d[category_col].astype("string").fillna("(Missing)")

    if not grouping_enabled or not groups_map:
        d["_Line_Group_"] = base
        return d, "_Line_Group_"

    value_to_group: Dict[str, str] = {}
    for gname, members in groups_map.items():
        for v in members:
            value_to_group[str(v)] = str(gname)

    mapped = base.map(lambda x: value_to_group.get(str(x), None))
    if include_other:
        mapped = mapped.fillna(other_label)
    else:
        mapped = mapped.fillna(np.nan)

    d["_Line_Group_"] = mapped.astype("string")
    d["_Line_Group_"] = _as_categorical_if_reasonable(d["_Line_Group_"])
    return d, "_Line_Group_"

def make_combo_multiline_chart(
    df_filtered: pd.DataFrame,
    x_dim: str,
    bar_left: str,
    bar_right: Optional[str],
    line_metric: str,
    line_category_for_lines: Optional[str],
    keep_line_values: Optional[List[str]],
    title: str,
) -> go.Figure:
    need_cols = {x_dim, bar_left}
    if bar_right:
        need_cols.add(bar_right)

    if line_category_for_lines:
        need_cols.add(line_category_for_lines)

    if line_metric in ("A/E (Count)", "A/E (Amount)"):
        need_cols.update(AE_INPUTS)
    else:
        need_cols.add(line_metric)

    cols = [c for c in need_cols if c in df_filtered.columns]
    d = df_filtered[cols].copy()

    if line_category_for_lines and keep_line_values is not None and len(keep_line_values) > 0:
        d = d[d[line_category_for_lines].astype("string").isin(pd.Series(keep_line_values, dtype="string"))]

    bar_agg = {bar_left: "sum"}
    if bar_right and bar_right in d.columns:
        bar_agg[bar_right] = "sum"
    bars = d.groupby(x_dim, dropna=False, observed=True).agg(bar_agg).reset_index()
    bars = sort_x(bars, x_dim)

    if line_metric in ("A/E (Count)", "A/E (Amount)"):
        if line_metric == "A/E (Count)":
            num_col, den_col = "Death_Count", "ExpDth_VBT2015_Cnt"
        else:
            num_col, den_col = "Death_Claim_Amount", "ExpDth_VBT2015_Amt"

        keys = [x_dim] + ([line_category_for_lines] if line_category_for_lines else [])
        sums = d.groupby(keys, dropna=False, observed=True)[[num_col, den_col]].sum().reset_index()
        sums[line_metric] = safe_div(sums[num_col], sums[den_col])
        line_df = sums[[*keys, line_metric]]
    else:
        keys = [x_dim] + ([line_category_for_lines] if line_category_for_lines else [])
        line_df = d.groupby(keys, dropna=False, observed=True)[line_metric].sum().reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x_vals = bars[x_dim].astype(str)

    fig.add_trace(go.Bar(x=x_vals, y=bars[bar_left], name=human(bar_left), opacity=0.85), secondary_y=False)
    if bar_right and bar_right in bars.columns:
        fig.add_trace(go.Bar(x=x_vals, y=bars[bar_right], name=human(bar_right), opacity=0.70), secondary_y=False)

    if line_category_for_lines:
        cats = line_df[line_category_for_lines].astype("string").unique().tolist()
        for cat in cats:
            sub = line_df[line_df[line_category_for_lines].astype("string") == cat]
            sub = sort_x(sub, x_dim)
            fig.add_trace(
                go.Scatter(
                    x=sub[x_dim].astype(str),
                    y=sub[line_metric],
                    mode="lines+markers",
                    name=f"{human(line_metric)} — {human(line_category_for_lines)}={cat}",
                    line=dict(width=2.5),
                    marker=dict(size=6),
                ),
                secondary_y=True,
            )
    else:
        line_df = sort_x(line_df, x_dim)
        fig.add_trace(
            go.Scatter(
                x=line_df[x_dim].astype(str),
                y=line_df[line_metric],
                mode="lines+markers",
                name=human(line_metric),
                line=dict(width=3),
                marker=dict(size=7),
            ),
            secondary_y=True,
        )

    bar_axis = infer_bar_axis_label([c for c in [bar_left, bar_right] if c])
    line_axis = infer_line_axis_label(line_metric)

    fig.update_layout(
        title=title,
        barmode="group",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=70, b=30),
        height=540,
    )
    fig.update_xaxes(title_text=human(x_dim), tickangle=-25)
    fig.update_yaxes(title_text=bar_axis, secondary_y=False, tickformat=",.4g")
    fig.update_yaxes(title_text=line_axis, secondary_y=True, tickformat=".4f" if line_metric.startswith("A/E") else ",.4g")
    return fig


# ---------------------------
# App UI
# ---------------------------
st.title("PivotTable + Multi-Line Combo Chart (Streamlit)")

st.subheader("Load data from local disk (no upload)")

default_path = r"D:\1_ilec-mort-text-data-dict-2012-2019\Juvenile_cleaned.txt"
path = st.text_input("TSV/CSV file path", value=st.session_state.get("last_path", default_path))

colA, colB = st.columns([1, 2])
with colA:
    load_clicked = st.button("Load file", type="primary")
with colB:
    st.caption("After loading once, you can interact without reloading.")

if load_clicked:
    try:
        df_loaded = load_local(path)
        st.session_state["df_loaded"] = df_loaded
        st.session_state["last_path"] = path
        st.session_state["source_name"] = os.path.basename(path)
        st.success(f"Loaded: {st.session_state['source_name']}")
    except Exception as e:
        st.error(f"Failed to load file:\n{e}")
        st.stop()

if "df_loaded" not in st.session_state:
    st.info("Enter a local path and click **Load file**.")
    st.stop()

df = st.session_state["df_loaded"]
source_name = st.session_state.get("source_name", os.path.basename(path))

missing_for_ae = [c for c in AE_INPUTS if c not in df.columns]
st.caption(f"Loaded: **{source_name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
if missing_for_ae:
    st.warning("A/E needs: " + ", ".join(AE_INPUTS) + f". Missing: {', '.join(missing_for_ae)}")


# Sidebar filters
with st.sidebar:
    st.header("Report Filters")

    candidate_filter_cols = [c for c in df.columns if (c in ALL_KNOWN or c == "Face_Amount_Band_Label")]
    default_filters = [c for c in ["Observation_Year", "Sex", "Smoker_Status", "Insurance_Plan", "SOA_Post_Lvl_Ind"] if c in candidate_filter_cols]

    filter_cols = st.multiselect("Fields to filter", options=candidate_filter_cols, default=default_filters)

    cat_filters: Dict[str, List[str]] = {}
    num_ranges: Dict[str, Tuple[float, float]] = {}

    for col in filter_cols:
        INT_FIELDS = {"Observation_Year", "Issue_Age", "Attained_Age", "Duration", "Issue_Year", "Death_Count"}

        if pd.api.types.is_numeric_dtype(df[col]):
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) == 0:
                continue

            if col in INT_FIELDS:
                min_v, max_v = int(s.min()), int(s.max())
                lo, hi = st.slider(
                    f"{human(col)} range",
                    min_value=min_v,
                    max_value=max_v,
                    value=(min_v, max_v),
                    step=1
                )
                num_ranges[col] = (lo, hi)
            else:
                min_v, max_v = float(s.min()), float(s.max())
                lo, hi = st.slider(
                    f"{human(col)} range",
                    min_value=min_v,
                    max_value=max_v,
                    value=(min_v, max_v)
                )
                num_ranges[col] = (lo, hi)
        else:
            opts = _category_options_including_NA(df, col)  # <-- includes literal "N/A"
            selected = st.multiselect(f"{human(col)} values", options=opts, default=opts)
            cat_filters[col] = selected

df_f = apply_report_filters(df, cat_filters, num_ranges)

if df_f.shape[0] == 0:
    st.error("No rows left after filters. Relax your filters.")
else:
    filtered_filename = build_filtered_filename(source_name, df_f, num_ranges)
    filtered_csv = df_to_csv_bytes(df_f)

    st.download_button(
        label="Download filtered data (.csv)",
        data=filtered_csv,
        file_name=filtered_filename,
        mime="text/csv",
    )

st.write("")
m1, m2 = st.columns(2)
m1.metric("Rows after filters", f"{df_f.shape[0]:,}")
m2.metric("Filters active", str(len([k for k, v in cat_filters.items() if v is not None]) + len(num_ranges)))


# Tabs
tab_chart, tab_pivot, tab_preview = st.tabs(["📈 Chart", "🧾 Pivot", "🔍 Data Preview"])

with tab_preview:
    st.dataframe(df_f.head(50), use_container_width=True)
    st.caption("Preview is from the filtered dataset.")

axis_fields = [c for c in df_f.columns if (c in ALL_KNOWN or c == "Face_Amount_Band_Label")]
cat_axis_fields = [c for c in axis_fields if not pd.api.types.is_numeric_dtype(df_f[c])]
value_options = [c for c in df_f.columns if pd.api.types.is_numeric_dtype(df_f[c])]

with tab_chart:
    c1, c2, c3 = st.columns([1.1, 1.1, 1.2], gap="large")

    with c1:
        x_dim = st.selectbox("X-axis", options=axis_fields, index=axis_fields.index("Observation_Year") if "Observation_Year" in axis_fields else 0)
        bar_options = [b for b in ["Death_Count", "ExpDth_VBT2015_Cnt", "Death_Claim_Amount", "ExpDth_VBT2015_Amt"] if b in df_f.columns]
        bar_left = st.selectbox("Bar 1", options=bar_options, index=0)
        bar2_opts = ["(none)"] + bar_options
        bar_right_raw = st.selectbox("Bar 2", options=bar2_opts, index=1 if len(bar2_opts) > 1 else 0)
        bar_right = None if bar_right_raw == "(none)" else bar_right_raw

    with c2:
        add_measures = st.checkbox("Enable A/E", value=True)
        line_metric_candidates: List[str] = []
        if add_measures:
            line_metric_candidates += ["A/E (Count)", "A/E (Amount)"]
        line_metric_candidates += [c for c in value_options if c not in (bar_left, bar_right)]
        line_metric_candidates = list(dict.fromkeys(line_metric_candidates))
        line_metric = st.selectbox("Line metric", options=line_metric_candidates, index=(line_metric_candidates.index("A/E (Count)") if "A/E (Count)" in line_metric_candidates else 0))

        line_category_raw = st.selectbox(
            "Split lines by",
            options=["(none)"] + cat_axis_fields + ["Observation_Year"],
            index=0
        )
        raw_line_category = None if line_category_raw == "(none)" else line_category_raw

    with c3:
        st.markdown("**Optional grouping**")
        grouping_enabled = st.checkbox("Custom groups for lines", value=False, disabled=(raw_line_category is None))

        if "line_groups" not in st.session_state:
            st.session_state["line_groups"] = {}

        groups_map: Dict[str, List[str]] = st.session_state["line_groups"]

        available_vals: List[str] = []
        if raw_line_category is not None:
            # IMPORTANT: don't dropna; keep literal "N/A"
            svals = df_f[raw_line_category].astype("string").unique().tolist()
            svals = [v for v in svals if v is not None and str(v) != "<NA>"]
            available_vals = sorted([str(v) for v in svals])

        if grouping_enabled and raw_line_category is not None:
            with st.expander("Manage groups", expanded=True):
                new_group_name = st.text_input("Group name", value="", placeholder="e.g., UL family")
                new_group_members = st.multiselect("Members", options=available_vals, default=[])

                a, b = st.columns(2)
                with a:
                    if st.button("Add / Update group"):
                        gname = new_group_name.strip()
                        if gname and new_group_members:
                            groups_map[gname] = list(dict.fromkeys([str(x) for x in new_group_members]))
                            st.session_state["line_groups"] = groups_map
                with b:
                    del_name = st.selectbox("Delete group", options=["(none)"] + sorted(list(groups_map.keys())))
                    if st.button("Delete selected") and del_name != "(none)":
                        groups_map.pop(del_name, None)
                        st.session_state["line_groups"] = groups_map

            include_other = st.checkbox("Include unassigned as 'Other'", value=True)
            other_label = st.text_input("Other label", value="Other", disabled=not include_other)
        else:
            include_other = True
            other_label = "Other"

    if raw_line_category is not None:
        df_for_chart, line_category_for_lines = build_line_group_column(
            df_f,
            category_col=raw_line_category,
            grouping_enabled=grouping_enabled,
            groups_map=groups_map if grouping_enabled else {},
            include_other=include_other,
            other_label=other_label,
        )
    else:
        df_for_chart, line_category_for_lines = df_f, None

    keep_line_values = None
    if line_category_for_lines is not None:
        # IMPORTANT: no dropna -> keep "N/A" as option; remove <NA> token from list
        uniq = df_for_chart[line_category_for_lines].astype("string").unique().tolist()
        uniq = [u for u in uniq if u is not None and str(u) != "<NA>"]
        uniq = sorted([str(u) for u in uniq])
        default_keep = uniq if len(uniq) <= 20 else uniq[:20]
        keep_line_values = st.multiselect("Line slicer", options=uniq, default=default_keep)

    fig = make_combo_multiline_chart(
        df_filtered=df_for_chart,
        x_dim=x_dim,
        bar_left=bar_left,
        bar_right=bar_right,
        line_metric=line_metric,
        line_category_for_lines=line_category_for_lines,
        keep_line_values=keep_line_values,
        title=f"Combo Chart by {human(x_dim)} — Bars + Lines",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_pivot:
    p1, p2, p3, p4 = st.columns([1.1, 1.1, 1.1, 1.0], gap="large")
    with p1:
        row_fields = st.multiselect("Rows", options=axis_fields, default=[c for c in ["Sex", "Attained_Age"] if c in axis_fields])
    with p2:
        col_fields = st.multiselect("Columns", options=axis_fields, default=[c for c in ["Observation_Year"] if c in axis_fields])
    with p3:
        values = st.multiselect("Values", options=value_options, default=[c for c in ["Death_Count", "ExpDth_VBT2015_Cnt"] if c in value_options])
    with p4:
        agg_choice = st.selectbox("Aggregation", list(AGG_MAP.keys()), index=0)
        aggfunc = AGG_MAP[agg_choice]
        add_measures_pivot = st.checkbox("Add A/E measures", value=True)

    dims = [c for c in (row_fields + col_fields) if c in df_f.columns]
    base = build_pivot_base(df_f, dims=dims, values=values, aggfunc=aggfunc, add_measures=add_measures_pivot)

    metric_fields: List[str] = [v for v in values if v in base.columns]
    if add_measures_pivot:
        if "A/E (Count)" in base.columns:
            metric_fields.append("A/E (Count)")
        if "A/E (Amount)" in base.columns:
            metric_fields.append("A/E (Amount)")

    pt = pivot_table_from_base(base, row_fields=row_fields, col_fields=col_fields, metric_fields=metric_fields)
    st.dataframe(flatten_columns(pt), use_container_width=True, height=560)