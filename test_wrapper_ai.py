import streamlit as st
from ai_assistant import render_ai_assistant_tab

# 提供最基础的上下文，防止组件因为找不到变量而报错
if "data_path" not in st.session_state:
    st.session_state["data_path"] = "dummy.parquet"

# [Phase 8 修复] 提供一个 Mock 的 Schema 上下文，防止类型错误
dummy_schema_context = """
Categorical Columns (Use for GROUP BY or WHERE):
- Sex (F, M)
- Smoker_Status (N, S, U)

Numeric Columns (Use for aggregations or filtering ranges):
- Observation_Year (Range: 2012 to 2019)
- Death_Count (Range: 0 to 100)
"""

# 传入假的 schema_context 进行独立渲染测试
render_ai_assistant_tab(dummy_schema_context)