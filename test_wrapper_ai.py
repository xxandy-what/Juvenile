import streamlit as st
from ai_assistant import render_ai_assistant_tab

# 提供最基础的上下文，防止组件因为找不到变量而报错
if "data_path" not in st.session_state:
    st.session_state["data_path"] = "dummy.parquet"

# 只渲染 AI 助手这一个组件，彻底屏蔽掉 appv2.py 里的各种复杂表单
render_ai_assistant_tab()