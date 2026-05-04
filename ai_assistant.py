import json
import re
import duckdb
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from typing import Optional
from plotly.subplots import make_subplots
from google import genai
from google.genai import types


def get_recent_chat_history(messages: list, max_turns: int = 3) -> str:
    """
    [Phase 9 Enhancement]
    Extracts recent conversation history (text only) to provide context for the LLM.
    We exclude the very last message assuming it's the current user prompt being processed.
    """
    if not messages or len(messages) <= 1:
        return "No previous context."
    
    history_lines = []
    # 获取最近的 max_turns 轮对话（一问一答算1轮，所以 * 2）
    # 切片 [:-1] 是因为此时最新的 prompt 已经 append 进去了，我们要排除它
    recent_msgs = messages[-(max_turns * 2 + 1):-1] 
    
    for msg in recent_msgs:
        # 只提取纯文本，忽略 DataFrame (msg["df"]) 或图表 (msg["fig"])
        role = "User" if msg["role"] == "user" else "Assistant"
        # 过滤掉较长的系统提示或报错，只保留核心交互
        content = msg["content"]
        if "Database Execution Error" in content or "Traceback" in content:
            content = "[System Error Message Omitted]"
            
        history_lines.append(f"{role}: {content}")
        
    return "\n".join(history_lines) if history_lines else "No previous context."

def get_active_filters_context(use_global_filters: bool) -> str:
    """
    [Phase 6 Refactored]
    Tells the LLM whether the underlying view is already filtered. 
    No longer constructs raw SQL strings for the prompt.
    """
    if not use_global_filters:
        return "The user wants to query the FULL dataset. The table `current_working_set` contains unfiltered data."

    return "The table `current_working_set` HAS ALREADY BEEN FILTERED based on the user's global settings. DO NOT add preliminary filters (like Age or Smoker_Status bounds) unless the user explicitly asks to filter them further in their prompt."


def parse_user_intent(user_prompt: str, chat_history: str = "") -> dict:
    """
    Calls the Gemini API to parse user intent and returns a JSON dictionary.
    """
    system_prompt = '''
    You are an intelligent Actuarial Data Assistant routing engine.
    Analyze the user's input and classify their intent into exactly one of the following categories:
    - "SQL_QUERY": The user is asking to view data, lists, or explicitly mentions "table", "pivot table", "dataframe", or "rows and columns".
    - "PLOT_GEN": The user is asking for visualizations, plots, trend displays, or explicitly mentions "graph", "chart", "plot", or "histogram".
    [CRITICAL RULE]: If the user mentions "Pivot table", this usually means they want a cross-tabulated data table. You MUST classify this as "SQL_QUERY" and NOT "PLOT_GEN".
    - "GENERAL_CHAT": The user is just saying hello, asking general non-data questions, or seeking help.

    You MUST respond in valid JSON format with the following schema:
    {
        "intent": "SQL_QUERY" | "PLOT_GEN" | "GENERAL_CHAT",
        "reasoning": "Brief explanation of why you chose this intent"
    }
    '''
    
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        # 【修改这里】将历史记录拼接到给模型的输入中
        combined_input = f"{system_prompt}\n\n[Conversation History]:\n{chat_history}\n\n[Current User Input]:\n{user_prompt}"
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=combined_input,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        if not response.text:
            raise ValueError("Received empty response from LLM.")
        return json.loads(response.text)
    except Exception as e:
        return {"intent": "ERROR", "reasoning": str(e)}


def generate_duckdb_sql(user_prompt: str, filter_context: str, schema_context: str, error_feedback: Optional[str] = None, chat_history: str = "") -> dict:
    """
    Translates natural language into a valid, read-only DuckDB SQL query.
    [Phase 8 Refactored] Now accepts dynamic schema_context to prevent schema drift.
    """
    system_prompt = f'''
    You are an expert Actuarial Data Scientist and DuckDB SQL developer.
    Translate the user's natural language question into a valid DuckDB SQL query.

    # Global Filters Context
    {filter_context}

    # Data Schema
    Table Name: current_working_set
    
    {schema_context}

    # Derived Metrics Logic (Crucial)
    - A/E (Count) = SUM(Death_Count) / NULLIF(SUM(ExpDth_VBT2015_Cnt), 0)
    - A/E (Amount) = SUM(Death_Claim_Amount) / NULLIF(SUM(ExpDth_VBT2015_Amt), 0)

    # Strict Rules:
    1. You MUST use the exact string `current_working_set` as the table name in your FROM clause. Do NOT use {{source_table}} or "mortality".
    2. ONLY use the exact column names provided in the Data Schema above. Do NOT hallucinate columns.
    3. ALWAYS use read-only SELECT statements. Do NOT generate INSERT, UPDATE, or DROP.
    4. Always use NULLIF for denominators to prevent division by zero errors.

    You MUST respond in valid JSON format with the following schema:
    {{
        "sql": "The complete DuckDB SQL query string using current_working_set",
        "explanation": "Brief explanation of how the query works"
    }}
    '''
    
    # === [Phase 6] Self-Correction Injection ===
    if error_feedback:
        system_prompt += f"\n\n[CRITICAL CORRECTION REQUIRED]: Your previous attempt failed with the following error:\n{error_feedback}\nPlease fix the SQL syntax or logic error and return the corrected JSON."
    
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        # 【修改这里】将历史记录拼接到给模型的输入中
        combined_input = f"{system_prompt}\n\n[Conversation History]:\n{chat_history}\n\n[Current User Input]:\n{user_prompt}"
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=combined_input,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        if not response.text:
            raise ValueError("Received empty response from LLM.")
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

def generate_plot_config(user_prompt: str, filter_context: str, schema_context: str, error_feedback: Optional[str] = None, chat_history: str = "") -> dict:
    """
    Translates natural language into both a SQL query for data aggregation
    and configuration parameters for Plotly visualization.
    """
    system_prompt = f'''
    You are an expert Actuarial Data Scientist and Data Visualization Specialist.
    The user wants to generate a chart based on the DuckDB mortality database.

    # Global Filters Context
    {filter_context}

    # Data Schema
    Table Name: current_working_set
    
    {schema_context}

    # Derived Metrics Logic (IMPORTANT)
    - AE_Count = SUM(Death_Count) / NULLIF(SUM(ExpDth_VBT2015_Cnt), 0)
    - AE_Amount = SUM(Death_Claim_Amount) / NULLIF(SUM(ExpDth_VBT2015_Amt), 0)

    # Strict Rules for SQL Generation:
    1. You MUST use the exact string `current_working_set` as the table name in your FROM clause. Do NOT use {{source_table}} or "mortality".
    2. ONLY use the exact column names provided in the Data Schema above.
    3. Always use read-only SELECT statements.
    4. If the user asks to split, group, or color the chart by a categorical column (like Sex or Insurance_Plan), you MUST pivot that category in your SQL using CASE WHEN (e.g., `SUM(CASE WHEN Sex='F' THEN Policies_Exposed ELSE 0 END) AS F_Policies_Exposed`). The final SQL output must have ONE column for the X-axis and MULTIPLE separate columns for the Y-axis.
    5. If the user specifies a 'Bin size' for a numeric X-axis (e.g., Age), you MUST format the resulting bin as a string range. 
       Example DuckDB SQL for Bin size 5: `CAST(CAST(FLOOR(Attained_Age/5)*5 AS INT) AS VARCHAR) || '-' || CAST(CAST(FLOOR(Attained_Age/5)*5 + 4 AS INT) AS VARCHAR)`.
    6. Ensure your SQL column aliases are human-readable and presentation-ready.

    # Task
    1. Write a DuckDB SQL query to aggregate the data.
    2. Determine the best Plotly chart type: "bar", "line", "pie", or "combo". 
    3. Identify the exact column names from your generated SQL output that map to the X and Y axes. 

    You MUST respond in valid JSON format:
    {{
        "sql": "DuckDB SELECT query using current_working_set",
        "chart_type": "bar" | "line" | "pie" | "combo",
        "x_axis": "Exact column name for X axis",
        "y_axis_bars": ["List of column names for bar charts (leave empty if none)"],
        "y_axis_lines": ["List of column names for line charts (leave empty if none)"],
        "title": "A clear title for the chart",
        "explanation": "Brief explanation of what the chart shows"
    }}
    '''
    
    if error_feedback:
        system_prompt += f"\n\n[CRITICAL CORRECTION REQUIRED]: Your previous attempt failed with the following error:\n{error_feedback}\nPlease fix the configuration or SQL error and return the corrected JSON."
    
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        # 【修改这里】将历史记录拼接到给模型的输入中
        combined_input = f"{system_prompt}\n\n[Conversation History]:\n{chat_history}\n\n[Current User Input]:\n{user_prompt}"
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=combined_input,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        if not response.text:
            raise ValueError("Received empty response from LLM.")
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}
    

def execute_read_only_sql(sql: str, data_path: str, use_global_filters: bool) -> pd.DataFrame:
    """
    [Phase 6 Refactored]
    Creates a Temporary View matching UI filters, then safely executes the LLM-generated SQL against it.
    """
    sql_clean = sql.strip().upper()
    
    # 1. Handle Empty/Missing SQL
    if not sql_clean or sql_clean == "N/A":
        raise ValueError("Column Mismatch: The AI could not generate a query because the requested fields are not in the database. Please check the Field Reference.")
        
    # 2. Security Check & Sandbox Firewall
    if not sql_clean.startswith("SELECT") and not sql_clean.startswith("WITH"):
        raise ValueError("Invalid Query Format: The AI generated an invalid response instead of a SQL query. Please rephrase your question.")
    
    # 定义危险操作和外部访问函数黑名单
    malicious_keywords = [
        "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "GRANT", "TRUNCATE", # DDL/DML
        "READ_CSV", "READ_PARQUET", "READ_JSON", "WRITE_CSV", "WRITE_PARQUET", "COPY", # File I/O
        "PRAGMA", "INSTALL", "LOAD", "ATTACH", "DETACH", "SYSTEM" # System/Config commands
    ]
    
    # 使用正则匹配独立单词，防止误伤（例如列名恰好叫 "UPDATE_TIME" 或 "DROP_RATE"）
    for kw in malicious_keywords:
        if re.search(r'\b' + kw + r'\b', sql_clean):
            raise ValueError(f"Security Restriction: For data safety, the keyword '{kw}' is strictly forbidden.")
    
    # 3. Dynamic Source Resolution
    safe_path = data_path.replace("'", "''")
    if safe_path.lower().endswith('.csv'):
         source_expr = f"read_csv_auto('{safe_path}', header=true)"
    else:
         source_expr = f"read_parquet('{safe_path}')"
         
    # 4. Build the Temporary View Definition
    view_sql = f"CREATE TEMP VIEW current_working_set AS SELECT * FROM {source_expr}"
    
    if use_global_filters:
        cat_filters = st.session_state.get("active_cat_filters", {})
        num_ranges = st.session_state.get("active_num_ranges", {})
        clauses = []
        
        for col, vals in cat_filters.items():
            if vals:
                # Need to handle missing values nicely or strictly match strings
                val_str = ", ".join([f"'{v}'" for v in vals if v != "(Missing)"])
                if val_str:
                    clauses.append(f"{col} IN ({val_str})")
                    
        for col, (lo, hi) in num_ranges.items():
            clauses.append(f"TRY_CAST({col} AS DOUBLE) BETWEEN {lo} AND {hi}")
            
        if clauses:
            view_sql += " WHERE " + " AND ".join(clauses)

    # 5. Execution with Friendly Error Catching
    con = duckdb.connect()
    try:
        # Create the sandbox view first
        con.execute(view_sql)
        # Execute the LLM's query
        return con.execute(sql).df()
    except Exception as e:
        error_str = str(e).lower()
        if "column" in error_str and "not found" in error_str:
            friendly_msg = "Column Mismatch: Some requested data fields do not exist in the current database schema."
        elif "divide by zero" in error_str:
            friendly_msg = "Calculation Error: Division by zero encountered. This usually happens when the Expected Deaths evaluate to 0."
        elif "syntax error" in error_str or "parser error" in error_str:
            friendly_msg = f"SQL Syntax Error: The generated query contains a logical flaw.\n\n*Details:* `{str(e)}`"
        else:
            friendly_msg = f"Database Execution Error:\n\n`{str(e)}`"
            
        raise ValueError(friendly_msg)
    finally:
        con.close()


def render_ai_assistant_tab(schema_context: str) -> None:
    """
    The main UI renderer for the AI Assistant tab in Streamlit.
    """
    st.header("🤖 AI Actuarial Assistant")
    st.caption("Query actuarial mortality data using natural language (Phase 5: Refinement & Polish)")

    # === [新增 Phase 5] 全局过滤同步开关 ===
    use_global_filters = st.toggle(
        "Sync with Preliminary filters", 
        value=True, 
        help="If enabled, AI will only query the subset of data defined in the Preliminary filters tab."
    )
    st.markdown("---")

    # 1. Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your Actuarial Data Assistant. You can ask me anything about the mortality data (e.g., 'What was the overall A/E Count ratio for Males last year?')."}
        ]

    chat_container = st.container()

    # 2. Render chat history (including stored DataFrames)
    for msg in st.session_state.messages:
        with chat_container.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg:
                st.dataframe(msg["df"], width="stretch")
            if "fig" in msg:
                st.plotly_chart(msg["fig"], width="stretch")

    # 3. Receive User Input
    if prompt := st.chat_input("Enter your actuarial data query..."):
        
        # Display user message immediately
        with chat_container.chat_message("user"):
            st.markdown(prompt)
            
        # 先保存用户输入到状态中
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 【新增这行】提取历史记录
        chat_history_str = get_recent_chat_history(st.session_state.messages)

        # Process the request
        with chat_container.chat_message("assistant"):
            
            # Step A: Intent Parsing
            with st.spinner("Analyzing intent..."):
                # 【修改这行】传入 chat_history_str
                intent_result = parse_user_intent(prompt, chat_history_str)
            
            intent = intent_result.get("intent")
            
            if intent == "ERROR":
                err_text = f"Parsing failed: {intent_result.get('reasoning')}"
                st.error(err_text)
                st.session_state.messages.append({"role": "assistant", "content": err_text})
            
            # Step B: SQL Pipeline
            elif intent == "SQL_QUERY":
                with st.spinner("Writing DuckDB query & fetching data..."):
                    filter_context = get_active_filters_context(use_global_filters)
                    data_path = st.session_state.get("data_path")
                    
                    max_retries = 1
                    error_feedback = None
                    
                    for attempt in range(max_retries + 1):
                        # 【修改这行】增加 chat_history_str 参数传递
                        sql_payload = generate_duckdb_sql(prompt, filter_context, schema_context, error_feedback, chat_history_str)
                        
                        if "error" in sql_payload:
                            err_text = f"Failed to generate SQL: {sql_payload['error']}"
                            st.error(err_text)
                            st.session_state.messages.append({"role": "assistant", "content": err_text})
                            break
                        
                        gen_sql = sql_payload.get("sql", "")
                        explanation = sql_payload.get("explanation", "")
                        
                        try:
                            if not isinstance(data_path, str):
                                raise ValueError("System error: Valid data_path not found in session state.")

                            result_df = execute_read_only_sql(gen_sql, data_path, use_global_filters)
                            
                            # === 成功拦截 ===
                            if result_df.empty:
                                warn_msg = f"**No Data Found.**\n\nYour query executed successfully, but returned 0 rows. This is likely due to the current Preliminary filters being too restrictive.\n\n*Attempted SQL:*\n```sql\n{gen_sql}\n```"
                                st.warning(warn_msg)
                                st.session_state.messages.append({"role": "assistant", "content": warn_msg})
                            else:
                                debug_note = "\n\n*(Self-corrected after initial error)*" if attempt > 0 else ""
                                resp_text = f"**Explanation:** {explanation}{debug_note}\n\n```sql\n{gen_sql}\n```"
                                
                                st.markdown(resp_text)
                                st.dataframe(result_df, width="stretch")
                                
                                csv_bytes = result_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Data as CSV",
                                    data=csv_bytes,
                                    file_name="ai_query_result.csv",
                                    mime="text/csv",
                                    key=f"download_{len(st.session_state.messages)}" # 保证每次循环生成的 key 唯一
                                )

                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": resp_text, 
                                    "df": result_df
                                })
                            break # 跳出重试循环
                            
                        except Exception as e:
                            # 捕获失败，准备下一轮反射
                            if attempt < max_retries:
                                error_feedback = f"Error: {str(e)}\nAttempted SQL:\n{gen_sql}"
                                # 循环继续，带上错误信息重新请求大模型
                            else:
                                err_msg = f"**Database Execution Error (After retry):** `{str(e)}`\n\n*Final Attempted SQL:*\n```sql\n{gen_sql}\n```"
                                st.error(err_msg)
                                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            
            # Step C: Plot Generation Pipeline
            elif intent == "PLOT_GEN":
                with st.spinner("Designing chart and fetching data..."):
                    filter_context = get_active_filters_context(use_global_filters)
                    data_path = st.session_state.get("data_path")
                    
                    max_retries = 1
                    error_feedback = None
                    
                    for attempt in range(max_retries + 1):
                        # 【修改这行】增加 chat_history_str 参数传递
                        plot_payload = generate_plot_config(prompt, filter_context, schema_context, error_feedback, chat_history_str)
                        
                        if "error" in plot_payload:
                            err_text = f"Failed to generate plot config: {plot_payload['error']}"
                            st.error(err_text)
                            st.session_state.messages.append({"role": "assistant", "content": err_text})
                            break
                            
                        gen_sql = plot_payload.get("sql", "")
                        chart_type = plot_payload.get("chart_type", "bar")
                        x_col = plot_payload.get("x_axis")
                        y_bars = plot_payload.get("y_axis_bars", [])
                        y_lines = plot_payload.get("y_axis_lines", [])
                        title = plot_payload.get("title", "Generated Chart")
                        explanation = plot_payload.get("explanation", "")
                        
                        try:
                            # 1. Fetch data
                            if not isinstance(data_path, str):
                                raise ValueError("System error: Valid data_path not found in session state.")
                                
                            result_df = execute_read_only_sql(gen_sql, data_path, use_global_filters)
                            
                            # === 空数据拦截 ===
                            if result_df.empty:
                                warn_msg = f"**No Data Available for Chart.**\n\nThe query returned 0 rows, likely due to the current Preliminary filters. Please adjust the filters and try again.\n\n*Attempted SQL:*\n```sql\n{gen_sql}\n```"
                                st.warning(warn_msg)
                                st.session_state.messages.append({"role": "assistant", "content": warn_msg})
                            else:
                                # 2. Build Advanced Plotly figure
                                fig = make_subplots(specs=[[{"secondary_y": True}]])
                                
                                # Add Bar traces (Primary Y-axis)
                                for y_col in y_bars:
                                    if y_col in result_df.columns:
                                        fig.add_trace(
                                            go.Bar(x=result_df[x_col].astype(str), y=result_df[y_col], name=y_col),
                                            secondary_y=False
                                        )
                                        
                                # Add Line traces (Secondary Y-axis)
                                for y_col in y_lines:
                                    if y_col in result_df.columns:
                                        fig.add_trace(
                                            go.Scatter(x=result_df[x_col].astype(str), y=result_df[y_col], name=y_col, mode='lines+markers'),
                                            secondary_y=True if chart_type == "combo" else False
                                        )
                                        
                                # Formatting
                                fig.update_layout(
                                    title=title, 
                                    barmode='group',
                                    hovermode="x unified",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                fig.update_xaxes(title_text=x_col)
                                
                                if chart_type == "combo":
                                    fig.update_yaxes(title_text="Volume (Bars)", secondary_y=False)
                                    fig.update_yaxes(title_text="Ratio (Lines)", secondary_y=True, tickformat=".2f")
                                    
                                # 3. Build UI Response
                                debug_note = "\n\n*(Self-corrected after initial error)*" if attempt > 0 else ""
                                resp_text = f"**Explanation:** {explanation}{debug_note}\n\n*Generated SQL:*\n```sql\n{gen_sql}\n```"
                                
                                st.markdown(resp_text)
                                st.plotly_chart(fig, width="stretch")
                                
                                csv_bytes = result_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Chart Data as CSV",
                                    data=csv_bytes,
                                    file_name="ai_chart_data.csv",
                                    mime="text/csv",
                                    key=f"dl_plot_{len(st.session_state.messages)}"
                                )

                                # 4. Persist figure to state
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": resp_text, 
                                    "fig": fig
                                })
                            break # 成功执行，跳出循环！
                            
                        except Exception as e:
                            # 捕获失败，准备下一轮反射
                            if attempt < max_retries:
                                error_feedback = f"Error: {str(e)}\nAttempted SQL:\n{gen_sql}\nx_axis: {x_col}, bars: {y_bars}, lines: {y_lines}"
                            else:
                                err_msg = f"**Plot Generation Error (After retry):** `{str(e)}`\n\n*Attempted Config:*\n```json\n{json.dumps(plot_payload, indent=2)}\n```"
                                st.error(err_msg)
                                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            
            # Step D: General Chat Fallback
            elif intent == "GENERAL_CHAT":
                resp_text = "I am ready to help you analyze mortality data. Try asking me a data question like 'Show me a bar chart of total death count by sex'."
                st.info(resp_text)
                st.session_state.messages.append({"role": "assistant", "content": resp_text})