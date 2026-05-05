import json
import duckdb
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai
from google.genai import types


def get_active_filters_context(use_global_filters: bool) -> str:
    """
    根据用户选择，读取 Streamlit session state 中的初步过滤条件并生成 LLM 提示词上下文。
    """
    if not use_global_filters:
        return "The user wants to query the FULL dataset. Do NOT apply any preliminary filters implicitly."

    cat_filters = st.session_state.get("active_cat_filters", {})
    num_ranges = st.session_state.get("active_num_ranges", {})
    
    context_parts = []
    
    for col, vals in cat_filters.items():
        if vals:
            val_str = ", ".join([f"'{v}'" for v in vals])
            context_parts.append(f"- {col} IN ({val_str})")
            
    for col, (lo, hi) in num_ranges.items():
        context_parts.append(f"- {col} BETWEEN {lo} AND {hi}")
        
    if not context_parts:
        return "No preliminary filters applied. Query the full dataset."
        
    return "The user has applied the following global preliminary filters. You MUST include these conditions in your WHERE clause:\n" + "\n".join(context_parts)


def parse_user_intent(user_prompt: str) -> dict:
    """
    Calls the Gemini API to parse user intent and returns a JSON dictionary.
    """
    system_prompt = '''
    You are an intelligent Actuarial Data Assistant routing engine.
    Analyze the user's input and classify their intent into exactly one of the following categories:
    - "SQL_QUERY": The user is asking a data question that requires querying the DuckDB mortality database.
    - "PLOT_GEN": The user is asking to visualize data or generate a chart/plot.
    - "GENERAL_CHAT": The user is just saying hello, asking general non-data questions, or seeking help.

    You MUST respond in valid JSON format with the following schema:
    {
        "intent": "SQL_QUERY" | "PLOT_GEN" | "GENERAL_CHAT",
        "reasoning": "Brief explanation of why you chose this intent"
    }
    '''
    
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"{system_prompt}\n\nUser Input: {user_prompt}",
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


def generate_duckdb_sql(user_prompt: str, filter_context: str) -> dict:
    """
    Translates natural language into a valid, read-only DuckDB SQL query based on the actuarial schema.
    """
    system_prompt = f'''
    You are an expert Actuarial Data Scientist and DuckDB SQL developer.
    Translate the user's natural language question into a valid DuckDB SQL query.

    # Global Filters Context
    {filter_context}

    # Data Schema
    Table Name: {{source_table}}

    Categorical Columns (Use for GROUP BY or WHERE):
    - Observation_Year (2012-2019)
    - Age_Ind (0=ANB, 1=ALB, 2=Age Next Birthday)
    - Sex (F = Female, M = Male)
    - Smoker_Status (N = Nonsmoker, S = Smoker, U = Uni-smoke)
    - Insurance_Plan (Perm, Term, UL, ULSG, VL, VLSG, Other)
    - Face_Amount_Band (01 to 11)
    - SOA_Post_Lvl_Ind (N/A, NLT, PLT, ULT, WLT)
    - Preferred_Class (1, 2, 3, 4, NA, U)

    Numeric Columns (Use for aggregations like SUM, AVG):
    - Issue_Age, Duration, Issue_Year, Attained_Age
    - Amount_Exposed, Policies_Exposed
    - Death_Claim_Amount, Death_Count
    - ExpDth_VBT2015_Cnt, ExpDth_VBT2015_Amt

    # Derived Metrics Logic (Crucial)
    - A/E (Count) = SUM(Death_Count) / NULLIF(SUM(ExpDth_VBT2015_Cnt), 0)
    - A/E (Amount) = SUM(Death_Claim_Amount) / NULLIF(SUM(ExpDth_VBT2015_Amt), 0)

    # Strict Rules:
    1. ONLY use the exact column names listed above. Do NOT hallucinate columns.
    2. ALWAYS use read-only SELECT statements. Do NOT generate INSERT, UPDATE, or DROP.
    3. Always use NULLIF for denominators to prevent division by zero errors.

    You MUST respond in valid JSON format with the following schema:
    {{
        "sql": "The complete DuckDB SQL query string using {{source_table}}",
        "explanation": "Brief explanation of how the query works"
    }}
    '''
    
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"{system_prompt}\n\nUser Input: {user_prompt}",
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

def generate_plot_config(user_prompt: str, filter_context: str) -> dict:
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
    Table Name: {{source_table}}

    Categorical Columns: Observation_Year, Age_Ind, Sex, Smoker_Status, Insurance_Plan, Face_Amount_Band, SOA_Post_Lvl_Ind, Preferred_Class
    Numeric Columns: Issue_Age, Duration, Issue_Year, Attained_Age, Amount_Exposed, Policies_Exposed, Death_Claim_Amount, Death_Count, ExpDth_VBT2015_Cnt, ExpDth_VBT2015_Amt

    # Derived Metrics Logic (IMPORTANT)
    - AE_Count = SUM(Death_Count) / NULLIF(SUM(ExpDth_VBT2015_Cnt), 0)
    - AE_Amount = SUM(Death_Claim_Amount) / NULLIF(SUM(ExpDth_VBT2015_Amt), 0)

    # Strict Rules for SQL Generation:
    1. You MUST use the exact string `{{source_table}}` as the table name in your FROM clause. Do NOT use "mortality".
    2. ONLY use the exact column names listed above.
    3. Always use read-only SELECT statements.

    # Task
    1. Write a DuckDB SQL query to aggregate the data.
    2. Determine the best Plotly chart type: "bar", "line", "pie", or "combo". 
       - Use "combo" if the user asks for both volume (e.g., Death_Count) and ratio (e.g., A/E) on the same chart.
    3. Identify the exact column names from your generated SQL output that map to the X and Y axes. 

    You MUST respond in valid JSON format:
    {{
        "sql": "DuckDB SELECT query using {{source_table}}",
        "chart_type": "bar" | "line" | "pie" | "combo",
        "x_axis": "Exact column name for X axis",
        "y_axis_bars": ["List of column names for bar charts (leave empty if none)"],
        "y_axis_lines": ["List of column names for line charts (leave empty if none)"],
        "title": "A clear title for the chart",
        "explanation": "Brief explanation of what the chart shows"
    }}
    '''
    
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"{system_prompt}\n\nUser Input: {user_prompt}",
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
    

def execute_read_only_sql(sql: str, data_path: str) -> pd.DataFrame:
    """
    Safely executes the LLM-generated SQL query against the underlying Parquet/CSV file,
    with humanized, professional English error handling for a better user experience.
    """
    sql_clean = sql.strip().upper()
    
    # 1. Handle Empty/Missing SQL (LLM gave up because columns don't exist)
    if not sql_clean or sql_clean == "N/A":
        raise ValueError("Column Mismatch: The AI could not generate a query because the requested fields (e.g., profit, region) are not in the database. Please check the Field Reference.")
        
    # 2. Security Check (Differentiate malicious requests from malformed ones)
    if not sql_clean.startswith("SELECT"):
        malicious_keywords = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "GRANT", "TRUNCATE"]
        if any(sql_clean.startswith(kw) for kw in malicious_keywords):
            raise ValueError("Security Restriction: For data safety, only read-only SELECT queries are permitted. Data modification or system-level requests are blocked.")
        else:
            raise ValueError("Invalid Query Format: The AI generated an invalid response instead of a SQL query. Please rephrase your question.")
    
    # 3. Dynamic Table Path Resolution
    safe_path = data_path.replace("'", "''")
    if safe_path.lower().endswith('.csv'):
         source_expr = f"read_csv_auto('{safe_path}', header=true)"
    else:
         source_expr = f"read_parquet('{safe_path}')"
         
    final_sql = sql.replace("{source_table}", source_expr)
    
    # 4. Execution with Friendly Error Catching
    con = duckdb.connect()
    try:
        return con.execute(final_sql).df()
    except Exception as e:
        error_str = str(e).lower()
        
        # Translate hard DuckDB errors into business-friendly logic
        if "column" in error_str and "not found" in error_str:
            friendly_msg = "Column Mismatch: Some requested data fields do not exist in the current actuarial database. Please refer to the Field Reference for available columns."
        elif "divide by zero" in error_str:
            friendly_msg = "Calculation Error: Division by zero encountered. This usually happens when the Expected Deaths for certain groups evaluate to 0."
        elif "out of memory" in error_str or "message size" in error_str:
            friendly_msg = "Payload Too Large: The query attempts to return too many rows. Please ask for aggregated data (e.g., using SUM or COUNT) or apply stricter Preliminary filters."
        elif "syntax error" in error_str or "parser error" in error_str:
            friendly_msg = f"SQL Syntax Error: The generated query contains a logical flaw and failed to execute.\n\n*Details:* `{str(e)}`"
        else:
            friendly_msg = f"Database Execution Error:\n\n`{str(e)}`"
            
        raise ValueError(friendly_msg)
    finally:
        con.close()


def render_ai_assistant_tab() -> None:
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
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process the request
        with chat_container.chat_message("assistant"):
            
            # Step A: Intent Parsing
            with st.spinner("Analyzing intent..."):
                intent_result = parse_user_intent(prompt)
            
            intent = intent_result.get("intent")
            
            if intent == "ERROR":
                err_text = f"Parsing failed: {intent_result.get('reasoning')}"
                st.error(err_text)
                st.session_state.messages.append({"role": "assistant", "content": err_text})
            
            # Step B: SQL Pipeline
            elif intent == "SQL_QUERY":
                with st.spinner("Writing DuckDB query & fetching data..."):
                    # 获取上下文并传给 LLM
                    filter_context = get_active_filters_context(use_global_filters)
                    sql_payload = generate_duckdb_sql(prompt, filter_context)
                    
                    if "error" in sql_payload:
                        err_text = f"Failed to generate SQL: {sql_payload['error']}"
                        st.error(err_text)
                        st.session_state.messages.append({"role": "assistant", "content": err_text})
                    else:
                        gen_sql = sql_payload.get("sql", "")
                        explanation = sql_payload.get("explanation", "")
                        
                        try:
                            # Retrieve the current dataset path from appv2.py state
                            data_path = st.session_state.get("data_path")
                            if not isinstance(data_path, str):
                                raise ValueError("System error: Valid data_path not found in session state.")
                            
                            # Execute query
                            result_df = execute_read_only_sql(gen_sql, data_path)
                            
                            # === [新增 Phase 5] 空数据拦截 ===
                            if result_df.empty:
                                warn_msg = f"**No Data Found.**\n\nYour query executed successfully, but returned 0 rows. This is likely due to the current Preliminary filters being too restrictive.\n\n*Attempted SQL:*\n```sql\n{gen_sql}\n```"
                                st.warning(warn_msg)
                                st.session_state.messages.append({"role": "assistant", "content": warn_msg})
                            else:
                                # Build response markdown
                                resp_text = f"**Explanation:** {explanation}\n\n```sql\n{gen_sql}\n```"
                                
                                # Render UI
                                st.markdown(resp_text)
                                st.dataframe(result_df, width="stretch")
                                
                                # Persist to history
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": resp_text, 
                                    "df": result_df
                                })
                            
                        except Exception as e:
                            err_msg = f"**Database Execution Error:** `{str(e)}`\n\n*Attempted SQL:*\n```sql\n{gen_sql}\n```"
                            st.error(err_msg)
                            st.session_state.messages.append({"role": "assistant", "content": err_msg})
            
            # Step C: Plot Generation Pipeline
            elif intent == "PLOT_GEN":
                with st.spinner("Designing chart and fetching data..."):
                    # 获取上下文并传给 LLM
                    filter_context = get_active_filters_context(use_global_filters)
                    plot_payload = generate_plot_config(prompt, filter_context)
                    
                    if "error" in plot_payload:
                        err_text = f"Failed to generate plot config: {plot_payload['error']}"
                        st.error(err_text)
                        st.session_state.messages.append({"role": "assistant", "content": err_text})
                    else:
                        gen_sql = plot_payload.get("sql", "")
                        chart_type = plot_payload.get("chart_type", "bar")
                        x_col = plot_payload.get("x_axis")
                        y_bars = plot_payload.get("y_axis_bars", [])
                        y_lines = plot_payload.get("y_axis_lines", [])
                        title = plot_payload.get("title", "Generated Chart")
                        explanation = plot_payload.get("explanation", "")
                        
                        try:
                            # 1. Fetch data
                            data_path = st.session_state.get("data_path")
                            if not isinstance(data_path, str):
                                raise ValueError("System error: Valid data_path not found in session state.")
                            result_df = execute_read_only_sql(gen_sql, data_path)
                            
                            # === [新增 Phase 5] 空数据拦截 ===
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
                                resp_text = f"**Explanation:** {explanation}\n\n*Generated SQL:*\n```sql\n{gen_sql}\n```"
                                st.markdown(resp_text)
                                st.plotly_chart(fig, width="stretch")
                                
                                # 4. Persist figure to state
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": resp_text, 
                                    "fig": fig
                                })
                            
                        except Exception as e:
                            err_msg = f"**Plot Generation Error:** `{str(e)}`\n\n*Attempted Config:*\n```json\n{json.dumps(plot_payload, indent=2)}\n```"
                            st.error(err_msg)
                            st.session_state.messages.append({"role": "assistant", "content": err_msg})
            
            # Step D: General Chat Fallback
            elif intent == "GENERAL_CHAT":
                resp_text = "I am ready to help you analyze mortality data. Try asking me a data question like 'Show me a bar chart of total death count by sex'."
                st.info(resp_text)
                st.session_state.messages.append({"role": "assistant", "content": resp_text})