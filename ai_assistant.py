import json
import duckdb
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

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


def generate_duckdb_sql(user_prompt: str) -> dict:
    """
    Translates natural language into a valid, read-only DuckDB SQL query based on the actuarial schema.
    """
    system_prompt = '''
    You are an expert Actuarial Data Scientist and DuckDB SQL developer.
    Translate the user's natural language question into a valid DuckDB SQL query.

    # Data Schema
    Table Name: {source_table}

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
    {
        "sql": "The complete DuckDB SQL query string using {source_table}",
        "explanation": "Brief explanation of how the query works"
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
        return {"error": str(e)}


def execute_read_only_sql(sql: str, data_path: str) -> pd.DataFrame:
    """
    Safely executes the LLM-generated SQL query against the underlying Parquet/CSV file.
    """
    # 1. Security Check
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Security Violation: Only SELECT statements are permitted.")
    
    # 2. Dynamic Table Path Resolution
    safe_path = data_path.replace("'", "''")
    if safe_path.lower().endswith('.csv'):
         source_expr = f"read_csv_auto('{safe_path}', header=true)"
    else:
         source_expr = f"read_parquet('{safe_path}')"
         
    final_sql = sql.replace("{source_table}", source_expr)
    
    # 3. Execution
    con = duckdb.connect()
    try:
        return con.execute(final_sql).df()
    finally:
        con.close()


def render_ai_assistant_tab() -> None:
    """
    The main UI renderer for the AI Assistant tab in Streamlit.
    """
    st.header("🤖 AI Actuarial Assistant")
    st.caption("Query actuarial mortality data using natural language (Phase 3: Text-to-SQL Pipeline)")

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
            
            # Step B: SQL Pipeline (Phase 3 Core)
            elif intent == "SQL_QUERY":
                with st.spinner("Writing DuckDB query & fetching data..."):
                    sql_payload = generate_duckdb_sql(prompt)
                    
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
            
            # Step C: Placeholder for future phases
            else:
                reasoning = intent_result.get("reasoning", "")
                resp_text = f"**Intent Detected:** `{intent}`\n\n**Reasoning:** {reasoning}\n\n*(Feature unlock pending: Plot generation is scheduled for Phase 4.)*"
                st.info(resp_text)
                st.session_state.messages.append({"role": "assistant", "content": resp_text})