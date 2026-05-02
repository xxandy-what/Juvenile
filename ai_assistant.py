import json
import duckdb
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types


def parse_user_intent(user_prompt: str) -> dict:
    """
    Calls the LLM to parse user intent and returns a JSON dictionary.
    """
    system_prompt = '''
    You are an intelligent Actuarial Data Assistant routing engine.
    Analyze the user's input and classify their intent into exactly one of the following categories:
    - "SQL_QUERY": The user is asking a data question that requires querying the duckdb mortality database.
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
    Translates natural language into a valid DuckDB SQL query based on the Schema.
    """
    system_prompt = '''
    You are an expert Actuarial Data Scientist and DuckDB SQL developer.
    Translate the user's natural language question into a valid DuckDB SQL query.

    # Data Schema
    Table Name: {source_table}

    Categorical Columns:
    - Observation_Year (2012-2019)
    - Age_Ind (0=ANB, 1=ALB, 2=Age Next Birthday)
    - Sex (F, M)
    - Smoker_Status (N, S, U)
    - Insurance_Plan (Perm, Term, UL, ULSG, VL, VLSG, Other)
    - Face_Amount_Band (01 to 11)
    - SOA_Post_Lvl_Ind (N/A, NLT, PLT, ULT, WLT)
    - Preferred_Class (1, 2, 3, 4, NA, U)

    Numeric Columns:
    - Issue_Age, Duration, Issue_Year, Attained_Age
    - Amount_Exposed, Policies_Exposed
    - Death_Claim_Amount, Death_Count
    - ExpDth_VBT2015_Cnt, ExpDth_VBT2015_Amt

    # Derived Metrics Logic
    - A/E (Count) = SUM(Death_Count) / NULLIF(SUM(ExpDth_VBT2015_Cnt), 0)
    - A/E (Amount) = SUM(Death_Claim_Amount) / NULLIF(SUM(ExpDth_VBT2015_Amt), 0)

    # Strict Rules:
    1. ONLY use the columns listed above.
    2. ALWAYS use read-only SELECT statements.
    3. Use NULLIF to prevent division by zero.

    You MUST respond in valid JSON format:
    {
        "sql": "The complete DuckDB SQL query string",
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
    Safely executes the SQL query and returns the resulting DataFrame.
    """
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Security Error: Only SELECT statements are permitted.")
    
    safe_path = data_path.replace("'", "''")
    if safe_path.lower().endswith('.csv'):
         source_expr = f"read_csv_auto('{safe_path}', header=true)"
    else:
         source_expr = f"read_parquet('{safe_path}')"
         
    final_sql = sql.replace("{source_table}", source_expr)
    
    con = duckdb.connect()
    try:
        return con.execute(final_sql).df()
    finally:
        con.close()


def render_ai_assistant_tab() -> None:
    st.header("🤖 AI Actuarial Assistant")
    st.caption("Query actuarial mortality data using natural language (Phase 3: Text-to-SQL)")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your Actuarial Data Assistant. You can ask me anything about the mortality data, or ask me to generate charts."}
        ]

    chat_container = st.container()

    # Render chat history
    for msg in st.session_state.messages:
        with chat_container.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg:
                st.dataframe(msg["df"], width="stretch")
            if "fig" in msg:
                st.plotly_chart(msg["fig"], width="stretch")

    # Receive user input
    if prompt := st.chat_input("Enter your data query request..."):
        
        # Display and save user input in the chat_container
        with chat_container.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Render assistant response in the chat_container
        with chat_container.chat_message("assistant"):
            # 1. Parse Intent
            with st.spinner("Analyzing your intent..."):
                intent_result = parse_user_intent(prompt)
            
            intent = intent_result.get("intent")
            
            if intent == "ERROR":
                st.error(f"Parsing failed: {intent_result.get('reasoning')}")
                st.session_state.messages.append({"role": "assistant", "content": "Error parsing intent."})
            
            # 2. Process Data Query (Phase 3)
            elif intent == "SQL_QUERY":
                with st.spinner("Generating and executing SQL..."):
                    sql_payload = generate_duckdb_sql(prompt)
                    
                    if "error" in sql_payload:
                        st.error(f"Failed to generate SQL: {sql_payload['error']}")
                    else:
                        gen_sql = sql_payload.get("sql", "")
                        explanation = sql_payload.get("explanation", "")
                        
                        try:
                            data_path = st.session_state.get("data_path")
                            if not isinstance(data_path, str):
                                raise ValueError("Data path is missing or invalid in session state.")
                            result_df = execute_read_only_sql(gen_sql, data_path)
                            
                            # Display results
                            resp_text = f"**Explanation:** {explanation}\n\n```sql\n{gen_sql}\n```"
                            st.markdown(resp_text)
                            st.dataframe(result_df, width="stretch")
                            
                            # Save state
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": resp_text, 
                                "df": result_df
                            })
                        except Exception as e:
                            err_msg = f"Data Execution Error: `{str(e)}`\n\n**Attempted to execute SQL:**\n```sql\n{gen_sql}\n```"
                            st.error(err_msg)
                            st.session_state.messages.append({"role": "assistant", "content": err_msg})
            
            # 3. Placeholder for other intents
            else:
                reasoning = intent_result.get("reasoning", "")
                resp_text = f"**Intent Detected:** `{intent}`\n\n**Reasoning:** {reasoning}\n\n*(Implementation for this intent is coming in Phase 4)*"
                st.info(resp_text)
                st.session_state.messages.append({"role": "assistant", "content": resp_text})