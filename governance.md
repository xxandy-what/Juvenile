# Project Governance & Change Log

This document tracks major architectural decisions, version changes, and development progress to ensure transparency and traceability in team collaboration.

## 1. Versioning & Commit Standards

- **Feature Branches**: Use `feature/phaseX-description` naming convention.
- **Commit Messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat:`, `fix:`, `docs:`, `refactor:`).

## 2. Core Architecture Decisions (ADR)

- **[AD-001] AI Assistant Modularization**: To maintain the maintainability of `appv2.py`, all AI-related rendering, API calls, and SQL generation logic must be encapsulated in the `ai_assistant.py` module.
- **[AD-002] State Persistence**: Conversation history must be stored in `st.session_state.messages` to prevent data loss during Streamlit reruns or tab switching.
- **[AD-003] Automated CI/CD Quality Gates**: All Pull Requests and pushes to `main` must pass automated checks (Ruff & Mypy) via GitHub Actions to ensure code quality and prevent regression.
- **[AD-004] Centralized Tool Configuration**: Project-wide tool settings (e.g., file exclusions for Ruff/Mypy) must be managed within `pyproject.toml` to ensure consistency between local development and CI environments.

## 3. Change Log

### [2026-05-06] Phase 10: Agentic Data Analysis & Automated Reporting (Major Overhaul)

- **Status**: Completed
- **Author**: [SeanChen327]
- **Key Changes**:
  - **Autonomous Agent Pipeline**: Implemented a two-pass LLM architecture (Planner & Summarizer) to handle broad, exploratory questions (e.g., "What can you tell me about Duration?").
  - **Resilient API Integration**: Introduced `safe_llm_call` to automatically retry failed API requests (such as `503 UNAVAILABLE` and `429` rate limits) using exponential backoff, ensuring UI stability.
  - **Advanced Visualization Logic**:
    - Enforced the "Actuarial Gold Standard" charting rule (A/E ratios always paired with Death/Exposure Volumes).
    - Added multi-chart generation support (`charts` array) to handle metrics with vastly different scales seamlessly.
    - Enforced strict SQL pivoting (`CASE WHEN`) to eliminate Plotly "sawtooth" chart rendering anomalies.
  - **Deep Analysis Capabilities**:
    - Added correlation analysis using DuckDB's `CORR()` function for statistical insights.
    - Implemented an exploratory "Dashboard" mode (Scenario D) that dynamically generates 3 diverse charts for open-ended single-dimension queries.
    - Injected conversational memory (`chat_history`) into the Agentic Planner to support context-aware follow-up analysis.

---

### [2026-05-05] Phase 10: Agentic Data Analysis & Automated Reporting

- **Status**: Completed
- **Author**: [SeanChen327]
- **Key Changes**:
  - **Autonomous Agent Pipeline**: Implemented a two-pass LLM architecture (Planner & Summarizer) to handle broad, exploratory questions (e.g., "What is the trend of Duration?").
  - **Intent Routing Update**: Added `DATA_ANALYSIS` intent to `parse_user_intent` for routing open-ended analytical queries.
  - **Dynamic Report & Visualization**: The Agent autonomously writes DuckDB SQL, fetches aggregated data, renders a Plotly chart, and generates a comprehensive textual analysis report directly in the chat UI.

---

### [2026-05-04] Phase 9: Conversational Memory & Multi-turn Context

- **Status**: Completed
- **Author**: [SeanChen327]
- **Key Changes**:
  - **Multi-turn Memory**: Implemented `get_recent_chat_history` to pass previous dialogue turns to the Gemini API, enabling the AI to answer context-dependent follow-up questions.
  - **Payload Optimization**: Safely filtered out DataFrame and Plotly objects from the history payload to prevent token bloat and API errors.

---

### [2026-05-04] Phase 8: Dynamic Schema Injection & Anti-Hallucination

- **Status**: Completed
- **Author**: [SeanChen327]
- **Key Changes**:
  - **Dynamic Schema Context**: Replaced the static, hardcoded schema string in AI prompts with a real-time database scanning mechanism (`build_dynamic_schema_context`).
  - **Prompt Staleness Resolution**: The LLM now receives the exact categorical enumeration values and numeric ranges based on the live DuckDB state, effectively preventing hallucinations when underlying data changes (e.g., adding new observation years or removing insurance plans).
  - **Test Resilience**: Updated `test_wrapper_ai.py` with a mock schema context to ensure UI End-to-End tests remain completely isolated and pass cleanly despite the new dynamic architectural requirements.

---

### [2026-05-04] Phase 7: Automated Testing & Decoupling

- **Status**: Completed
- **Author**: [SeanChen327]
- **Key Changes**:
  - Established `pytest` framework with Unit, Mock, and E2E coverage.
  - Refactored `appv2.py`: Extracted core utilities to `utils.py` to achieve logical decoupling.
  - Resolved "Phantom Form" context bleed issues in Streamlit testing via Isolated Wrapper strategy.
  - Verified 100% test pass rate for SQL security firewall and LLM intent parsing.

---

### [2026-05-03] Phase 6: Architecture Hardening & Self-Correction

- **Status**: Completed
- **Author**: [SeanChen327]
- **Key Changes**:
  - **View Abstraction (AD-005)**: Shifted to a Temporary View pattern (`current_working_set`), ensuring the LLM always queries a pre-filtered dataset without needing complex `WHERE` clause logic in the prompt.
  - **Self-Correction Engine**: Implemented a "Single-shot Reflection" loop; the assistant now intercepts database errors (e.g., column mismatches) and self-fixes its SQL before the user sees an error.
  - **Enhanced Sandbox Defense**: Deployed a regex-based firewall using word boundaries to block malicious DuckDB functions like `READ_CSV` or `SYSTEM` calls.
  - **Data Portability**: Added CSV export functionality for all AI-generated data summaries, improving user workflow for downstream analysis.
  - **Stability Fixes**: Resolved issues where complex multi-metric groupings caused SQL syntax errors by enforcing explicit `CASE WHEN` pivoting.

---

### [2026-05-03] Phase 5: Refinement & Polish (UX & Integration)

- **Status**: Completed
- **Author**: [SeanChen327]
- **Key Changes**:
  - **Global Filter Synchronization**: Integrated `get_active_filters_context` to allow the AI Assistant to inherit the state of the Preliminary filters tab.
  - **Professional Error Handling**: Implemented a humanized error-catching layer in `execute_read_only_sql` with descriptive English messages.
  - **Intent Routing Hardening**: Refined the classifier to explicitly map "Pivot table" keywords to `SQL_QUERY`, ensuring users receive tabular data instead of unwanted charts.
  - **Plotting Precision (Sawtooth Fix)**: Updated the SQL generation engine to force SQL-level pivoting (`CASE WHEN`) for split dimensions, ensuring clean multi-line rendering in Plotly.
  - **Aesthetic Formatting**: Automated the formatting of numeric bins as string ranges (e.g., "0-4") and enforced human-readable column aliases for professional chart legends.
  - **Security & UX**: Refined the SQL interceptor and added warnings for empty result sets to prevent user confusion.

---

### [2026-05-03] Phase 5: Refinement & Polish (UX & Integration)

- **Status**: Completed
- **Author**: [SeanChen327]
- **Key Changes**:
  - **Global Filter Synchronization**: Integrated `get_active_filters_context` to allow the AI Assistant to inherit the state of the Preliminary filters tab, ensuring query consistency.
  - **Professional Error Handling**: Implemented a humanized error-catching layer in `execute_read_only_sql` with descriptive English messages for Column Mismatches, Security Restrictions, and Payload limits.
  - **Security Hardening**: Refined the SQL interceptor to distinguish between malicious DDL/DML attempts and invalid AI-generated responses (empty/NA SQL).
  - **Empty State Feedback**: Added proactive UI warnings for successful queries that return 0 rows due to restrictive filtering.
  - **Bug Fixes**: Resolved function signature mismatches and f-string escaping issues (`{{source_table}}`) in the prompt generation engine.

---

### [2026-05-02] Phase 4: Chart Generation Copilot

- **Status**: Completed
- **Author**: [SeanChen327]
- **Key Changes**:
  - Implemented advanced chart generation using plotly.graph_objects to support dual Y-axis combo charts (Volume vs. Ratio).
  - Enhanced LLM prompt engineering to handle complex semantic mapping, allowing users to query using natural language/business terms instead of exact schema names.
  - Developed a dynamic Plotly rendering pipeline in ai_assistant.py that interprets multi-metric JSON configurations from the model.

---

### [2026-05-02] Phase 3: Text-to-SQL Core Data Pipeline

- **Status**: Completed
- **Author**: [SeanChen327]
- **Key Changes**:
  - Implemented schema-aware natural language to DuckDB SQL generation using `gemini-2.5-flash` with strict adherence to actuarial metrics logic (e.g., A/E calculation).
  - Built an execution sandbox (`execute_read_only_sql`) to enforce read-only `SELECT` statements and prevent data modification.
  - Integrated dynamic data path binding to query underlying Parquet/CSV files directly, avoiding massive Pandas memory overhead.
  - Wired the pipeline into the Streamlit chat UI, persisting generated DataFrames within `st.session_state.messages` for seamless tab switching.

---

### [2026-05-02] Phase 2: Gemini API Integration for Intent Parsing

- **Status**: Completed
- **Author**: [SeanChen327]
- **Key Changes**:
  - Migrated LLM client from the deprecated `google.generativeai` to the new `google.genai` SDK.
  - Implemented `parse_user_intent` using `gemini-2.5-flash` to accurately classify user input into SQL_QUERY, PLOT_GEN, or GENERAL_CHAT using forced JSON output.
  - UI Hardening: Resolved Streamlit layout deprecation warnings (`use_container_width` to `width="stretch"`) and fixed a multi-select state conflict in the preliminary filters tab.

---

### [2024-10-24] Phase 1: UI Skeleton & Infrastructure Setup

- **Status**: Completed
- **Author**: [SeanChen327]
- **Key Changes**:
  - Integrated "AI Assistant" tab using decoupled `ai_assistant.py` module.
  - Implemented session-based chat history storage.
  - Infrastructure Upgrade: Established GitHub Actions CI pipeline (`lint.yml`) for automated code linting.
  - Code Hardening: Resolved 17+ linting and typing issues; introduced `pyproject.toml` to standardize environment configurations.
  - Validated "Echo Mode" stability across Streamlit tab-switches.
