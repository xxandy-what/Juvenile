# Project Governance & Change Log

This document tracks major architectural decisions, version changes, and development progress to ensure transparency and traceability in team collaboration.

## 1. Versioning & Commit Standards

- **Feature Branches**: Use `feature/phaseX-description` naming convention.
- **Commit Messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat:`, `fix:`, `docs:`, `refactor:`).

## 2. Core Architecture Decisions (ADR)

- **[AD-001] AI Assistant Modularization**: To maintain the maintainability of `appv2.py`, all AI-related rendering, API calls, and SQL generation logic must be encapsulated in the `ai_assistant.py` module.
- **[AD-002] State Persistence**: Conversation history must be stored in `st.session_state.messages` to prevent data loss during Streamlit reruns or tab switching.

## 3. Change Log

### [2024-10-24] Phase 1: UI Skeleton & Echo System Validation

- **Status**: Completed (Merged via PR #1)
- **Author**: [SeanChen327]
- **Key Changes**:
  - Integrated "AI Assistant" tab into `appv2.py`.
  - Implemented logic decoupling via the `ai_assistant.py` module.
  - Established session-based chat history storage.
  - Validated "Echo Mode" to confirm UI stability and state retention.
- **Open Issues**: None.

---

### [In Progress] Phase 2: Gemini API Integration for Intent Parsing

- **Status**: In Progress
- **Objective**: Implement Natural Language classification (SQL Query vs. Plot Generation vs. General Chat).
