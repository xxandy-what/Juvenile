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

### [2024-10-24] Phase 1: UI Skeleton & Infrastructure Setup

- **Status**: Completed
- **Author**: [SeanChen327]
- **Key Changes**:
  - Integrated "AI Assistant" tab using decoupled `ai_assistant.py` module.
  - Implemented session-based chat history storage.
  - Infrastructure Upgrade: Established GitHub Actions CI pipeline (`lint.yml`) for automated code linting.
  - Code Hardening: Resolved 17+ linting and typing issues; introduced `pyproject.toml` to standardize environment configurations.
  - Validated "Echo Mode" stability across Streamlit tab-switches.

---

### [In Progress] Phase 2: Gemini API Integration for Intent Parsing

- **Status**: In Progress
- **Objective**: Implement Natural Language classification (SQL Query vs. Plot Generation vs. General Chat).
