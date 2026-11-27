# Open-LLM-VTuber Agent Guide

`CLAUDE.md` and `.cursor/rules/olv-core-rules.mdc` contain the detailed project rules and remain authoritative. Keep this file concise.

## Required workflow

- Use `uv` for Python environment and dependency operations: `uv sync`, `uv run`, `uv add`, and `uv remove`. Do not use `pip` for project dependency management.
- For non-trivial backend, audio, WebSocket, configuration, or engine-integration work, invoke the project `open-llm-vtuber` skill before editing.
- Before finishing Python changes, run:
  - `uv run ruff check .`
  - `uv run ruff format --check .`
  - `uv run python -m compileall -q src run_server.py upgrade.py`
- There is currently no automated test suite in this repository. Use targeted smoke checks and state any remaining manual verification gap honestly.

## Project constraints

- Preserve the offline-ready core, strict frontend/backend separation, low-latency behavior, async/non-blocking implementation, and cross-platform support.
- `frontend/` is a git submodule containing compiled frontend artifacts. Do not edit it directly in this repository.
- `conf.yaml`, `mcp_servers.json`, `cache/`, `logs/`, `models/`, `chat_history/`, and local credentials are runtime state, not source code. Do not commit them.
- FFmpeg is required for audio runtime behavior; verify `ffmpeg -version` before runtime smoke tests.
- When configuration structure changes, update both files in `config_templates/` and the Pydantic models under `src/open_llm_vtuber/config_manager/`.
- `.devin/blueprint.yaml` is the git-backed Devin Cloud environment blueprint. Keep its setup, maintenance, and knowledge commands synchronized with project workflow changes.
