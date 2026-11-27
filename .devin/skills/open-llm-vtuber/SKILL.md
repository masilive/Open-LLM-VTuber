---
name: open-llm-vtuber
description: Work safely on Open-LLM-VTuber backend, realtime audio, WebSocket, configuration, and engine integrations in this repository.
triggers:
  - user
  - model
---

# Open-LLM-VTuber workflow

Use this skill for non-trivial changes in this repository. `CLAUDE.md` and `.cursor/rules/olv-core-rules.mdc` are the authoritative project rules; this skill is an execution checklist, not a replacement for them.

## Establish context before editing

1. Inspect the neighboring implementation, interface, factory, and configuration model for the area being changed.
2. Identify whether the change crosses any of these boundaries:
   - WebSocket message handling or client/server protocol
   - ASR, TTS, VAD, LLM, agent, or MCP engine integration
   - YAML configuration schema or character configuration
   - realtime audio streaming, cancellation, or latency-sensitive async code
   - optional/platform-specific dependencies
3. Preserve existing architecture and naming conventions. Do not introduce a new framework when an existing project pattern solves the problem.

## Non-negotiable constraints

- Use `uv` for dependency and Python command management. Do not use `pip` for project dependency management.
- Target Python `>=3.10,<3.13` and modern type-hint syntax (`str | None`, `list[int]`).
- Keep core functionality offline-ready. Network-only capability must be optional and fail clearly.
- Do not block the event loop in async code. Use async-native APIs or appropriate offloading for CPU/blocking work.
- Keep the frontend/backend separation. Do not edit the compiled `frontend/` submodule from this backend repository.
- Treat `conf.yaml`, `mcp_servers.json`, credentials, `cache/`, `logs/`, `models/`, and `chat_history/` as local runtime state.
- Keep `.devin/blueprint.yaml` synchronized when setup, dependency, validation, or startup commands change.
- Use `loguru` for logging. Public modules, classes, functions, and methods need English Google-style docstrings when added or materially changed.

## Change-specific checklists

### Configuration schema changes

When adding, renaming, removing, or restructuring configuration:

1. Update the Pydantic models in `src/open_llm_vtuber/config_manager/`.
2. Update both `config_templates/conf.default.yaml` and `config_templates/conf.ZH.default.yaml`.
3. Check `characters/` for affected alternate configurations.
4. Consider backward compatibility and the upgrade system in `upgrade_codes/` for breaking changes.
5. Validate the resulting configuration with:
   `uv run python -c "from src.open_llm_vtuber.config_manager import read_yaml, validate_config; validate_config(read_yaml('conf.yaml')); print('conf.yaml valid')"`

### Engine or provider integrations

When adding or changing an ASR, TTS, VAD, LLM, agent, or MCP provider:

1. Read the corresponding interface and factory first.
2. Follow an existing concrete implementation with similar lifecycle and streaming behavior.
3. Register the implementation in the appropriate factory and configuration models.
4. Update both configuration templates and relevant documentation/comments already present in those templates.
5. Keep optional dependencies optional: import lazily where appropriate, degrade gracefully, and do not prevent core startup when the dependency or hardware is unavailable.
6. Prefer streaming/incremental processing over buffering when latency matters.

### WebSocket, conversation, or audio pipeline changes

1. Add or update the message type and handler registration consistently.
2. Trace single-user and group-conversation effects before editing.
3. Handle disconnects, cancellation, interrupts, and cleanup paths explicitly.
4. Avoid unbounded queues or memory growth for streaming audio.
5. Verify behavior around interruption and TTS playback, not only message acceptance.

## Verification workflow

For Python changes, run these from the repository root:

```sh
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall -q src run_server.py upgrade.py
```

If formatting fails only in files you did not touch, report that separately instead of reformatting unrelated code. Format files you touched with `uv run ruff format <paths>`.

There is currently no automated test suite in this repository. Do not claim that tests passed. Use focused smoke checks appropriate to the change, such as:

- configuration validation for config schema changes
- direct factory/interface instantiation where practical
- compileall for syntax/import-time issues
- targeted manual or scripted smoke tests for WebSocket behavior

Runtime smoke tests require additional prerequisites:

- `ffmpeg -version` must succeed
- `frontend/index.html` must exist (`git submodule update --init --recursive` if missing)
- `conf.yaml` must exist and validate
- external/local providers referenced by `conf.yaml` must be reachable

The server is long-running: start `uv run run_server.py --verbose` only when runtime verification is needed, observe startup, then stop it cleanly. Startup may download ASR or other model files and may fail if configured providers such as Ollama are unavailable.

## Completion standard

Before finishing, report:

- which verification commands passed or failed
- whether failures were pre-existing or caused by the current change
- any runtime behavior not verified because models, FFmpeg, API keys, or external providers were unavailable
