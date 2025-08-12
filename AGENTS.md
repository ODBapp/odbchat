# Repository Guidelines

## Project Structure & Module Organization
- `odbchat_mcp_server.py`: FastMCP server exposing tools (`chat_with_odb`, `list_available_models`, `check_ollama_status`) and resource `odb://knowledge-base`.
- `odbchat_cli.py`: CLI client for local development and manual testing.
- `mcp_config.json`: Example MCP client registration (SSE at `http://127.0.0.1:8045/mcp`).
- `requirements.txt`: Runtime deps (fastmcp, ollama, mcp). Consider a virtualenv.
- `change_log.md`: Update with notable changes in PRs.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Run server: `python odbchat_mcp_server.py` (HTTP transport on `127.0.0.1:8045/mcp`; requires `ollama serve`).
- Run CLI (interactive): `python odbchat_cli.py`.
- One-off query: `python odbchat_cli.py -q "status" --status` or list models: `python odbchat_cli.py --list-models`.
- Discover tools: `python odbchat_cli.py --list-tools`.

## Coding Style & Naming Conventions
- Python, PEP 8, 4-space indentation, keep type hints and docstrings (triple-quoted) as in current files.
- Names: `snake_case` for modules/functions, `PascalCase` for classes, short/long CLI flags (`-m/--model`).
- Logging via `logging` (INFO default). No formatter configured; prefer Black/Ruff locally if available.

## Testing Guidelines
- Framework: pytest (not yet present). Add tests under `tests/` as `test_*.py`.
- Focus: tool functions in `odbchat_mcp_server.py` and CLI flows. Mock Ollama client to avoid network.
- Example: `pytest -q` (target ≥80% coverage when feasible).

## Commit & Pull Request Guidelines
- Commit style: current history is minimal; use clear, imperative subjects or Conventional Commits.
  Example: `feat(server): add check_ollama_status tool`.
- PRs: describe change, link issues, include how to run/test (server + CLI steps), and update `change_log.md`.
- Keep diffs focused; include screenshots/terminal snippets for CLI behavior when helpful.

## MCP/Agent Integration Tips
- Ensure `mcp_config.json` points to your server URL.
- Available tools: `chat_with_odb`, `list_available_models`, `check_ollama_status`.
- Server uses `OLLAMA_URL` constant (127.0.0.1:11434). Update it if your Ollama host differs.
