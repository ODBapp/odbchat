# Copilot Instructions for ODBChat Repository

## Big Picture Architecture
- **Main Components:**
  - `odbchat_mcp_server.py`: FastMCP server exposing AI tools and knowledge base resource (`odb://knowledge-base`).
  - `odbchat_cli.py`: CLI for local dev/testing; supports interactive and one-off queries.
  - `mcp_config.json`: MCP client registration (SSE endpoint).
  - `requirements.txt`: Runtime dependencies (fastmcp, ollama, mcp).
  - `change_log.md`: Track notable changes for PRs.
- **Knowledge Base:**
  - `knowledge_base/` (see README): Contains manuals, schemas, code snippets, papers, web docs, and data summaries for RAG and chat.
- **Data Flow:**
  - Server exposes tools via HTTP (default: `127.0.0.1:8045/mcp`).
  - Ollama must be running (`OLLAMA_URL` default: `127.0.0.1:11434`).
  - CLI and server communicate via MCP protocol.

## Developer Workflows
- **Environment Setup:**
  - `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- **Run Server:**
  - `python odbchat_mcp_server.py` (requires `ollama serve` running)
- **Run CLI:**
  - `python odbchat_cli.py` (interactive)
  - One-off: `python odbchat_cli.py -q "status" --status`
  - List models/tools: `python odbchat_cli.py --list-models`, `--list-tools`
- **Testing:**
  - Use `pytest` (add tests under `tests/` as `test_*.py`).
  - Mock Ollama client for tests to avoid network dependency.
- **Commits/PRs:**
  - Use clear, imperative commit subjects (e.g., `feat(server): add check_ollama_status tool`).
  - Update `change_log.md` for notable changes.

## Project-Specific Conventions
- **Python Style:** PEP 8, 4-space indent, type hints, triple-quoted docstrings.
- **Naming:**
  - `snake_case` for functions/modules
  - `PascalCase` for classes
  - CLI flags: short/long (`-m/--model`)
- **Logging:** Use `logging` (INFO default); no custom formatter.
- **Tests:** Focus on tool functions and CLI flows; aim for ≥80% coverage.

## Integration & External Dependencies
- **MCP/Agent Integration:**
  - Ensure `mcp_config.json` points to correct server URL.
  - Available tools: `chat_with_odb`, `list_available_models`, `check_ollama_status`.
  - Ollama must be running and accessible at configured URL.

## Key Patterns & Examples
- **Tool Exposure:** See `odbchat_mcp_server.py` for FastMCP tool definitions.
- **CLI Usage:** See `odbchat_cli.py` for argument parsing and tool invocation.
- **Knowledge Base Structure:** See `README.md` for directory layout and content types.
- **Test Structure:** Place tests in `tests/` as `test_*.py`.

---

For unclear or missing sections, please provide feedback to improve these instructions.
