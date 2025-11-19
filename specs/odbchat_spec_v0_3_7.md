# ODBchat Platform Spec (v0.3.7)

**Date:** 2025-11-14  
**Owners:** ODBchat server + CLI maintainers  
**Status:** Living spec that reflects the current codebase (router-first MCP pipeline + metocean integration). This supersedes `odbchat_spec_v0_3.md`.

---

## 0. TL;DR

- **Router-first:** All free-text queries hit `router.answer`. The router runs one LLM pass that classifies into `mcp_tools | explain | code | fallback`, infers tool arguments, and can override the LLM decision (e.g., force GHRSST/tide even when the classifier said `code`).
- **Metocean suite:** GHRSST and tide tools now share the upstream `https://eco.odb.ntu.edu.tw/mcp/metocean` endpoint with UA `metocean-mcp`. `tide.forecast` feeds tide + sun/moon answers in zh-TW or the user’s language, with translation tables and inland/no-data handling.
- **CLI UX:** `odbchat_cli.py` buffers multi-line paste before sending, always passes `tz` + `query_time`, exposes slash commands, and falls back to `rag.onepass_answer` only when the router returns explain/code.
- **Specs & schemas:** JSON Schemas live in `specs/mcp.tools.schema.json`. This spec and `odbchat_mcp_tools_metocean_integration_spec_v0_3_7.md` define router behavior, tide rendering, and MCP contracts.

---

## 1. Components & Responsibilities

| Layer | Responsibilities |
| --- | --- |
| CLI (`odbchat_cli.py`) | Chat REPL + one-liners, multi-line paste buffer, slash commands, attaches `tz/query_time`, renders MCP results (SST/tide) and citations, surfaces router debug when `--debug`. |
| MCP server (`odbchat_mcp_server.py`) | Registers all tools, wires proxies, exposes router + one-pass + config tools, primes embeddings/Qdrant on boot. |
| Router (`server/router_classifier.py`) | Single entry for free-text. Runs `run_onepass` to get LLM mode, but also parses queries locally to override decisions. Executes GHRSST/tide MCP tools, renders answers, or returns the one-pass payload. |
| RAG/one-pass (`server/rag/onepass_core.py`) | Retrieval (Qdrant), whitelist extraction, prompt assembly, and guardrails for explain/code/mcp plans. |
| MCP proxies | `server/api/ghrsst_mcp_proxy.py` & `server/api/metocean_mcp_proxy.py` forward calls upstream with UA headers and normalized responses. |
| Specs/docs | `specs/*.md`, `specs/mcp.tools.schema.json`, and `rag/manuals/*.yml` describe contracts + CLI docs. |

---

## 2. MCP Tool Catalog

| Tool | Purpose | Notes |
| --- | --- | --- |
| `router.answer` | Primary chat surface. Input `{query, tz?, query_time?, debug?}`. Output includes `mode`, `text/code`, `tool/arguments/result` (when MCP), `citations`, optional `debug`. |
| `rag.onepass_answer` | Direct access to the one-pass RAG pipeline (`mode` may be `explain | code | fallback | mcp_tools`). Supports `collection`, `top_k`, `temperature`, `debug`. |
| `config.get_model`, `config.set_model`, `config.set_provider`, `config.llm_status` | Manage/query LLM provider/model state. |
| `ghrsst.point_value`, `ghrsst.bbox_mean` | Daily SST/SSTA from GHRSST via metocean MCP endpoint. Accepts `date`, `fields`, `method` (`exact|nearest`). |
| `tide.forecast` | Tides + sun/moon for a point or station. Inputs: `{longitude?, latitude?, station_id?, date?, query_time?, tz?}`. No `include_sun_moon` flag; sun/moon always returned when available. |
| (Legacy) `mhw.plot_*` | Still registered; CLI plugin calls them when `/mhw …` commands are used. |

Each tool’s I/O is defined under `specs/mcp.tools.schema.json`. Schemas must be Draft-07 compatible because validation is enforced in `server/tools/rag_onepass_tool.py`.

---

## 3. Updated JSON Schema Expectations

- `specs/mcp.tools.schema.json` now tracks **version `0.3.7`**.
- `rag.onepass_answer.output.mode` enumerates `["explain","code","fallback","mcp_tools"]`.
- When the LLM emits an MCP plan, the output includes an `mcp` object (`{ "tool": str, "arguments": {...}, "raw_plan"?: {...} }`). Downstream code treats it as opaque JSON; schema should allow nested objects.
- `router.answer.output` requires `mode` + `citations` and allows one of:
  - MCP result: `tool`, `arguments`, `result`, `text`, optional `error`, `confidence`, `plan`, `debug`.
  - Explain/code fallback: `text?`, `code?`, `plan?`, `citations`, `debug?`.
- `router.answer.input` requires `query` and accepts optional `tz`, `query_time`, `debug`.

See the updated schema file in this repo for authoritative structures.

---

## 4. Router & One-Pass Behavior

1. **Single LLM call:** Router invokes `run_onepass` with today/tz/query_time. The LLM decides mode and may include an `mcp` block.
2. **Local overrides:** Even if the LLM says `code`/`explain`, the router checks heuristics:
   - `_looks_like_tide_query` (regex covers tide keywords + sun/moon) + `_contains_code_request` gate.
   - `_looks_like_ghrsst_query` (regex + bbox/point detection + time hints).
   - `_is_sea_state_only_query`: replies with “目前僅提供海表溫度與潮汐資訊…” instead of code.
3. **Tool execution:**
   - GHRSST point vs bbox detection is deterministic based on parsed coordinates.
   - `method` defaults to `"exact"` but switches to `nearest` if user mentioned “今天/now/today/current” **and** requested today’s date; fallback also occurs on upstream “data not exist” errors.
   - Tide normalization ensures `tz`/`query_time`/`date` exist, coerces lon/lat order, and rejects missing coordinates.
4. **Language handling:** `_is_zh_query` toggles translation tables. Non-zh queries get English phrasing; zh queries use the translation map in §6.
5. **Outputs:** Router builds a normalized dict with `mode`, `text` (or `code`), `citations`, plus `tool/arguments/result` when MCP. Errors from proxies are mapped to `mode="mcp_tools"` with `error` strings; text always starts with “目前無法取得資料：…”.

---

## 5. Metocean & GHRSST Rules (High Level)

See `specs/odbchat_mcp_tools_metocean_integration_spec_v0_3_7.md` for the full contract. Highlights:

- Both GHRSST tools and `tide.forecast` hit `https://eco.odb.ntu.edu.tw/mcp/metocean` with UA `metocean-mcp`.
- CLI always supplies `tz` (IANA name, default `Asia/Taipei`) and `query_time` (ISO-8601) in router requests; the server reuses them when populating tide arguments and can synthesize defaults if missing.
- Tide rendering:
  - **Relative mode** (query date == today): state label (“目前屬於漲潮 / Currently rising tide”), next/previous extremes with `(~ 6小時31分 後)` or `(about 4h32m later)`, sun/moon segments (`曙光 05:45 …` / `Civil dawn 05:45 …`), and the tide list appended on a new line (`潮位資訊：滿潮 ...` / `Tide list: …`). Notes go at the end.
  - **Summary mode** (future/past date): omit state/relative text; list extremes with “資料日期 YYYY-MM-DD” (zh) / “Date YYYY-MM-DD” (en).
  - **Sun/Moon-only questions:** still call `tide.forecast` but hide tide/state unless keywords like `潮/滿潮/tide` appear. The note about NTDE/MSL is skipped when no tide data is shown.
  - **No tide data (inland points):** when the user explicitly asked for tide, append `此點位並無潮汐資料。` / `No tide data is available for this location.` after the rendered text.
  - **Translations:** Moon phase map (e.g., `Waning Crescent → 殘月`), `Fraction of the Moon Illuminated → 月盈`, `Tide heights → 潮高`, `MSL → 平均海水面`.
  - **Sea-state queries:** respond with “目前僅提供海表溫度與潮汐資訊…” to avoid hallucinated code.
- GHRSST responses keep the formatted string `YYYY-MM-DD｜point/bbox […] SST ≈ …, Anomaly …（原請求 YYYY-MM-DD）`.

---

## 6. CLI Expectations

1. **Transport:** CLI uses MCP (stdio for local, HTTP dev optional). Free-text always calls `router.answer`.
2. **Input handling:** Multi-line paste is buffered until the user presses Enter on an empty line; CLI then sends the entire block as one prompt.
3. **Context fields:** Every router call includes `tz` (resolved via `tzlocal` fallback to `Asia/Taipei`) and `query_time` (ISO timestamp in that tz). `/help` documents these semantics for debugging.
4. **Rendering:**
   - SST replies: print `result["text"]`, then optional debug/citations.
   - Tide replies: keep the multi-line format from the server (sun/moon on same line, tide list on separate line, note last). Do not re-translate.
   - Explain/code replies: show `text` then syntax-highlighted code; cite docs underneath unless the server gave zero citations.
5. **Slash commands:** `/server connect`, `/llm status`, `/model set`, `/mhw …` (plugin), `/history`, `/help`, `/exit`.
6. **Error surfaces:** Router errors are shown verbatim; CLI adds context when MCP transport fails (e.g., SSE 400).

---

## 7. RAG Docs & Ingestion

- `ingest/ingest.py --root rag --mode overwrite` is the canonical ingest path. Each YAML doc must include front-matter: `dataset`, `collection`, `doc_type`, `title`, `lang`, `tags`. The ingest script writes payload metadata into Qdrant.
- Docs are chunked with semantic-friendly heuristics (current implementation is simple slicing; improving this is on the roadmap).
- CLI/manual guides have been split:
  - `rag/manuals/odbchat_cli_manual.yml`
  - `rag/manuals/odbchat_cli_quickstart.yml`
  Each emphasises how to run `/mhw` commands and clarifies supported flags (e.g., `--bbox` not per-parameter lon/lat).
- `ingest` metadata controls `collection` fan-out. Server defaults to `active=["mhw"]` but the prompt includes the whitelist extracted from whichever OAS docs matched the query.

---

## 8. Testing & Observability

- **Unit tests:** `tests/test_router_classifier.py` covers:
  - English/zh tide queries (with/without tide data, future dates, N/E coordinate notation, sun-only questions).
  - GHRSST routing heuristics.
  - Sea-state fallback and CLI-specific keyword handling.
  - Language-specific rendering (duration format `4h32m` vs `4小時32分`).
- **CLI tests:** `tests/test_cli_onepass.py` ensures router piping, multi-line input, tide/GHRSST debug surfaces, and `/server` flows.
- **Logging:** Router logs `mode`, `needs_tide`, and upstream errors. Proxy modules log remote failures with tool names and sanitized args. Tide answers include `state_now`, `next/prev`, `tz`, `query_time` in debug.
- **Schema validation:** `rag.onepass_answer` validates inputs/outputs at runtime when `jsonschema` is available. Router schemas are enforced via CI by running `jsonschema` CLI in `ingest` tests (TODO automation).

---

## 9. Migration Checklist for Future Work

1. Keep `specs/*` aligned with behavior after every release bump (v0.3.7 is the current baseline).
2. When adding new metocean tools (e.g., `tide.harmonics`, `swell.forecast`), update both this spec and `mcp.tools.schema.json` before coding.
3. Before expanding CLI commands, ensure the RAG docs (`rag/manuals/*.yml`) mention the canonical syntax so the LLM can cite them.
4. For “sea state” or SADCP future work, extend the router heuristics via prompt/context—not by piling keyword lists—unless there is a deterministic parser.
5. Continue capturing representative transcripts (zh + en) each time the router heuristics change; add new pytest cases alongside spec updates.

---

This document, together with `specs/mcp.tools.schema.json` and `specs/odbchat_mcp_tools_metocean_integration_spec_v0_3_7.md`, forms the authoritative contract for ODBchat v0.3.7.
