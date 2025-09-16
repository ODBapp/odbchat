# ODBchat Spec‑Driven Development (SDD) — Refactor Plan for CLI/MCP/RAG (draft v0.3)

> Update focus: **OAS‑first RAG**, two‑part YAML ingestion, thin CLI chat + MCP one‑pass tool, and minimal ChatGPT plugin surface. Includes concrete JSON Schemas and Codex task list.

> Goal: converge `rag_cli.py` (one‑pass LLM only), `odbchat_cli.py`, and `odbchat_mcp_server.py` into a clean, spec‑first architecture that scales to the ChatGPT plugin target.

---

## 0) TL;DR

- **Keep**: one‑pass LLM path, RAG (Qdrant), CLI chat + plotting via MCP tools.
- **Move**: retrieval, planning, and ODB knowledge ops **server‑side** (MCP tools), leave CLI as thin UX shell.
- **Spec‑first**: write contracts (JSON Schema/OpenAPI), generate stubs, write tests, then implement.
- **Pluginable**: keep *patch/plugins* model (e.g., `mhw_cli_patch.py` ↔ `api/mhw_mcp.py`).
- **Endgame**: the same server/tool contracts power a ChatGPT plugin.

---

## 1) Definitions & Scope

**SDD**: We write the interface/specs first (schemas, error codes, prompts), generate code, and test against spec fixtures.

**One‑pass LLM**: No multi‑step agent; a single call does: (a) normalize user question → (b) retrieve (RAG) → (c) answer or route to a tool.

**MCP tools**: Functions exposed by `odbchat_mcp_server.py` (HTTP for dev + native MCP). Tools are atomic, stateless, and idempotent when possible.

**Plugins**: Optional modules that add MCP tools and optional CLI subcommands (e.g., MHW plotting, WOA23 table lookup).

Out‑of‑scope for this refactor: multi‑step agents (kept in a separate experimental patch), advanced orchestration.

---

## 2) High‑Level Architecture

```
CLI (odbchat_cli)  ─┬─>  MCP transport (stdio/ws/http)
                    │
                    └─>  ODBchat MCP Server
                           ├─ Core tools: RAG.search, RAG.answer_onepass, Doc.get, Health
                           ├─ ODB tools: MHW.plot_map, MHW.plot_ts, WOA23.table_query
                           ├─ Plugin slots: /plugins/* (dynamic)
                           └─ LLM adapters: (ollama/llama/gemma/gpt)

Qdrant (vector DB)  ⇄  Ingestion pipeline (ingest_mhw.py + future WOA23/API specs)

```

---

## 3) Repos & Packages (aligned to current tree)

```
repo/
  rag_cli.py                      # legacy tester (keep one-pass only; multi-steps gated for debug)
  odbchat_cli.py                  # thin CLI (REPL + one-liners); loads /mhw plugin
  odbchat_mcp_server.py           # FastMCP server (dev HTTP façade optional)

  api/
    mhw_mcp.py                    # server-side MHW MCP tools (kept)

  cli/
    plugins/
      mhw_cli_patch.py            # client-side /mhw commands; calls mhw_plot/map_plot

  plots/
    mhw_plot.py
    map_plot.py

  ingest/
    ingest.py                     # generalized multi-doc YAML → Qdrant (replaces ingest_mhw.py)

  server/
    rag/
      onepass_core.py             # extracted from rag_cli one-pass path
    tools/
      rag_onepass_tool.py         # MCP wrapper for one-pass

  rag/                            # RAG docs (.yml two-part files; OAS lives in later docs)
    odb_mhw_openapi.yml
    odb_woa23_openapi.yml
    mhw_odb_html_zh.yml

  specs/
    mcp.tools.schema.json         # tool I/O schemas (hand-written)
    prompts/
      onepass.system.txt
      onepass.user_template.txt

  tests/
    e2e_cli/
    e2e_server/
    unit_core/
    fixtures/
```

**Migration note:** keep `ingest_mhw.py` as a thin shim importing `ingest.py` for backwards compatibility, then deprecate.

---

## 4) Contracts (Spec‑First)

### 4.1 MCP Tool Catalog (v1)

- `health.ping()` → `{ ok: true, version }`
- `config.get()` → `{ llm: {provider, model}, qdrant: {host, collections:[...]}, plugins:[...] }`
- `rag.search_docs({ query, k, collection? })` → `{ hits: [{id, score, title, doc_type, chunk_id, source_file, snippet?}], usage? }`
- `rag.onepass_answer({ query, top_k?, temperature?, history?, collection? })` → `{ mode:'explain'|'code', text?, code?, plan?, citations:[...] }`
- `doc.get({ id })` → `{ id, title, content, meta }`
- **MHW**
  - `mhw.plot_map({ start, end, bbox, fields, level?, plot_field, map_method })` → `{ png_path|svg, meta }`
  - `mhw.plot_ts({ region|point, start, end, fields, level? })` → `{ png_path|svg, meta }`

> Each tool has a JSON Schema in `specs/mcp.tools.schema.json`. Inputs validate *before* execution; standardized error object on failure.

### 4.2 Error Model

```json
{
  "error": {
    "code": "BadRequest|NotFound|Timeout|Upstream|Validation",
    "message": "human readable",
    "details": { "field": "...", "hint": "..." },
    "trace_id": "uuid"
  }
}
```

### 4.3 One‑Pass Prompt Contract

- `prompts/onepass.system.txt`: guardrails, routing hints (when to call MHW tools vs pure text answer), citation format.
- `prompts/onepass.user_template.txt`: inject `{query}`, `{topK}`, `{tool_catalog}`.

---

## 5) Ingestion Specs

### 5.1 `ingest.py` (generalized, **metadata‑driven**)

- Entry: `python ingest/ingest.py --root rag --mode overwrite`（批次覆蓋）或 `--file <path>`（單檔）。
- **不強制 CLI 指定 dataset/collection/doc\_type**，以每檔 **front‑matter** 為準：
  - `dataset`（例：`marineheatwave`, `woa`）
  - `collection`（例：`mhw`, `woa23`）
  - `doc_type`（例：`api_spec`, `web_article`, `cli_tool_guide`, `data_summary`）
  - 其他：`title`, `version`, `lang`, `tags`, `source_file`
- 多段 YAML：Doc[0] 前言（可含 `content: |`）；Doc[1+] 若偵測 `openapi|swagger` **或** `doc_type=api_spec` → 視為 **OAS 並整段單塊寫入**。
- 模式：`overwrite|upsert|insert|dry-run`（`overwrite` 先清同 collection 再重建）。
- 嵌入：沿用現行模型；payload 統一含 `dataset, collection, doc_type, title, source_file, lang, version, tags`。
- Back‑compat：保留 `dev/ingest_mhw.py` shim。

### 5.2 Qdrant strategy（最小變更）

- **一個 collection 對應一個 Qdrant collection**（目前 `mhw`）。未來開多主題時，可在 server 端啟用「多 collection fan‑out + 合併 top‑K」。
- Server 設定：

```toml
[rag.collections]
active = ["mhw"]
```

- 查詢 API：`rag.search_docs({ query, k, collection?: ["mhw","woa23"] })`，預設查 `active`。

---

## 6) Server Responsibilities

- Input validation (JSON Schema) and auth (dev: optional token; prod: OAuth for plugin future).
- RAG pipeline (retriever\_qdrant): hybrid search (dense + keyword fallback), dedupe, compress‑rerank (optional).
- One‑pass reasoning (rag\_onepass.py): assemble context, apply prompt, call LLM adapter, post‑process citations.
- Tool routing: allow model to suggest a tool call; server validates and executes tool.
- Plugins: auto‑discover `api/*_mcp.py` modules that register tool specs + handlers.

---

## 7) CLI Responsibilities

- Dual modes: **REPL** (run `odbchat` with no args) and **one-liners** (`odbchat chat ...`, `odbchat search ...`).
- Free text in REPL → `rag.onepass_answer`; `/mhw ...` → client plugin; `/server|/tools|/status|/model|/help|/exit` 保留。
- Resilient transport to MCP server (stdio/ws/http) with retries and concise errors.
- Cutover: keep a tiny `rag_cli.py` shim for local tests, but **no** server‑side logic remains in it.

**CLI examples**

```
# REPL
$ odbchat
> 你好，請問 MHW 月資料與日資料定義差異？
…
> /mhw plot --start 2007-12 --bbox 150,-35,-29,35 --fields level --plot-field level --map-method basemap

# One-liners
$ odbchat chat "How to plot MHW level for Taiwan in 2015?"
$ odbchat search --k 6 "MHW definition vs ODB monthly version"
$ odbchat tool run mhw.plot_map --start 2007-12 --bbox 150,-35,-29,35 --fields level --plot-field level --map-method basemap
$ odbchat config show
```

---

## 8) Patch/Plugin Pattern (kept)

- **Client patch**: `cli/patches/mhw_cli_patch.py` adds `odbchat plot mhw-*` commands.
- **Server patch**: `server/api/mhw_mcp.py` registers `mhw.plot_*` tools with schemas.
- Load order is declarative via `specs/plugins.json`.

---

## 9) Testing Strategy

- **Contract tests**: validate sample payloads against JSON Schemas.
- **Golden answers**: fixtures for `rag.answer_onepass` with small in‑repo Qdrant sample.
- **CLI E2E**: spin a dev server (HTTP façade), run `odbchat` commands, assert outputs/files.
- **Plot regression**: checksum of generated PNG/SVG; tolerate small diffs.

---

## 10) Observability & Config

- Structured logs (JSON) with `trace_id`. Latency histograms around retrieval, LLM, tools.
- Config precedence: env → file (`config.toml`) → CLI flags. Hot‑reload for LLM model switch.

---

## 11) Security & Packaging

- Dev: token or local only. Prod/plugin: OAuth/OpenID (scoped to “ODB tools”).
- Wheel for server; pipx for CLI. Dockerfiles for both (GPU optional for local LLMs).

---

## 12) Migration Plan

1. Freeze `rag_cli.py` to **one‑pass only**.
2. Extract RAG + one‑pass logic into `server/core` and `adapters`.
3. Wire MCP tool catalog + JSON Schemas.
4. Move plotting/WOA23 features behind tools.
5. Thin `odbchat_cli` to transport + UX; keep patches.
6. Add HTTP façade + OpenAPI for dev tests.
7. E2E suite; then deprecate `rag_cli` serverish code.

---

## 13) Files to Upload (priority order)

**Core**

- `rag_cli.py` (current; one‑pass path is enough)
- `odbchat_cli.py`
- `odbchat_mcp_server.py`
- `ingest_mhw.py`

**Patches/Plugins**

- `mhw_cli_patch.py`, `mhw_plot.py`, `map_plot.py`
- `server/api/mhw_mcp.py`

**Configs & Examples**

- Any Qdrant collection schema/init script
- Sample `.yml` docs under `rag/`

**Optional**

- Current prompt templates (if any)

---

## 14) Open Questions

- Exact shape of WOA23 table spec source (CSV? MD tables? OpenAPI?).
- Minimum viable set of MCP tools for v1 plugin target.
- Where to host the dev HTTP façade (port, auth, CORS) for local tests.

---

## 15) Milestones

- **M0** (1–2 days): Upload core files; generate initial JSON Schemas + HTTP façade; stub tests.
- **M1**: Extract one‑pass to server; CLI talks to server; search/answer works.
- **M2**: MHW/WOA23 tools behind schemas; CLI patches call tools; plot regression tests.
- **M3**: Hardening: logging, config, packaging; minimal plugin scaffold for ChatGPT.

---

### Appendix A — Example Schemas (sketch)

```json
{
  "$id": "tool.mhw.plot_map.input",
  "type": "object",
  "required": ["start","end","bbox","fields","plot_field","map_method"],
  "properties": {
    "start": {"type":"string","pattern":"^\\d{4}-\\d{2}$|^\\d{4}-\\d{2}-\\d{2}$"},
    "end":   {"type":"string","pattern":"^\\d{4}-\\d{2}$|^\\d{4}-\\d{2}-\\d{2}$"},
    "bbox":  {"type":"string","pattern":"^-?\\d+(?:\\.\\d+)?,-?\\d+(?:\\.\\d+)?,-?\\d+(?:\\.\\d+)?,-?\\d+(?:\\.\\d+)?$"},
    "fields": {"type":"string"},
    "plot_field": {"type":"string"},
    "level": {"type":["integer","null"]},
    "map_method": {"enum":["basemap","cartopy"]}
  }
}
```

```json
{
  "$id": "tool.rag.answer_onepass.input",
  "type": "object",
  "required": ["query"],
  "properties": {
    "query": {"type":"string","minLength": 3},
    "top_k": {"type":"integer","minimum": 1, "maximum": 15, "default": 6},
    "cite": {"type":"boolean","default": true},
    "route_tools": {"type":"boolean","default": true}
  }
}
```

### Appendix B — Example CLI UX

```
$ odbchat chat "定義 MHW 與 ODB 月資料差異"
> [LLM] brief answer with 2–3 citations

$ odbchat tool run mhw.plot_map --start 2007-12 --end 2007-12 \
  --bbox 150,-35,-29,35 --fields level --plot-field level --map-method basemap
> Saved plot to ./out/mhw_2007-12_150_-35_-29_35.png
```



---



## 3’) One‑pass Pipeline (server extraction from `rag_cli.py`)

1. **Search**: retrieve top‑K from Qdrant.
2. **Whitelist**: harvest OAS whitelist (endpoints/params/enums/append) from retrieved OAS chunks.
3. **Decide**: LLM chooses `explain` or `code`; if `code`, produce `{plan, code}`.
4. **Validate**: `validate_plan(plan, oas)`; disallow hand‑rolled query strings; require `requests.get(..., params=...)`.
5. **Continue (optional)**: if code incomplete, single continuation with preferred chunk rule.
6. **Static guard**: enforce I/O & plotting rules (JSON→DataFrame, `pcolormesh`, datetime for `date`, etc.).
7. **Cite**: format citations.

---





---

## 16) Scaffolds — drop‑in code for Codex

### A) `ingest/ingest.py`

```python
#!/usr/bin/env python3
from __future__ import annotations
import argparse, pathlib, sys, json
from typing import List, Dict, Any, Optional
import yaml

# NOTE: This is a scaffold. Fill TODOs to wire embeddings + Qdrant.

def iter_yaml_docs(path: pathlib.Path):
    with path.open('r', encoding='utf-8') as f:
        return list(yaml.safe_load_all(f))

def is_oas(doc: Any) -> bool:
    return isinstance(doc, dict) and (('openapi' in doc) or ('swagger' in doc))

def chunk_text(text: str, maxlen: int = 1200) -> List[str]:
    # TODO: replace with semantic chunking if desired
    return [text[i:i+maxlen] for i in range(0, len(text), maxlen)]

def upsert_chunk(meta: Dict[str, Any], text: str, doc_id: str, chunk_id: int, mode: str):
    # TODO: embed + write to Qdrant with payload
    print(json.dumps({
        'action':'upsert','mode':mode,'doc_id':doc_id,'chunk_id':chunk_id,
        'collection':meta.get('collection'),'doc_type':meta.get('doc_type'),
        'title':meta.get('title')
    }, ensure_ascii=False))

def process_file(path: pathlib.Path, mode: str):
    docs = iter_yaml_docs(path)
    if not docs:
        return
    front = docs[0] if isinstance(docs[0], dict) else {}
    meta = {
        'dataset': front.get('dataset'),
        'collection': front.get('collection'),
        'doc_type': front.get('doc_type'),
        'title': front.get('title') or path.stem,
        'version': front.get('version'),
        'lang': front.get('lang'),
        'tags': front.get('tags') or [],
        'source_file': str(path),
    }
    # TODO: validate required metadata (collection/doc_type)

    # Front-matter content → short chunks
    content = front.get('content')
    if isinstance(content, str) and content.strip():
        for i, text in enumerate(chunk_text(content)):
            upsert_chunk(meta, text, doc_id=f"{path.name}:front", chunk_id=i, mode=mode)

    # Payload docs
    for idx, d in enumerate(docs[1:], start=1):
        if meta.get('doc_type') == 'api_spec' or is_oas(d):
            raw = yaml.safe_dump(d, sort_keys=False, allow_unicode=True)
            upsert_chunk({**meta, 'doc_type': 'api_spec'}, raw, doc_id=f"{path.name}:oas", chunk_id=0, mode=mode)
        else:
            raw = yaml.safe_dump(d, sort_keys=False, allow_unicode=True)
            for j, text in enumerate(chunk_text(raw)):
                upsert_chunk(meta, text, doc_id=f"{path.name}:doc{idx}", chunk_id=j, mode=mode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=pathlib.Path, help='Directory to scan for .yml')
    ap.add_argument('--file', type=pathlib.Path, help='Single .yml file to ingest')
    ap.add_argument('--mode', choices=['overwrite','upsert','insert','dry-run'], default='upsert')
    args = ap.parse_args()

    files: List[pathlib.Path] = []
    if args.file:
        files = [args.file]
    elif args.root:
        files = sorted(args.root.rglob('*.yml'))
    else:
        ap.error('Provide --root or --file')

    # TODO: if mode == overwrite → clear target collection(s) first

    for p in files:
        process_file(p, mode=args.mode)

if __name__ == '__main__':
    main()
```

### B) `server/rag/onepass_core.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Hit:
    id: str
    score: float
    title: str
    doc_type: str
    source_file: str
    chunk_id: int
    text: str

@dataclass
class Citation:
    title: str
    source: str
    chunk_id: int

@dataclass
class OnePassResult:
    mode: str  # 'explain' | 'code'
    text: Optional[str] = None
    code: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    citations: List[Citation] = None

# TODO: wire to existing retriever/utils extracted from rag_cli.py

def search_qdrant(query: str, k: int = 6, collections: Optional[List[str]] = None) -> List[Hit]:
    return []

def harvest_oas_whitelist(hits: List[Hit]) -> Dict[str, Any]:
    return {}

def decide_and_generate(query: str, notes: str, whitelist: Dict[str, Any], temperature: float = 0.2) -> OnePassResult:
    # TODO: call LLM with one-pass prompt; return either text or code+plan
    return OnePassResult(mode='explain', text='[placeholder]', citations=[])

def validate_plan(plan: Dict[str, Any], whitelist: Dict[str, Any]) -> None:
    # TODO: endpoints/params/enums validation; forbid handcrafted query strings
    pass

def enforce_static_guards(code: str) -> None:
    # TODO: JSON→DataFrame, requests.get(..., params=...), pcolormesh, datetime for date, etc.
    pass

def format_citations(hits: List[Hit]) -> List[Citation]:
    return []

def run_onepass(query: str, k: int = 6, collections: Optional[List[str]] = None, temperature: float = 0.2) -> OnePassResult:
    hits = search_qdrant(query, k=k, collections=collections)
    whitelist = harvest_oas_whitelist(hits)
    notes = ''  # TODO: compress context from hits
    result = decide_and_generate(query, notes, whitelist, temperature=temperature)
    if result.plan and result.code:
        validate_plan(result.plan, whitelist)
        enforce_static_guards(result.code)
    result.citations = format_citations(hits)
    return result
```

### C) `server/tools/rag_onepass_tool.py`

```python
from __future__ import annotations
from typing import List, Optional, Dict, Any
try:
    from fastmcp import MCP, tool
except ImportError:
    class MCP: ...
    def tool(fn=None, **kwargs):
        def deco(f): return f
        return deco

from server.rag.onepass_core import run_onepass

mcp = MCP(name='odbchat', version='0.1.0')

@tool(name='rag.onepass_answer', desc='RAG one-pass answer over ODB docs (OAS-first).')
def rag_onepass_answer(query: str, top_k: int = 6, temperature: float = 0.2, collection: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return structured one-pass answer.
    Args:
        query: user question
        top_k: number of docs to retrieve
        temperature: LLM sampling temperature
        collection: optional list of collections to search (defaults to active server collections)
    """
    res = run_onepass(query, k=top_k, collections=collection, temperature=temperature)
    return {
        "mode": res.mode,
        "text": res.text,
        "code": res.code,
        "plan": res.plan,
        "citations": [c.__dict__ for c in (res.citations or [])],
    }
```

### D) `specs/mcp.tools.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "MCP Tool Schemas",
  "version": "0.1.0",
  "tools": {
    "rag.onepass_answer.input": {
      "type": "object",
      "required": ["query"],
      "properties": {
        "query": {"type": "string", "minLength": 3},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 15, "default": 6},
        "temperature": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.2},
        "collection": {"type": "array", "items": {"type": "string"}}
      },
      "additionalProperties": false
    },
    "rag.onepass_answer.output": {
      "type": "object",
      "required": ["mode", "citations"],
      "properties": {
        "mode": {"enum": ["explain", "code"]},
        "text": {"type": "string"},
        "code": {"type": "string"},
        "plan": {"type": "object"},
        "citations": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["title", "source", "chunk_id"],
            "properties": {
              "title": {"type": "string"},
              "source": {"type": "string"},
              "chunk_id": {"type": "integer"}
            }
          }
        }
      },
      "additionalProperties": false
    },
    "rag.search_docs.input": {
      "type": "object",
      "required": ["query"],
      "properties": {
        "query": {"type": "string", "minLength": 3},
        "k": {"type": "integer", "minimum": 1, "maximum": 30, "default": 6},
        "collection": {"type": "array", "items": {"type": "string"}}
      },
      "additionalProperties": false
    },
    "rag.search_docs.output": {
      "type": "object",
      "required": ["hits"],
      "properties": {
        "hits": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["id", "score", "title", "doc_type", "chunk_id", "source_file"],
            "properties": {
              "id": {"type": "string"},
              "score": {"type": "number"},
              "title": {"type": "string"},
              "doc_type": {"type": "string"},
              "chunk_id": {"type": "integer"},
              "source_file": {"type": "string"},
              "snippet": {"type": "string"}
            }
          }
        }
      },
      "additionalProperties": false
    }
  }
}
```

