from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

try:  # optional schema validation
    from jsonschema import Draft7Validator  # type: ignore
except Exception:  # pragma: no cover
    Draft7Validator = None

from server.rag.onepass_core import run_onepass

DEBUG_ENV = os.getenv("ONEPASS_DEBUG", "").strip().lower() in {"1", "true", "yes"}
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "specs" / "mcp.tools.schema.json"
_SCHEMAS: dict[str, Any] | None = None


def _load_schemas() -> dict[str, Any]:
    global _SCHEMAS
    if _SCHEMAS is None:
        if SCHEMA_PATH.exists():
            with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
                _SCHEMAS = json.load(fh)
        else:
            _SCHEMAS = {}
    return _SCHEMAS or {}


def _get_schema(name: str) -> Any:
    root = _load_schemas()
    if isinstance(root, dict) and "tools" in root and isinstance(root["tools"], dict):
        return root["tools"].get(name)
    if name.endswith(".output"):
        return root  # support schema file being the output schema itself
    return None


def _validate(name: str, instance: dict) -> None:
    schema = _get_schema(name)
    if not schema or Draft7Validator is None:
        return  # graceful no-op
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    if errors:
        snippets = []
        for err in errors:
            path = "$" + "".join(
                f"[{repr(p)}]" if isinstance(p, int) else f".{p}"
                for p in err.path
            )
            snippets.append(f"{path}: {err.message}")
        raise ValueError(f"Validation failed for {name}: " + "; ".join(snippets))


def _norm_citation(item: Any) -> dict:
    if isinstance(item, dict):
        title = item.get("title") or item.get("doc_title") or ""
        source = item.get("source") or item.get("source_file") or item.get("file") or ""
        chunk_id = item.get("chunk_id", 0)
    else:
        title = getattr(item, "title", "") or getattr(item, "doc_title", "") or ""
        source = getattr(item, "source", "") or getattr(item, "source_file", "") or getattr(item, "file", "") or ""
        chunk_id = getattr(item, "chunk_id", 0)
    try:
        chunk_id = int(chunk_id)
    except Exception:
        chunk_id = 0
    return {"title": str(title), "source": str(source), "chunk_id": chunk_id}


def _norm_plan(plan: Any) -> dict | None:
    if plan is None:
        return None
    if isinstance(plan, dict):
        return plan
    if isinstance(plan, str):
        try:
            obj = json.loads(plan)
            return obj if isinstance(obj, dict) else {"_raw": plan}
        except Exception:
            return {"_raw": plan}
    try:
        return {"_raw": json.dumps(plan, ensure_ascii=False)}
    except Exception:
        return {"_raw": str(plan)}


def _norm_mode(mode: Any, code: Any, text: Any) -> str:
    m = (mode or "").strip().lower()
    if m in {"code", "explain", "fallback", "mcp_tools"}:
        return m
    code_has = bool(code and str(code).strip())
    return "code" if code_has else "explain"


def register_rag_onepass_tool(mcp: FastMCP) -> None:
    @mcp.tool("rag.onepass_answer")
    def rag_onepass_answer(
        query: str,
        top_k: int = 6,
        temperature: float = 0.2,
        collection: list[str] | None = None,
        debug: bool = False,
    ) -> dict:
        """Execute the one-pass RAG pipeline and return structured output."""
        debug_flag = bool(debug)

        payload_in: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "temperature": temperature,
        }
        if collection is not None:
            payload_in["collection"] = collection
        if debug_flag:
            payload_in["debug"] = True
        _validate("rag.onepass_answer.input", payload_in)

        result = run_onepass(
            query,
            k=top_k,
            collections=collection,
            temperature=temperature,
            debug=debug_flag,
        )

        res_mode = getattr(result, "mode", None)
        res_text = getattr(result, "text", None)
        res_code = getattr(result, "code", None)
        res_plan = getattr(result, "plan", None)
        res_citations = getattr(result, "citations", None) or []
        res_debug = getattr(result, "debug", None)
        res_mcp = getattr(result, "mcp", None)

        payload_out: dict[str, Any] = {
            "mode": _norm_mode(res_mode, res_code, res_text),
            "citations": [_norm_citation(c) for c in res_citations],
        }
        if res_text and str(res_text).strip():
            payload_out["text"] = str(res_text)
        if res_code and str(res_code).strip():
            payload_out["code"] = str(res_code)
        plan_obj = _norm_plan(res_plan)
        if plan_obj:
            payload_out["plan"] = plan_obj
        if payload_out["mode"] == "mcp_tools" and isinstance(res_mcp, dict):
            payload_out["mcp"] = res_mcp

        debug_enabled = debug_flag or DEBUG_ENV
        if debug_enabled and isinstance(res_debug, dict):
            dbg = dict(res_debug)
            dbg.setdefault("mode", payload_out["mode"])
            dbg.setdefault("code_len", len(res_code or ""))
            if plan_obj:
                dbg["plan"] = plan_obj
            elif dbg.get("plan") is None:
                dbg.pop("plan", None)
            wl = dbg.get("whitelist") or {}
            dbg["whitelist"] = {
                "paths_count": int(wl.get("paths_count", 0)),
                "params_count": int(wl.get("params_count", 0)),
                "append_allowed": list(wl.get("append_allowed", [])),
                "sample_paths": list(wl.get("sample_paths", [])),
                "sample_params": list(wl.get("sample_params", [])),
            }
            dbg["continued"] = bool(dbg.get("continued", False))
            durations = dbg.get("durations_ms") or {}
            dbg["durations_ms"] = {
                "search": int(durations.get("search", 0)),
                "whitelist": int(durations.get("whitelist", 0)),
                "llm": int(durations.get("llm", 0)),
                "validate_guard": int(durations.get("validate_guard") or durations.get("validate+guard", 0)),
                "total": int(durations.get("total", 0)),
            }
            payload_out["debug"] = dbg

        _validate("rag.onepass_answer.output", payload_out)
        return payload_out
