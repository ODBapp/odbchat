from __future__ import annotations

import json
from pathlib import Path

from fastmcp import FastMCP
from jsonschema import Draft7Validator

from server.rag.onepass_core import run_onepass

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "specs" / "mcp.tools.schema.json"
_SCHEMAS = {}


def _load_schemas():
    global _SCHEMAS
    if _SCHEMAS:
        return _SCHEMAS
    if SCHEMA_PATH.exists():
        data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _SCHEMAS = data.get("tools", {}) or {}
    return _SCHEMAS


def _validate(schema_name: str, instance: dict) -> None:
    schema = _load_schemas().get(schema_name)
    if not schema:
        return
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    if errors:
        messages = ", ".join(error.message for error in errors)
        raise ValueError(f"Validation failed for {schema_name}: {messages}")


def register_rag_onepass_tool(mcp: FastMCP) -> None:
    @mcp.tool("rag.onepass_answer")
    def rag_onepass_answer(
        query: str,
        top_k: int = 6,
        temperature: float = 0.2,
        collection=None,
    ):
        """RAG one-pass answer over ODB docs (OAS-first)."""
        payload_in = {
            "query": query,
            "top_k": top_k,
            "temperature": temperature,
        }
        if collection is not None:
            payload_in["collection"] = collection
        _validate("rag.onepass_answer.input", payload_in)

        result = run_onepass(query, k=top_k, collections=collection, temperature=temperature)
        payload_out = {
            "mode": result.mode,
            "citations": [c.__dict__ for c in (result.citations or [])],
        }
        if result.text:
            payload_out["text"] = result.text
        if result.code:
            payload_out["code"] = result.code
        if result.plan:
            payload_out["plan"] = result.plan

        _validate("rag.onepass_answer.output", payload_out)
        return payload_out
