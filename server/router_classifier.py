"""Lightweight classifier/router for deciding between MCP tools and RAG."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from server.llm_adapter import LLM
from server.api.ghrsst_mcp_proxy import _ghrsst_bbox_mean, _ghrsst_point_value
from server.rag.onepass_core import run_onepass
from server.tools.rag_onepass_tool import (
    _norm_citation,
    _norm_mode,
    _norm_plan,
)

logger = logging.getLogger("odbchat.router")

DEFAULT_FIELDS = ["sst", "sst_anomaly"]
TPE_TZ = ZoneInfo("Asia/Taipei") if ZoneInfo else None
NEAREST_KEYWORDS = (
    "今天",
    "今日",
    "現在",
    "目前",
    "最新",
    "nearest",
    "最近",
    "current",
)

SYSTEM_PROMPT = (
    "You are an intent classifier for ocean data questions. "
    "Every user turn is a JSON object with keys 'today' (YYYY-MM-DD) and 'query'. "
    "Reply with a single JSON object matching this schema:\n"
    "{\n"
    "  \"decision\": \"mcp_tools|explain|code\",\n"
    "  \"confidence\": number (0-1, optional),\n"
    "  \"tool\": {\n"
    "    \"name\": \"ghrsst.point_value|ghrsst.bbox_mean\",\n"
    "    \"arguments\": object\n"
    "  } (required when decision is mcp_tools),\n"
    "  \"rationale\": string (optional)\n"
    "}.\n"
    "Rules: respond with JSON only; no markdown or commentary. "
    "If the query asks for numerical SST at a specific lon/lat use point_value. "
    "If it asks for SST over a latitude/longitude range use bbox_mean. "
    "Use the provided 'today' value when a date is needed. "
    "Return decision='explain' for conceptual questions; 'code' for programming tasks."
)

FEWSHOT_MESSAGES = [
    {
        "role": "user",
        "content": json.dumps(
            {
                "today": "2025-01-05",
                "query": "今天台灣外海(23N, 123E)海溫多少？",
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "decision": "mcp_tools",
                "tool": {
                    "name": "ghrsst.point_value",
                    "arguments": {
                        "longitude": 123.0,
                        "latitude": 23.0,
                        "date": "2025-01-05",
                        "fields": DEFAULT_FIELDS,
                        "method": "nearest",
                    },
                },
                "confidence": 0.82,
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": json.dumps(
            {
                "today": "2025-01-05",
                "query": "今天台灣外海經緯度範圍[123, 21, 124, 22] 海溫多少",
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "decision": "mcp_tools",
                "tool": {
                    "name": "ghrsst.bbox_mean",
                    "arguments": {
                        "bbox": [123.0, 21.0, 124.0, 22.0],
                        "date": "2025-01-05",
                        "fields": DEFAULT_FIELDS,
                        "method": "nearest",
                    },
                },
                "confidence": 0.78,
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": json.dumps(
            {
                "today": "2025-01-05",
                "query": "GHRSST 是什麼？",
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "decision": "explain",
                "confidence": 0.71,
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": json.dumps(
            {
                "today": "2025-01-05",
                "query": "幫我寫 Python 把 JSON 轉 CSV",
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "decision": "code",
                "confidence": 0.76,
            },
            ensure_ascii=False,
        ),
    },
]


@dataclass
class MCPDecision:
    decision: str
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] | None = None
    confidence: float | None = None
    raw_plan: Dict[str, Any] | None = None


def _today_tpe() -> str:
    tz = TPE_TZ
    now = datetime.now(tz) if tz else datetime.utcnow()
    return now.strftime("%Y-%m-%d")


def _build_messages(query: str, today: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    messages.extend(FEWSHOT_MESSAGES)
    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                {"today": today, "query": query}, ensure_ascii=False
            ),
        }
    )
    return messages


def _extract_json_obj(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            logger.debug("Failed to parse JSON from snippet: %s", snippet)
    return {}


def _classify_sync(query: str, today: str) -> MCPDecision:
    messages = _build_messages(query, today)
    raw = LLM.chat(messages, temperature=0.0, max_tokens=256)
    plan = _extract_json_obj(raw)

    decision = str(plan.get("decision", "explain")).strip().lower()
    if decision not in {"mcp_tools", "explain", "code"}:
        decision = "explain"

    tool_name = None
    arguments: Dict[str, Any] | None = None
    if decision == "mcp_tools":
        tool = plan.get("tool") or {}
        tool_name = str(tool.get("name") or "").strip()
        if not tool_name:
            decision = "explain"
        else:
            arguments = tool.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {}

    confidence = plan.get("confidence")
    try:
        if confidence is not None:
            confidence = float(confidence)
    except Exception:
        confidence = None

    return MCPDecision(
        decision=decision,
        tool_name=tool_name,
        arguments=arguments,
        confidence=confidence,
        raw_plan=plan if isinstance(plan, dict) else None,
    )


def _ensure_date(value: Any, fallback: str) -> str:
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        return value
    return fallback


def _ensure_fields(value: Any) -> List[str]:
    if isinstance(value, list):
        fields: List[str] = []
        for item in value:
            if isinstance(item, str) and item in {"sst", "sst_anomaly"}:
                fields.append(item)
        if fields:
            return list(dict.fromkeys(fields))
    return DEFAULT_FIELDS.copy()


def _ensure_bbox(value: Any) -> List[float]:
    if isinstance(value, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
        value = [float(num) for num in numbers]
    if isinstance(value, list) and len(value) == 4:
        try:
            floats = [float(x) for x in value]
            return floats
        except Exception as exc:
            raise ToolError(f"Invalid bbox values: {value}") from exc
    raise ToolError("bbox must contain four numbers [lon0, lat0, lon1, lat1]")


def _ensure_coordinates(arguments: Dict[str, Any]) -> Tuple[float, float]:
    try:
        lon = float(arguments.get("longitude"))
        lat = float(arguments.get("latitude"))
    except Exception as exc:
        raise ToolError("longitude and latitude must be numeric") from exc
    if math.isnan(lon) or math.isnan(lat):
        raise ToolError("longitude and latitude cannot be NaN")
    return lon, lat


def _normalize_method(value: Any) -> tuple[str, bool]:
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in {"exact", "nearest"}:
            return candidate, True
    return "exact", False


def _should_force_nearest(
    explicit: bool,
    method: str,
    requested_date: str,
    today: str,
    query: str,
) -> bool:
    if method == "nearest" or explicit:
        return False
    if requested_date == today:
        return True
    lowered = query.lower()
    return any(keyword in query or keyword in lowered for keyword in NEAREST_KEYWORDS)


def _is_missing_data_error(message: str) -> bool:
    lower = message.lower()
    return (
        "data not exist" in lower
        or "no data" in lower
        or "unavailable" in lower
        or "available range" in lower
    )


def _clean_tool_error_message(message: str) -> str:
    cleaned = message.strip()
    prefix = "Error calling tool"
    if prefix in cleaned:
        idx = cleaned.find(":")
        if idx != -1:
            cleaned = cleaned[idx + 1 :].strip()
    return cleaned


def _build_mcp_error_response(
    tool: str,
    normalized_args: Dict[str, Any],
    decision: MCPDecision,
    error_message: str,
    debug: bool,
) -> Dict[str, Any]:
    user_message = _clean_tool_error_message(error_message)
    text = f"目前無法取得資料：{user_message}"
    response: Dict[str, Any] = {
        "mode": "mcp_tools",
        "tool": tool,
        "arguments": normalized_args,
        "error": user_message,
        "text": text,
        "citations": [],
    }
    if decision.confidence is not None:
        response["confidence"] = decision.confidence
    if decision.raw_plan:
        response["plan"] = decision.raw_plan
    if debug:
        response["debug"] = {
            "plan": decision.raw_plan,
            "arguments": normalized_args,
            "error": user_message,
        }
    return response


def _render_tool_text(tool: str, result: Dict[str, Any]) -> str:
    date = result.get("date") or "(unknown date)"
    sst = result.get("sst")
    anomaly = result.get("sst_anomaly")
    parts: List[str] = []
    if isinstance(sst, (int, float)):
        parts.append(f"SST ≈ {sst:.2f} °C")
    if isinstance(anomaly, (int, float)):
        sign = "+" if anomaly >= 0 else ""
        parts.append(f"Anomaly {sign}{anomaly:.2f} °C")
    payload = ", ".join(parts) if parts else "No SST data available"

    if tool.endswith("bbox_mean"):
        region = result.get("region") or []
        return f"{date}｜bbox {region}: {payload}"
    if tool.endswith("point_value"):
        region = result.get("region") or []
        return f"{date}｜point {region}: {payload}"
    return f"{date}: {payload}"


async def _run_onepass_async(query: str, debug: bool) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()

    def _call() -> Any:
        return run_onepass(query, k=6, collections=None, temperature=0.2, debug=debug)

    result = await loop.run_in_executor(None, _call)

    res_mode = getattr(result, "mode", None)
    res_text = getattr(result, "text", None)
    res_code = getattr(result, "code", None)
    res_plan = getattr(result, "plan", None)
    res_citations = getattr(result, "citations", None) or []
    res_debug = getattr(result, "debug", None)

    payload_out: Dict[str, Any] = {
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

    if debug and isinstance(res_debug, dict):
        dbg = dict(res_debug)
        dbg.setdefault("mode", payload_out["mode"])
        dbg.setdefault("code_len", len(res_code or ""))
        if plan_obj:
            dbg["plan"] = plan_obj
        payload_out["debug"] = dbg

    return payload_out


async def _execute_mcp_tool(
    decision: MCPDecision,
    today: str,
    debug: bool,
    query: str,
) -> Dict[str, Any]:
    if decision.tool_name is None:
        raise ToolError("Missing tool name for MCP decision")

    args = decision.arguments or {}
    args = dict(args)
    tool = decision.tool_name

    date = _ensure_date(args.get("date"), today)
    fields = _ensure_fields(args.get("fields"))
    method_value = args.get("method")
    method, explicit_method = _normalize_method(method_value)
    if _should_force_nearest(explicit_method, method, date, today, query):
        method = "nearest"

    if tool == "ghrsst.point_value":
        lon, lat = _ensure_coordinates(args)
        normalized_args = {
            "longitude": lon,
            "latitude": lat,
            "date": date,
            "fields": fields,
            "method": method,
        }
        try:
            result = await _ghrsst_point_value(
                lon, lat, date=date, fields=fields, method=method
            )
        except ToolError as exc:
            message = str(exc)
            if method != "nearest" and _is_missing_data_error(message):
                method = "nearest"
                normalized_args["method"] = method
                try:
                    result = await _ghrsst_point_value(
                        lon, lat, date=date, fields=fields, method=method
                    )
                except ToolError as exc2:
                    return _build_mcp_error_response(
                        tool, normalized_args, decision, str(exc2), debug
                    )
            else:
                return _build_mcp_error_response(
                    tool, normalized_args, decision, message, debug
                )

    elif tool == "ghrsst.bbox_mean":
        bbox = _ensure_bbox(args.get("bbox"))
        normalized_args = {
            "bbox": bbox,
            "date": date,
            "fields": fields,
            "method": method,
        }
        try:
            result = await _ghrsst_bbox_mean(
                bbox=bbox, date=date, fields=fields, method=method
            )
        except ToolError as exc:
            message = str(exc)
            if method != "nearest" and _is_missing_data_error(message):
                method = "nearest"
                normalized_args["method"] = method
                try:
                    result = await _ghrsst_bbox_mean(
                        bbox=bbox, date=date, fields=fields, method=method
                    )
                except ToolError as exc2:
                    return _build_mcp_error_response(
                        tool, normalized_args, decision, str(exc2), debug
                    )
            else:
                return _build_mcp_error_response(
                    tool, normalized_args, decision, message, debug
                )

    else:
        raise ToolError(f"Unsupported MCP tool: {tool}")

    if not isinstance(result, dict):
        return _build_mcp_error_response(
            tool,
            normalized_args,
            decision,
            "Unexpected response format from GHRSST tool",
            debug,
        )

    result_date = result.get("date")
    if method == "nearest" and result_date:
        normalized_args.setdefault("requested_date", date)
        normalized_args["resolved_date"] = result_date

    text = _render_tool_text(tool, result)
    if method == "nearest" and result_date and result_date != date:
        text += f"（原請求 {date}）"

    response: Dict[str, Any] = {
        "mode": "mcp_tools",
        "tool": tool,
        "arguments": normalized_args,
        "result": result,
        "text": text,
        "citations": [],
    }

    if result_date:
        response["resolved_date"] = result_date

    if decision.confidence is not None:
        response["confidence"] = decision.confidence
    if decision.raw_plan:
        response["plan"] = decision.raw_plan
    if debug:
        response["debug"] = {"plan": decision.raw_plan, "arguments": normalized_args}

    return response


async def classify_and_route(query: str, debug: bool = False) -> Dict[str, Any]:
    today = _today_tpe()
    loop = asyncio.get_running_loop()
    decision = await loop.run_in_executor(None, _classify_sync, query, today)

    logger.info(
        "router decision=%s tool=%s confidence=%s",
        decision.decision,
        decision.tool_name,
        decision.confidence,
    )

    if decision.decision == "mcp_tools":
        return await _execute_mcp_tool(decision, today, debug, query)

    rag_result = await _run_onepass_async(query, debug)
    if decision.confidence is not None:
        rag_result.setdefault("confidence", decision.confidence)
    if decision.raw_plan and "plan" not in rag_result:
        rag_result["plan"] = decision.raw_plan
    return rag_result


def register_router_tool(mcp: FastMCP) -> None:
    @mcp.tool("router.answer")
    async def router_answer(query: str, debug: bool = False) -> Dict[str, Any]:
        return await classify_and_route(query, debug=debug)
