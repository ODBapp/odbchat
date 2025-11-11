"""Lightweight classifier/router for deciding between MCP tools and RAG."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from server.api.ghrsst_mcp_proxy import _ghrsst_bbox_mean, _ghrsst_point_value
from server.rag.onepass_core import run_onepass
from server.tools.rag_onepass_tool import (
    _norm_citation,
    _norm_mode,
    _norm_plan,
)
from server.time_utils import today_tpe

logger = logging.getLogger("odbchat.router")

DEFAULT_FIELDS = ["sst", "sst_anomaly"]
BBOX_KEY_SYNONYMS = {
    "lon0": ("lon0", "west", "left", "minlon", "lon_min"),
    "lat0": ("lat0", "south", "bottom", "minlat", "lat_min"),
    "lon1": ("lon1", "east", "right", "maxlon", "lon_max"),
    "lat1": ("lat1", "north", "top", "maxlat", "lat_max"),
}
GHRSST_QUERY_KEYWORDS = (
    "海溫",
    "水溫",
    "sea surface temperature",
    "sea-surface temperature",
    "sea temperature",
    "sst",
)
CODE_REQUEST_KEYWORDS = (
    "程式",
    "python",
    "code",
    "寫",
    "繪",
    "畫",
    "plot",
    "腳本",
    "script",
)
FULLWIDTH_MAP = str.maketrans(
    {
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "，": ",",
        "。": ".",
        "：": ":",
        "－": "-",
        "　": " ",
    }
)
PAIR_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*[,\uff0c]\s*(-?\d+(?:\.\d+)?)\s*\)")
BBOX_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)(?:\s*,\s*(-?\d+(?:\.\d+)?)){3}\s*\]")
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



@dataclass
class MCPDecision:
    decision: str
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] | None = None
    confidence: float | None = None
    raw_plan: Dict[str, Any] | None = None


def _decision_from_spec(spec: Any) -> MCPDecision | None:
    if not isinstance(spec, dict):
        return None
    tool_field = spec.get("tool")
    arguments = spec.get("arguments")
    if isinstance(tool_field, dict):
        tool_name = str(tool_field.get("name") or tool_field.get("tool") or "").strip()
        if not isinstance(arguments, dict):
            arguments = tool_field.get("arguments")
    else:
        tool_name = str(tool_field or spec.get("name") or "").strip()
    if not tool_name:
        return None
    if not isinstance(arguments, dict):
        arguments = {}
    confidence = spec.get("confidence")
    try:
        if confidence is not None:
            confidence = float(confidence)
    except Exception:
        confidence = None
    return MCPDecision(
        decision="mcp_tools",
        tool_name=tool_name,
        arguments=arguments,
        confidence=confidence,
        raw_plan=spec if isinstance(spec, dict) else None,
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
        text = _normalize_fullwidth(value)
        match = BBOX_RE.search(text)
        if match:
            text = match.group(0)
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        floats = [float(num) for num in numbers]
        if len(floats) >= 4:
            value = floats[:4]
        else:
            raise ToolError("bbox must contain four numbers [lon0, lat0, lon1, lat1]")
    elif isinstance(value, dict):
        floats: List[float] = []
        for key in ("lon0", "lat0", "lon1", "lat1"):
            synonyms = BBOX_KEY_SYNONYMS.get(key, (key,))
            found = None
            for candidate in synonyms:
                if candidate in value:
                    found = value[candidate]
                    break
                upper = candidate.upper()
                if upper in value:
                    found = value[upper]
                    break
            if found is None:
                raise ToolError(f"bbox dict missing key '{key}'")
            try:
                floats.append(float(found))
            except Exception as exc:  # pragma: no cover
                raise ToolError(f"Invalid bbox values: {value}") from exc
        value = floats
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


async def _run_onepass_async(query: str, debug: bool, today: str) -> tuple[Any, Dict[str, Any]]:
    loop = asyncio.get_running_loop()

    def _call() -> Any:
        return run_onepass(
            query,
            k=6,
            collections=None,
            temperature=0.2,
            debug=debug,
            today=today,
        )

    result = await loop.run_in_executor(None, _call)

    res_mode = getattr(result, "mode", None)
    res_text = getattr(result, "text", None)
    res_code = getattr(result, "code", None)
    res_plan = getattr(result, "plan", None)
    res_citations = getattr(result, "citations", None) or []
    res_debug = getattr(result, "debug", None)
    res_mcp = getattr(result, "mcp", None)

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

    if payload_out["mode"] == "mcp_tools" and isinstance(res_mcp, dict):
        payload_out["mcp"] = res_mcp

    if debug and isinstance(res_debug, dict):
        dbg = dict(res_debug)
        dbg.setdefault("mode", payload_out["mode"])
        dbg.setdefault("code_len", len(res_code or ""))
        if plan_obj:
            dbg["plan"] = plan_obj
        payload_out["debug"] = dbg

    return result, payload_out


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
        try:
            bbox = _ensure_bbox(args.get("bbox"))
        except ToolError as exc:
            fallback_bbox = _extract_bbox_from_query(query)
            if fallback_bbox:
                bbox = fallback_bbox
            else:
                raise exc
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


def _normalize_fullwidth(text: str) -> str:
    return text.translate(FULLWIDTH_MAP)


def _contains_code_request(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in CODE_REQUEST_KEYWORDS)


def _extract_point_from_query(query: str) -> Tuple[float, float] | None:
    normalized = _normalize_fullwidth(query)
    match = PAIR_RE.search(normalized)
    if not match:
        return None
    try:
        lon = float(match.group(1))
        lat = float(match.group(2))
        return lon, lat
    except Exception:
        return None


def _extract_bbox_from_query(query: str) -> List[float] | None:
    normalized = _normalize_fullwidth(query)
    match = BBOX_RE.search(normalized)
    if not match:
        return None
    fragment = match.group(0)
    try:
        return _ensure_bbox(fragment)
    except ToolError:
        return None


def _looks_like_ghrsst_query(query: str) -> bool:
    if not query:
        return False
    normalized = _normalize_fullwidth(query).lower()
    if not any(keyword in normalized for keyword in GHRSST_QUERY_KEYWORDS):
        return False
    return bool(_extract_point_from_query(query) or _extract_bbox_from_query(query))


def _infer_mcp_decision_from_query(query: str, today: str) -> MCPDecision | None:
    if _contains_code_request(query):
        return None
    bbox = _extract_bbox_from_query(query)
    if bbox:
        arguments = {
            "bbox": bbox,
            "date": today,
            "fields": DEFAULT_FIELDS,
            "method": "nearest",
        }
        return MCPDecision(
            decision="mcp_tools",
            tool_name="ghrsst.bbox_mean",
            arguments=arguments,
        )
    point = _extract_point_from_query(query)
    if point:
        lon, lat = point
        arguments = {
            "longitude": lon,
            "latitude": lat,
            "date": today,
            "fields": DEFAULT_FIELDS,
            "method": "nearest",
        }
        return MCPDecision(
            decision="mcp_tools",
            tool_name="ghrsst.point_value",
            arguments=arguments,
        )
    return None


async def classify_and_route(query: str, debug: bool = False) -> Dict[str, Any]:
    today = today_tpe()
    try:
        raw_result, payload = await _run_onepass_async(query, debug, today)
    except Exception as exc:
        logger.error("router onepass failed: %s", exc)
        fallback = {
            "mode": "explain",
            "text": "路由判斷階段失敗或逾時，請稍後再試並使用 /llm status 檢查後端。",
            "source": "router.error",
            "citations": [],
        }
        if debug:
            fallback["debug"] = {"error": str(exc)}
        return fallback

    mode = (payload.get("mode") or "").strip().lower()
    logger.info("router mode=%s", mode or "<unknown>")

    if mode == "mcp_tools":
        spec = getattr(raw_result, "mcp", None) or payload.get("mcp")
        decision = _decision_from_spec(spec)
        if decision is None or not decision.tool_name:
            logger.error("mcp mode missing tool specification: %s", spec)
            fallback = {
                "mode": "explain",
                "text": "GHRSST 工具決策缺少資料，請重新提問或提供明確經緯度。",
                "citations": [],
            }
            if debug:
                fallback["debug"] = {"error": "missing_mcp_spec", "spec": spec}
            return fallback
        return await _execute_mcp_tool(decision, today, debug, query)

    if mode != "code" and _looks_like_ghrsst_query(query):
        auto_decision = _infer_mcp_decision_from_query(query, today)
        if auto_decision:
            logger.info("router auto-switching to mcp_tools based on query parsing")
            return await _execute_mcp_tool(auto_decision, today, debug, query)

    return payload


def register_router_tool(mcp: FastMCP) -> None:
    @mcp.tool("router.answer")
    async def router_answer(query: str, debug: bool = False) -> Dict[str, Any]:
        return await classify_and_route(query, debug=debug)
