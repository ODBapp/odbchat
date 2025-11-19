"""Lightweight classifier/router for deciding between MCP tools and RAG."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from server.api.ghrsst_mcp_proxy import _ghrsst_bbox_mean, _ghrsst_point_value
from server.api.metocean_mcp_proxy import _tide_forecast
from server.keyword_sets import (
    CLI_TOOL_REGEX,
    CODE_REGEX,
    GHRSST_GEO_REGEX,
    GHRSST_REGEX,
    NO_CODE_REGEX,
    SEA_STATE_REGEX,
    TIDE_REGEX,
    TIDE_STRONG_REGEX,
    TIME_HINT_REGEX,
)
from server.rag.onepass_core import run_onepass
from server.tools.rag_onepass_tool import (
    _norm_citation,
    _norm_mode,
    _norm_plan,
)
from server.time_utils import now_iso_in_tz, resolve_tz_name, today_in_tz

logger = logging.getLogger("odbchat.router")

DEFAULT_FIELDS = ["sst", "sst_anomaly"]
BBOX_KEY_SYNONYMS = {
    "lon0": ("lon0", "west", "left", "minlon", "lon_min"),
    "lat0": ("lat0", "south", "bottom", "minlat", "lat_min"),
    "lon1": ("lon1", "east", "right", "maxlon", "lon_max"),
    "lat1": ("lat1", "north", "top", "maxlat", "lat_max"),
}
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
PAIR_RE = re.compile(r"\(\s*([-\d\.]+[NSEWnsew]?)\s*[,\uff0c]\s*([-\d\.]+[NSEWnsew]?)\s*\)")
LOOSE_PAIR_RE = re.compile(r"([-\d\.]+[NSEWnsew]?)\s*[,\uff0c]\s*([-\d\.]+[NSEWnsew]?)")
BBOX_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)(?:\s*,\s*(-?\d+(?:\.\d+)?)){3}\s*\]")
DATE_ISO_RE = re.compile(r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})")
DATE_SLASH_RE = re.compile(r"(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2})")
DATE_CJK_RE = re.compile(r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日?")
DURATION_RE = re.compile(
    r"P?(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?"
)
LEGACY_STATE_LABELS = {
    "rising": "漲潮",
    "falling": "退潮",
    "high": "滿潮",
    "low": "乾潮",
}
TIDE_PHASE_MAP = {
    "Waning Crescent": "殘月",
    "Third Quarter": "下弦月",
    "Waning Gibbous": "虧凸月",
    "Full Moon": "滿月",
    "Waxing Gibbous": "盈凸月",
    "First Quarter": "上弦月",
    "Waxing Crescent": "眉月",
    "New Moon": "新月",
}
NOTE_REPLACEMENTS = (
    ("Tide heights", "潮高"),
    ("MSL", "平均海水面"),
)
STATE_LABEL_EN = {
    "rising": "rising tide",
    "falling": "falling tide",
    "high": "high tide",
    "low": "low tide",
}
LABEL_MAP_EN = {"high": "high tide", "low": "low tide"}
SPECIAL_NOTES = {
    "Note: Tide heights are relative to NTDE 1983–2001 MSL and provided for reference only.":
        "Note: 潮高為相對於 NTDE 1983–2001 平均海水面為基準，僅供參考。",
}



@dataclass
class MCPDecision:
    decision: str
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] | None = None
    confidence: float | None = None
    raw_plan: Dict[str, Any] | None = None


def _is_zh_query(query: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", query))


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
    if isinstance(value, str):
        normalized = _normalize_date_string(value)
        if normalized:
            return normalized
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


def _format_date_parts(year: str, month: str, day: str) -> str | None:
    try:
        y = int(year)
        m = int(month)
        d = int(day)
        if not (1 <= m <= 12 and 1 <= d <= 31):
            return None
        return f"{y:04d}-{m:02d}-{d:02d}"
    except Exception:
        return None


def _normalize_date_string(value: str) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None
    match = DATE_ISO_RE.fullmatch(stripped)
    if match:
        return _format_date_parts(match.group("year"), match.group("month"), match.group("day"))
    for regex in (DATE_SLASH_RE, DATE_CJK_RE, DATE_ISO_RE):
        match = regex.search(stripped)
        if match:
            formatted = _format_date_parts(match.group("year"), match.group("month"), match.group("day"))
            if formatted:
                return formatted
    return None


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
    return bool(TIME_HINT_REGEX.search(lowered))


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


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


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


def _normalize_tide_arguments(
    raw: Dict[str, Any] | None,
    tz: str | None,
    query_time: str | None,
    today: str,
    requested_date: str | None = None,
) -> Dict[str, Any]:
    data = dict(raw or {})
    tz_value = resolve_tz_name(data.get("tz") or tz)
    query_time_value = data.get("query_time") or query_time or now_iso_in_tz(tz_value)
    raw_date = data.get("date")
    if isinstance(raw_date, str):
        stripped = raw_date.strip()
        if not stripped or stripped in {"{today}", "today", "TODAY"}:
            date_value = today
        else:
            date_value = _normalize_date_string(stripped) or today
    else:
        date_value = today
    if requested_date:
        date_value = requested_date
    station_id = data.get("station_id")
    lon = data.get("longitude")
    lat = data.get("latitude")
    if station_id is None:
        if lon is None or lat is None:
            raise ToolError("tide.forecast requires station_id or longitude+latitude")
        try:
            lon = float(lon)
            lat = float(lat)
        except Exception as exc:
            raise ToolError("longitude and latitude must be numeric for tide.forecast") from exc
    normalized = {
        "station_id": station_id,
        "longitude": lon,
        "latitude": lat,
        "tz": tz_value,
        "query_time": query_time_value,
        "date": date_value,
    }
    return {k: v for k, v in normalized.items() if v is not None}


def _format_duration_text(value: str | None) -> str:
    if not value:
        return ""
    match = DURATION_RE.match(value)
    if not match:
        return ""
    parts: list[str] = []
    days = match.group("days")
    hours = match.group("hours")
    minutes = match.group("minutes")
    if days and days != "0":
        parts.append(f"{int(days)}天")
    if hours and hours != "0":
        parts.append(f"{int(hours)}小時")
    if minutes and minutes != "0":
        parts.append(f"{int(minutes)}分")
    if not parts and match.group("seconds"):
        parts.append(f"{int(match.group('seconds'))}秒")
    return "".join(parts)


def _fmt_short_time(value: str | None) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime("%H:%M")
    except Exception:
        return value


def _extreme_label(ext: Dict[str, Any] | None) -> str:
    if not ext:
        return ""
    ext_type = (ext.get("type") or "").lower()
    return "滿潮" if ext_type == "high" else ("乾潮" if ext_type == "low" else ext_type)


def _event_label(event_type: str | None, force_zh: bool) -> str:
    if not event_type:
        return ""
    event_type = event_type.lower()
    if force_zh:
        return "滿潮" if event_type == "high" else ("乾潮" if event_type == "low" else event_type)
    return LABEL_MAP_EN.get(event_type, event_type)


def _sun_moon_segments(result: Dict[str, Any], force_zh: bool, joiner: str) -> List[str]:
    segments: List[str] = []
    sun = result.get("sun") or {}
    if isinstance(sun, dict):
        dawn = _fmt_short_time(sun.get("begin_civil_twilight") or sun.get("civil_dawn"))
        dusk = _fmt_short_time(sun.get("end_civil_twilight") or sun.get("civil_dusk"))
        sunrise = _fmt_short_time(sun.get("sunrise") or sun.get("rise"))
        sunset = _fmt_short_time(sun.get("sunset") or sun.get("set"))
    else:
        dawn = dusk = sunrise = sunset = ""
    if dawn or sunrise or sunset or dusk:
        labels = {
            "dawn": "曙光" if force_zh else "Civil dawn",
            "sunrise": "日出" if force_zh else "Sunrise",
            "sunset": "日落" if force_zh else "Sunset",
            "dusk": "暮光" if force_zh else "Civil dusk",
        }
        sun_bits: List[str] = []
        if dawn:
            sun_bits.append(f"{labels['dawn']} {dawn}")
        if sunrise:
            sun_bits.append(f"{labels['sunrise']} {sunrise}")
        if sunset:
            sun_bits.append(f"{labels['sunset']} {sunset}")
        if dusk:
            sun_bits.append(f"{labels['dusk']} {dusk}")
        if sun_bits:
            segments.append(joiner.join(sun_bits))

    moon = result.get("moon") or {}
    moonphase = result.get("moonphase") or {}
    if isinstance(moon, dict):
        moonrise = _fmt_short_time(moon.get("moonrise") or moon.get("rise"))
        moonset = _fmt_short_time(moon.get("moonset") or moon.get("set"))
    else:
        moonrise = moonset = ""
    phase_source = None
    if isinstance(moonphase, dict):
        phase_source = moonphase.get("current")
    if phase_source is None and isinstance(moon, dict):
        phase_source = moon.get("phase")
    phase_label = _translate_moon_phase(phase_source, force_zh)
    illumination = None
    if isinstance(moonphase, dict):
        illumination = moonphase.get("illumination") or moonphase.get("fracillum")
    if illumination is None and isinstance(moon, dict):
        illumination = moon.get("illumination")
    illum_text = _format_illumination(illumination)
    moonrise_label = "月出" if force_zh else "Moonrise"
    moonset_label = "月落" if force_zh else "Moonset"
    moon_bits: List[str] = []
    if moonrise:
        moon_bits.append(f"{moonrise_label} {moonrise}")
    if moonset:
        moon_bits.append(f"{moonset_label} {moonset}")
    if phase_label or illum_text:
        if force_zh:
            desc = "今日月相：" + (phase_label or phase_source or "")
            if illum_text:
                desc += f"(月盈:{illum_text})"
        else:
            desc = "Moon phase: " + (phase_label or phase_source or "")
            if illum_text:
                desc += f" (Illumination: {illum_text})"
        moon_bits.append(desc)
    if moon_bits:
        segments.append(joiner.join(moon_bits))

    return segments


def _format_event_entry(entry: Dict[str, Any], force_zh: bool) -> str:
    time_str = _fmt_short_time(entry.get("time"))
    height_str = _format_height(entry)
    if not time_str and not height_str:
        return ""
    if force_zh:
        if time_str and height_str:
            return f"{time_str} 高度{height_str}"
        if time_str:
            return time_str
        return f"高度{height_str}"
    # English
    parts: list[str] = []
    if time_str:
        parts.append(time_str)
    if height_str:
        parts.append(f"height {height_str}")
    return " ".join(parts)


def _format_tide_summary(
    highs: List[Dict[str, Any]],
    lows: List[Dict[str, Any]],
    result_date: str | None,
    force_zh: bool,
    include_date_label: bool,
    prefix_label: str | None = None,
) -> str:
    if not highs and not lows:
        return ""
    event_join = "、" if force_zh else ", "
    clause_join = "；" if force_zh else "; "
    clauses: list[str] = []

    def _build_clause(entries: List[Dict[str, Any]], label_zh: str, label_en: str) -> None:
        items = [entry for entry in entries if _fmt_short_time(entry.get("time"))]
        formatted = [ _format_event_entry(entry, force_zh) for entry in items ]
        formatted = [item for item in formatted if item]
        if formatted:
            label = label_zh if force_zh else label_en
            clauses.append(f"{label} {event_join.join(formatted)}")

    _build_clause(highs, "滿潮", "High tide")
    _build_clause(lows, "乾潮", "Low tide")

    summary = clause_join.join(clauses)
    if prefix_label and summary:
        summary = f"{prefix_label}{summary}"
    if include_date_label and result_date:
        date_label = f"資料日期 {result_date}" if force_zh else f"Date {result_date}"
        if summary:
            summary += ("。" if force_zh else ". ") + date_label
        else:
            summary = date_label
    if not force_zh and summary and not summary.endswith("."):
        summary += "."
    return summary


def _format_height(entry: Dict[str, Any] | None) -> str:
    if not entry:
        return ""
    if "height_cm" in entry and entry["height_cm"] is not None:
        value = float(entry["height_cm"])
        return f"{value:.0f} cm"
    if "height_m" in entry and entry["height_m"] is not None:
        value = float(entry["height_m"])
        return f"{value:.2f} m"
    return ""


def _format_timespan(delta: timedelta | None, force_zh: bool) -> str:
    if delta is None:
        return ""
    seconds = int(abs(delta.total_seconds()))
    hours, rem = divmod(seconds, 3600)
    minutes, _ = divmod(rem, 60)
    parts: list[str] = []
    if force_zh:
        if hours:
            parts.append(f"{hours}小時")
        if minutes:
            parts.append(f"{minutes}分")
        if not parts and seconds:
            parts.append(f"{seconds}秒")
        return "".join(parts)
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if not parts and seconds:
        parts.append(f"{seconds}s")
    return "".join(parts)


def _format_illumination(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        num = float(value)
        if num <= 1:
            return f"{int(round(num * 100))}%"
        return f"{int(round(num))}%"
    except Exception:
        return str(value)


def _render_legacy_tide_text(result: Dict[str, Any], force_zh: bool) -> str:
    pieces: list[str] = []
    joiner = "、" if force_zh else ", "
    sentence_join = "。" if force_zh else ". "
    label_map = {"high": "滿潮", "low": "乾潮"} if force_zh else LABEL_MAP_EN
    state_map = LEGACY_STATE_LABELS if force_zh else STATE_LABEL_EN
    state = (result.get("state_now") or "").lower()
    state_label = state_map.get(state, state or "")

    next_ext = result.get("next_extreme") or {}
    last_ext = result.get("last_extreme") or {}
    until_text = _format_duration_text(result.get("until_extreme"))
    since_text = _format_duration_text(result.get("since_extreme"))
    next_label = _event_label(next_ext.get("type"), force_zh)
    last_label = _event_label(last_ext.get("type"), force_zh)
    next_time = _fmt_short_time(next_ext.get("time") if isinstance(next_ext, dict) else None)
    last_time = _fmt_short_time(last_ext.get("time") if isinstance(last_ext, dict) else None)

    clause_parts: list[str] = []
    if state_label:
        clause_parts.append(f"{'現在屬於' if force_zh else 'Currently '}{state_label}")
    if next_label and until_text:
        clause_parts.append(f"{'距' if force_zh else 'Until next '}{next_label}{until_text if force_zh else ''}")
    elif next_label and next_time:
        clause_parts.append(f"{next_label} {next_time}")
    if clause_parts:
        pieces.append(("，" if force_zh else ", ").join(clause_parts))

    extreme_desc: list[str] = []
    if next_label and next_time:
        extreme_desc.append(f"{next_label} {next_time}")
    if last_label and last_time:
        last_piece = f"{last_label} {last_time}"
        if since_text:
            last_piece += (f"（已過 {since_text}）" if force_zh else f" (elapsed {since_text})")
        extreme_desc.append(last_piece)
    if extreme_desc:
        pieces.append(("；" if force_zh else "; ").join(extreme_desc))

    pieces.extend(_sun_moon_segments(result, force_zh, joiner))

    tide_data = result.get('tide') or {}
    tide_available = bool(tide_data.get('highs') or tide_data.get('lows'))
    note_blocks: list[str] = []
    if tide_available:
        meta = result.get('meta') or {}
        status = ''
        if isinstance(meta, dict):
            status = str(meta.get('status') or '').strip()
        if status:
            note_blocks.append(_translate_note_text(status, force_zh))
        messages = result.get('messages') or []
        if isinstance(messages, list):
            note = ' '.join(str(m).strip() for m in messages if m)
            if note:
                note_blocks.append(_translate_note_text(note, force_zh))

    text = sentence_join.join(part for part in pieces if part).strip()
    if not force_zh and text and not text.endswith('.'):
        text += '.'
    for note in note_blocks:
        if note:
            text = f"{text}\n{note}" if text else note
    return text or ('潮汐資料暫時無法取得。' if force_zh else 'Tide information unavailable.')


def _render_tide_text(
    result: Dict[str, Any],
    args: Dict[str, Any],
    force_zh: bool,
    show_tide: bool = True,
) -> str:
    tide = result.get('tide') or {}
    joiner = "、" if force_zh else ", "
    sentence_join = "。" if force_zh else ". "
    label_map = {"high": "滿潮", "low": "乾潮"} if force_zh else LABEL_MAP_EN
    state_map = LEGACY_STATE_LABELS if force_zh else STATE_LABEL_EN
    highs = tide.get('highs') or []
    lows = tide.get('lows') or []
    tide_available = bool(highs or lows)
    result_date_raw = result.get('date') or args.get('date')
    result_date = _normalize_date_string(str(result_date_raw)) if result_date_raw else None
    qtime = _parse_iso_datetime(args.get('query_time'))
    same_day = bool(qtime and result_date and qtime.date().isoformat() == result_date)
    summary_mode = show_tide and tide_available and bool(result_date) and not same_day
    relative_mode = show_tide and not summary_mode
    events: list[tuple[str, datetime, Dict[str, Any]]] = []
    for entry in highs:
        dt = _parse_iso_datetime(entry.get('time'))
        if dt:
            events.append(('high', dt, entry))
    for entry in lows:
        dt = _parse_iso_datetime(entry.get('time'))
        if dt:
            events.append(('low', dt, entry))
    events.sort(key=lambda item: item[1])

    if not events:
        if not show_tide:
            astro_segments = _sun_moon_segments(result, force_zh, joiner)
            text = sentence_join.join(seg for seg in astro_segments if seg).strip()
            if not force_zh and text and not text.endswith('.'):
                text += '.'
            if text:
                return text
            return '暫時無法取得日月資訊。' if force_zh else 'Sun/moon information unavailable.'
        legacy_text = _render_legacy_tide_text(result, force_zh)
        if legacy_text:
            return legacy_text
        return '潮汐資料暫時無法取得。' if force_zh else 'Tide information unavailable.'

    pieces: list[str] = []
    prev_event: tuple[str, datetime, Dict[str, Any]] | None = None
    next_event: tuple[str, datetime, Dict[str, Any]] | None = None
    if relative_mode:
        if qtime:
            for event in events:
                if event[1] <= qtime:
                    prev_event = event
                if event[1] > qtime:
                    next_event = event
                    break
        if not qtime and events:
            next_event = events[0]

    summary_block = ""
    note_blocks: list[str] = []

    if relative_mode:
        state_label = ''
        if prev_event and next_event:
            if prev_event[0] == 'low' and next_event[0] == 'high':
                state_label = state_map.get('rising', '')
            elif prev_event[0] == 'high' and next_event[0] == 'low':
                state_label = state_map.get('falling', '')
        if state_label:
            prefix = '目前屬於' if force_zh else 'Currently '
            pieces.append(f"{prefix}{state_label}")

        if next_event:
            label = label_map.get(next_event[0], next_event[0])
            time_str = _fmt_short_time(next_event[1].isoformat())
            height_str = _format_height(next_event[2])
            clause = ("下一次" if force_zh else "Next ") + label
            if time_str:
                clause += f" {time_str}"
            if height_str:
                clause += (f" 高度{height_str}" if force_zh else f" height {height_str}")
            if qtime:
                delta = next_event[1] - qtime
                if delta.total_seconds() > 0:
                    span = _format_timespan(delta, force_zh)
                    if span:
                        clause += f"（約 {span} 後）" if force_zh else f" (about {span} later)"
            pieces.append(clause)

        if prev_event:
            label = label_map.get(prev_event[0], prev_event[0])
            time_str = _fmt_short_time(prev_event[1].isoformat())
            height_str = _format_height(prev_event[2])
            clause = ("上一個" if force_zh else "Previous ") + label
            if time_str:
                clause += f" {time_str}"
            if height_str:
                clause += (f" 高度{height_str}" if force_zh else f" height {height_str}")
            if qtime:
                delta = qtime - prev_event[1]
                if delta.total_seconds() > 0:
                    span = _format_timespan(delta, force_zh)
                    if span:
                        clause += f"（已過 {span}）" if force_zh else f" (elapsed {span})"
            pieces.append(clause)
    elif summary_mode:
        summary_block = _format_tide_summary(
            highs,
            lows,
            result_date,
            force_zh,
            include_date_label=True,
        )

    pieces.extend(_sun_moon_segments(result, force_zh, joiner))

    if tide_available and show_tide:
        if not summary_mode:
            label = "潮位資訊：" if force_zh else "Tide list: "
            summary_block = _format_tide_summary(
                highs,
                lows,
                result_date,
                force_zh,
                include_date_label=False,
                prefix_label=label,
            )
        messages = result.get('messages') or []
        if isinstance(messages, list):
            note = ' '.join(str(m).strip() for m in messages if m)
            if note:
                note_blocks.append(_translate_note_text(note, force_zh))
        meta = result.get('meta') or {}
        status = ''
        if isinstance(meta, dict):
            status = str(meta.get('status') or '').strip()
        if status:
            note_blocks.append(_translate_note_text(status, force_zh))

    text = sentence_join.join(part for part in pieces if part).strip()
    if not force_zh and text and not text.endswith('.'):
        text += '.'
    if summary_block:
        summary_block = summary_block.strip()
        if summary_block:
            if text:
                text = f"{text}\n{summary_block}"
            else:
                text = summary_block

    for note in note_blocks:
        if note:
            text = f"{text}\n{note}" if text else note

    return text or ('潮汐資料暫時無法取得。' if force_zh else 'Tide information unavailable.')

async def _run_onepass_async(
    query: str,
    debug: bool,
    today: str,
    tz: str,
    query_time: str,
) -> tuple[Any, Dict[str, Any]]:
    loop = asyncio.get_running_loop()

    def _call() -> Any:
        return run_onepass(
            query,
            k=6,
            collections=None,
            temperature=0.2,
            debug=debug,
            today=today,
            tz=tz,
            query_time=query_time,
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
    tz: str | None = None,
    query_time: str | None = None,
    force_zh: bool = True,
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

    elif tool == "tide.forecast":
        requested_date = _extract_requested_date(query)
        normalized_args = _normalize_tide_arguments(
            args,
            tz,
            query_time,
            today,
            requested_date=requested_date,
        )
        try:
            result = await _tide_forecast(**normalized_args)
        except ToolError as exc:
            return _build_mcp_error_response(
                tool,
                normalized_args,
                decision,
                str(exc),
                debug,
            )
        if not isinstance(result, dict):
            return _build_mcp_error_response(
                tool,
                normalized_args,
                decision,
                "Unexpected response format from tide.forecast",
                debug,
            )
        show_tide = _query_requests_tide_details(query)
        tide_available = bool((result.get("tide") or {}).get("highs") or (result.get("tide") or {}).get("lows"))
        text = _render_tide_text(result, normalized_args, force_zh, show_tide=show_tide)
        if show_tide and not tide_available:
            detail = "此點位並無潮汐資料。" if force_zh else "No tide data is available for this location."
            text = f"{text}\n{detail}" if text else detail
        response = {
            "mode": "mcp_tools",
            "tool": tool,
            "arguments": normalized_args,
            "result": result,
            "text": text,
            "citations": [],
        }
        if decision.confidence is not None:
            response["confidence"] = decision.confidence
        if decision.raw_plan:
            response["plan"] = decision.raw_plan
        if debug:
            response["debug"] = {"plan": decision.raw_plan, "arguments": normalized_args}
        return response

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


def _contains_no_code_phrase(query: str) -> bool:
    if not query:
        return False
    normalized = _normalize_fullwidth(query).lower()
    return bool(NO_CODE_REGEX.search(normalized))


def _is_sea_state_only_query(query: str) -> bool:
    if not query:
        return False
    normalized = _normalize_fullwidth(query).lower()
    return bool(SEA_STATE_REGEX.search(normalized))


def _contains_code_request(query: str) -> bool:
    if not query:
        return False
    lowered = query.lower()
    if NO_CODE_REGEX.search(lowered):
        return False
    if CLI_TOOL_REGEX.search(lowered):
        return False
    return bool(CODE_REGEX.search(lowered))


def _extract_point_from_query(query: str) -> Tuple[float, float] | None:
    normalized = _normalize_fullwidth(query)
    match = PAIR_RE.search(normalized)
    if not match:
        match = LOOSE_PAIR_RE.search(normalized)
    if not match:
        return None
    first_raw = match.group(1)
    second_raw = match.group(2)
    first_val, first_dir = _parse_coordinate_token(first_raw)
    second_val, second_dir = _parse_coordinate_token(second_raw)
    if first_val is None or second_val is None:
        return None
    lon, lat = _resolve_point_components(first_val, first_dir, second_val, second_dir)
    if abs(lon) > 180 or abs(lat) > 90:
        return None
    return lon, lat


def _parse_coordinate_token(token: str) -> Tuple[float | None, str | None]:
    token = token.strip()
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)([NSEWnsew])?", token)
    if not match:
        return None, None
    value = float(match.group(1))
    direction = match.group(2)
    if direction:
        direction = direction.upper()
        if direction in {"S", "W"}:
            value = -abs(value)
    return value, direction


def _resolve_point_components(
    first_val: float,
    first_dir: str | None,
    second_val: float,
    second_dir: str | None,
) -> Tuple[float, float]:
    lon: float | None = None
    lat: float | None = None

    for value, direction in ((first_val, first_dir), (second_val, second_dir)):
        if direction in {"N", "S"}:
            lat = value
        elif direction in {"E", "W"}:
            lon = value

    if lon is not None and lat is not None:
        return lon, lat

    if lon is None or lat is None:
        lon_guess, lat_guess = _normalize_point_order(first_val, second_val)
        if lon is None:
            lon = lon_guess
        if lat is None:
            lat = lat_guess

    return lon if lon is not None else first_val, lat if lat is not None else second_val


def _normalize_point_order(first: float, second: float) -> Tuple[float, float]:
    """Attempt to infer lon/lat order based on value ranges."""
    lon, lat = first, second
    first_abs = abs(first)
    second_abs = abs(second)
    if first_abs <= 90 < second_abs <= 180:
        lon, lat = second, first
    elif first_abs > 90 and second_abs <= 90:
        lon, lat = first, second
    return lon, lat


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


def _translate_moon_phase(phase: str | None, force_zh: bool) -> str:
    if not phase:
        return ""
    if force_zh:
        return TIDE_PHASE_MAP.get(phase, phase)
    return phase


def _translate_note_text(note: str, force_zh: bool) -> str:
    if not force_zh:
        return note
    normalized = note.strip()
    if normalized in SPECIAL_NOTES:
        return SPECIAL_NOTES[normalized]
    translated = note
    for en, zh in NOTE_REPLACEMENTS:
        translated = translated.replace(en, zh)
    return translated


def _looks_like_ghrsst_query(query: str) -> bool:
    if not query:
        return False
    normalized = _normalize_fullwidth(query)
    lowered = normalized.lower()
    if not GHRSST_REGEX.search(lowered):
        return False
    if _extract_point_from_query(query) or _extract_bbox_from_query(query):
        return True
    if GHRSST_GEO_REGEX.search(lowered):
        return True
    digits = sum(ch.isdigit() for ch in normalized)
    return bool(TIME_HINT_REGEX.search(lowered) and digits >= 2)


def _infer_ghrsst_decision_from_query(query: str, today: str) -> MCPDecision | None:
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


def _looks_like_tide_query(query: str) -> bool:
    if not query:
        return False
    normalized = _normalize_fullwidth(query).lower()
    return bool(TIDE_REGEX.search(normalized))


def _query_requests_tide_details(query: str) -> bool:
    if not query:
        return False
    normalized = _normalize_fullwidth(query).lower()
    return bool(TIDE_STRONG_REGEX.search(normalized))


def _extract_requested_date(query: str) -> str | None:
    if not query:
        return None
    normalized = _normalize_fullwidth(query)
    for regex in (DATE_CJK_RE, DATE_SLASH_RE, DATE_ISO_RE):
        match = regex.search(normalized)
        if match:
            formatted = _format_date_parts(match.group("year"), match.group("month"), match.group("day"))
            if formatted:
                return formatted
    return None



def _infer_tide_decision_from_query(
    query: str,
    tz: str,
    query_time: str,
    today: str,
) -> MCPDecision | None:
    if _contains_code_request(query) or not _looks_like_tide_query(query):
        return None
    point = _extract_point_from_query(query)
    if not point:
        return None
    lon, lat = point
    requested_date = _extract_requested_date(query)
    date_value = requested_date or today
    arguments = {
        "longitude": lon,
        "latitude": lat,
        "tz": tz,
        "query_time": query_time,
        "date": date_value,
    }
    return MCPDecision(
        decision="mcp_tools",
        tool_name="tide.forecast",
        arguments=arguments,
    )


async def classify_and_route(
    query: str,
    tz: str | None = None,
    query_time: str | None = None,
    debug: bool = False,
) -> Dict[str, Any]:
    tz_value = resolve_tz_name(tz)
    query_time_value = query_time or now_iso_in_tz(tz_value)
    today = today_in_tz(tz_value)
    force_zh = _is_zh_query(query)
    decision: MCPDecision | None = None
    decision: MCPDecision | None = None

    try:
        raw_result, payload = await _run_onepass_async(
            query,
            debug,
            today,
            tz_value,
            query_time_value,
        )
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

    needs_tide = _looks_like_tide_query(query) and not _contains_code_request(query)

    if _is_sea_state_only_query(query) and not needs_tide:
        return {
            "mode": "explain",
            "text": "目前僅提供海表溫度與潮汐資訊，無法回答「海況/sea state」相關資料。請改以潮汐或海溫為關鍵字重新提問。",
            "citations": [],
        }

    if mode == "mcp_tools":
        spec = getattr(raw_result, "mcp", None) or payload.get("mcp")
        decision = _decision_from_spec(spec)
        if decision is None or not decision.tool_name:
            logger.error("mcp mode missing tool specification: %s", spec)
            auto_decision = None
            if _looks_like_tide_query(query):
                auto_decision = _infer_tide_decision_from_query(
                    query,
                    tz_value,
                    query_time_value,
                    today,
                )
            if auto_decision is None and _looks_like_ghrsst_query(query):
                auto_decision = _infer_ghrsst_decision_from_query(query, today)
            if auto_decision is None:
                fallback = {
                    "mode": "explain",
                    "text": "MCP 工具決策缺少資料，請重新提問或提供明確參數。",
                    "citations": [],
                }
                if debug:
                    fallback["debug"] = {"error": "missing_mcp_spec", "spec": spec}
                return fallback
            decision = auto_decision
        return await _execute_mcp_tool(
            decision,
            today,
            debug,
            query,
            tz_value,
            query_time_value,
            force_zh,
        )

    if needs_tide:
        auto_tide = _infer_tide_decision_from_query(query, tz_value, query_time_value, today)
        if auto_tide:
            logger.info("router auto-switching to tide.forecast based on query parsing (mode=%s)", mode or "<unknown>")
            return await _execute_mcp_tool(
                auto_tide,
                today,
                debug,
                query,
                tz_value,
                query_time_value,
                force_zh,
            )
    gh_request = _looks_like_ghrsst_query(query) and not _contains_code_request(query)
    if gh_request:
        auto_decision = _infer_ghrsst_decision_from_query(query, today)
        if auto_decision:
            logger.info("router auto-switching to mcp_tools based on query parsing")
            return await _execute_mcp_tool(
                auto_decision,
                today,
                debug,
                query,
                tz_value,
                query_time_value,
                force_zh,
            )

    return payload


def register_router_tool(mcp: FastMCP) -> None:
    @mcp.tool("router.answer")
    async def router_answer(
        query: str,
        tz: str | None = None,
        query_time: str | None = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        return await classify_and_route(query, tz=tz, query_time=query_time, debug=debug)
