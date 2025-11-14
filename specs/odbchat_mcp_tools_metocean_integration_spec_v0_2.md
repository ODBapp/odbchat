# ODBchat — Metocean MCP Integration Δ-Spec (for Codex)

**Date:** 2025-11-13 (Asia/Taipei)
**Scope:** Extend current MCP setup from “GHRSST-only” to a **Metocean** MCP surface by:

1. changing the upstream endpoint and client `User-Agent`, and
2. **adding a new tool `tide.forecast`** (tide + sun/moon) while **keeping existing GHRSST tools unchanged**.
3. updating the unified **single-pass classifier+router** so LLM can choose **`mcp | explain | code`** and, if `mcp`, pick **which** metocean tool and fill arguments.

> Note: Do **not** restate unchanged GHRSST logic. Only implement the deltas described here. For context, our current GHRSST proxy module and router spec are here:  

---

## 0) Background (what exists today)

* We already proxy upstream **GHRSST** MCP tools (point/bbox) via a local FastMCP server so clients can call them as first-class tools. 
* We have a **single-pass classifier+router** spec that routes free-text to `mcp | explain | code` in one LLM hop and, when `mcp`, selects GHRSST tool + arguments. 

This Δ-spec adds **Metocean** surface and one new tool.

---

## 1) What changes (summary)

1. **Upstream endpoint (MCP server)**

   * From: `https://eco.odb.ntu.edu.tw/mcp/ghrsst`
   * To (for metocean suite): `https://eco.odb.ntu.edu.tw/mcp/metocean`
2. **MCP Client User-Agent**

   * Set to `metocean-mcp` (when our local server connects to the upstream metocean MCP).
3. **New MCP tool** (upstream exposes it; we proxy it locally):

   * `tide.forecast` — integrates **ODB Tide API** (our tides) + **USNO** (sun/moon).
4. **Router/Classifier catalog update**

   * Add `tide.forecast` to the **tool catalog** so the unified classifier+router can select it for tide/sun/moon questions.
5. **CLI behavior**

   * For tide questions, the CLI must **always attach** the **current timezone** and a **query timestamp** so the tool can compute `state_now`, `last_extreme`, `next_extreme`, `since_extreme`, `until_extreme`.

Unchanged: **existing GHRSST tools and their proxy paths** (keep as-is).

---

## 2) New upstream surface & proxy (local server)

### 2.1 New proxy module

Create `server/api/metocean_mcp_proxy.py` (mirrors `ghrsst_mcp_proxy.py`, but points to metocean endpoint and sets UA):

```python
# server/api/metocean_mcp_proxy.py
from __future__ import annotations
import os, json, logging
from typing import Any
from fastmcp import FastMCP
from fastmcp.client import Client as MCPClient
from fastmcp.exceptions import ToolError

logger = logging.getLogger(__name__)

METOCEAN_MCP_URL = os.getenv(
    "METOCEAN_MCP_URL",
    "https://eco.odb.ntu.edu.tw/mcp/metocean",
)
METOCEAN_UA = os.getenv("METOCEAN_MCP_USER_AGENT", "metocean-mcp")

async def _call_metocean_tool(tool_name: str, payload: dict[str, Any]) -> Any:
    # NOTE: fastmcp.Client should support passing headers; if not, wrap transport to inject UA.
    headers = {"User-Agent": METOCEAN_UA}
    try:
        async with MCPClient(METOCEAN_MCP_URL, headers=headers) as client:
            result = await client.call_tool(tool_name, payload)
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("Metocean MCP call failed: %s", exc)
        raise ToolError(f"Failed upstream metocean tool {tool_name!r}: {exc}")

    # Normalize like GHRSST proxy does (data → structured_content → text → raw)
    if result.data is not None:
        return result.data
    if result.structured_content:
        return result.structured_content
    if result.content:
        texts = [blk.text for blk in (result.content or []) if getattr(blk, "text", None)]
        combined = "".join(texts).strip()
        if combined:
            try:
                return json.loads(combined)
            except json.JSONDecodeError:
                return combined
    return {"content": result.content, "structured_content": result.structured_content, "is_error": result.is_error}

def register_metocean_tools(mcp: FastMCP) -> None:
    """Register local proxy wrappers for upstream metocean tools."""

    @mcp.tool(name="tide.forecast")
    async def tide_forecast(
        longitude: float | None = None,
        latitude: float | None = None,
        # Optional: station_id if upstream supports station lookup
        station_id: str | None = None,
        # Time context
        date: str | None = None,            # YYYY-MM-DD (day to forecast; omit → today in tz)
        query_time: str | None = None,      # ISO 8601 timestamp (current moment)
        tz: str | None = None,              # IANA TZ (e.g., "Asia/Taipei")
        # Optional: output toggles
        include_sun_moon: bool = True
    ) -> Any:
        payload = {
            "longitude": longitude, "latitude": latitude,
            "station_id": station_id,
            "date": date, "query_time": query_time, "tz": tz,
            "include_sun_moon": include_sun_moon,
        }
        # Remove None to keep upstream payload clean
        payload = {k: v for k, v in payload.items() if v is not None}
        return await _call_metocean_tool("tide.forecast", payload)
```

> Use `ghrsst_mcp_proxy.py` as a reference for structure/normalization patterns. 

### 2.2 Register in local FastMCP server

In `server/odbchat_mcp_server.py`, after existing registrations:

```python
from server.api.metocean_mcp_proxy import register_metocean_tools
register_metocean_tools(mcp)  # exposes tide.forecast locally
```

(Leave GHRSST registration as-is.)

---

## 3) Tool contract (LLM-visible) — `tide.forecast`

> The upstream spec (`metocean_mcp_v0_2_0.spec`) governs precise fields. Below is the **LLM-facing** contract we expect from our proxy.

### 3.1 Input (what classifier+router will fill)

```json
{
  "longitude": 121.5,         // or use "station_id": "TPE-Danshui" if supported
  "latitude": 25.0,
  "date": "2025-11-13",       // day for forecast; omit → today in tz
  "query_time": "2025-11-13T16:05:00+08:00",  // NOW from CLI (ISO 8601)
  "tz": "Asia/Taipei",
  "include_sun_moon": true
}
```

* **Required for “now” state**: `query_time` (ISO-8601) **and** `tz`.
* If only tide extremes are needed and “now” status is irrelevant, `query_time` may be omitted (but we will still send it by default from CLI).

### 3.2 Output (key fields the router expects)

```json
{
  "date": "2025-11-13",
  "tz": "Asia/Taipei",
  "location": { "longitude": 121.5, "latitude": 25.0, "station_id": null },
  "state_now": "rising|falling|high|low|unknown",
  "last_extreme": { "type": "high|low", "time": "2025-11-13T13:12:00+08:00", "height": 1.23 },
  "next_extreme": { "type": "high|low", "time": "2025-11-13T19:36:00+08:00", "height": 0.87 },
  "since_extreme": "PT02H53M",   // ISO-8601 duration since last_extreme at query_time
  "until_extreme": "PT03H31M",   // ISO-8601 duration until next_extreme at query_time

  "high_tides": [
    {"time": "2025-11-13T06:01:00+08:00", "height": 1.05},
    {"time": "2025-11-13T18:23:00+08:00", "height": 1.11}
  ],
  "low_tides": [
    {"time": "2025-11-13T12:09:00+08:00", "height": 0.12},
    {"time": "2025-11-13T23:59:00+08:00", "height": 0.08}
  ],

  "sun":  { "sunrise": "2025-11-13T06:16:00+08:00", "sunset": "2025-11-13T17:07:00+08:00" },
  "moon": {
    "moonrise": "2025-11-13T15:52:00+08:00",
    "moonset": "2025-11-14T03:22:00+08:00",
    "phase": "Waxing Gibbous",
    "illumination": 0.78
  },

  "meta": {
    "sources": { "tide": "ODB Tide API", "sunmoon": "USNO" },
    "status": ""  // if USNO fails: put error message here; still return tide data
  }
}
```

* If USNO fails, keep `sun`/`moon` fields as `null` or omit them, and put the USNO error text into `meta.status` (still return tide).
* Error codes to surface upstream failures: `INVALID_ARGUMENT | NOT_FOUND | UNAVAILABLE` (same mapping we already use for GHRSST paths).

---

## 4) Router/Classifier deltas (single-pass)

Extend the **tool catalog** injected to the classifier to include `tide.forecast`:

```python
TOOL_CATALOG = [
  # existing GHRSST entries...
  {
    "name": "tide.forecast",
    "args_schema": {
      "longitude": "float (optional if station_id provided)",
      "latitude":  "float (optional if station_id provided)",
      "station_id":"string (optional if lon/lat provided)",
      "date":      "YYYY-MM-DD (optional; default=tz today)",
      "query_time":"ISO-8601 timestamp (required for state_now/since/until)",
      "tz":        "IANA TZ name (e.g., Asia/Taipei)",
      "include_sun_moon": "boolean (default true)"
    },
    "description": "Daily tide forecast + sun/moon. Computes 'state_now', last/next extremes, since/until durations."
  }
]
```

**Intent patterns** that should trigger `tide.forecast` (both zh/en):

* 「現在是漲潮還是退潮？」/ “Is it rising or falling now?”
* 「何時滿潮？乾潮？」/ “When is high/low tide?”
* 「何時日出？日落？何時月出？今天月相？」 / “Sunrise/sunset? Moonrise? Today’s moon phase?”
* Any query that contains: `潮`, `滿潮`, `乾潮`, `漲潮`, `退潮`, `日出`, `日落`, `月出`, `月相`, `tide`, `high tide`, `low tide`, `sunrise`, `sunset`, `moonrise`, `moon phase`.

**Slot filling policy** (the classifier returns arguments; the server still normalizes):

* **Location**: prefer explicit `lon/lat`; if user gives a named place you can add a registry later; for now we expect lon/lat in most CLI usage.
* **Time**:

  * `date`: if omitted, server fills **today** in `tz`.
  * **Always pass** `query_time` and `tz` from **CLI** (see §5). The server forwards them to the tool unchanged so `state_now` can be computed precisely.
* **include_sun_moon**: default `true`.

Rendering guideline in router: produce a **single concise line** answering the user’s verb (e.g., “現在屬於**漲潮**，距離**滿潮**約 **3 小時 31 分**。滿潮 18:23、乾潮 12:09；日出 06:16、日落 17:07；今日月相：盈凸月(78%).”).

---

## 5) CLI changes (free-text path)

* **No change** to the entry tool name if you already switched to `router.answer` (per v0.1 spec). 
* **Always attach time context** in the payload sent to the server (so router/classifier can include it in tool args):

  * `tz`: machine’s IANA TZ (e.g., from env or pytz; default `Asia/Taipei` for our deployment).
  * `query_time`: current time in **ISO-8601** with timezone offset.
* Implementation options:

  * Put `tz`/`query_time` inside the **router’s user context** (so LLM sees them and fills them into `tide.forecast.arguments`), or
  * Have the server enrich `tide.forecast` **after** plan creation (server inserts `tz`/`query_time` if decision=`mcp` and tool=`tide.forecast`).

We recommend the **first** (LLM sees and reuses them), with a server-side safety net that fills them if missing.

---

## 6) Server normalization (when executing `tide.forecast`)

* Ensure either `station_id` **or** (`longitude` & `latitude`) is present; reject otherwise with `INVALID_ARGUMENT`.
* If `date` missing → set to **today** in `tz`.
* If `query_time` missing → **synthesize** using server’s time in `tz` (but CLI should always supply it).
* Always forward `tz` unchanged; if missing → default `Asia/Taipei` (configurable).
* Keep error mapping consistent with GHRSST proxy.

---

## 7) LLM prompt deltas (classifier)

Add a short **tool usage note** to the system prompt:

> “For tide/sun/moon questions, choose `tide.forecast`. Always include `query_time` (ISO-8601) and `tz` in arguments so the tool can compute `state_now`, `last_extreme`, `next_extreme`, `since_extreme`, and `until_extreme`.”

Add 2–3 **few-shots**:

1. “(121.5,25.0) 現在是漲潮嗎？何時滿潮？” → `mcp` + `tide.forecast` with `longitude`, `latitude`, `date=now`, `query_time=now`, `tz=Asia/Taipei`.
2. “今天台北日出日落？月相？” → `mcp` + `tide.forecast` (works even if user only wants sun/moon).
3. “GHRSST 是什麼資料？” → `explain`.

---

## 8) Acceptance Criteria

* **AC-T1:** “(121.5,25.0) 現在是漲潮還是退潮？” → classifier selects **`tide.forecast`**; proxy calls upstream metocean MCP; router formats line with `state_now` + `until_extreme`/`since_extreme`.
* **AC-T2:** “何時滿潮？乾潮？日出？日落？月出？今天月相？” (with same coordinates) → single call to `tide.forecast` returns all; rendered in one line.
* **AC-T3:** If USNO down, **tide still returns**, `sun`/`moon` may be null; `meta.status` includes USNO error note; the answer states tide info and briefly says “sun/moon temporarily unavailable”.
* **AC-T4:** GHRSST flows remain unaffected (same outputs as before).
* **AC-T5:** CLI **always** passes `tz` and `query_time`; server fills defaults if missing.

---

## 9) Logging

Add structured fields for tide calls:

* `tool="tide.forecast"`, `lon`, `lat`, `station_id`, `date`, `tz`, `query_time`
* `state_now`, `last_extreme.type/time`, `next_extreme.type/time`
* latency: `t_classify_ms`, `t_mcp_ms`, `t_total_ms`
* upstream meta.status if present (USNO failures)

---

## 10) Work Breakdown (Δ only)

1. **New file:** `server/api/metocean_mcp_proxy.py` (as above) with UA injection and upstream URL.
2. **Register** metocean proxy in `server/odbchat_mcp_server.py` (keep GHRSST registration).
3. **Router/Classifier:** extend **tool catalog** and **prompt few-shots** to include `tide.forecast`; teach the LLM to always include `query_time`/`tz`. (Based on the existing unified router spec.) 
4. **CLI:** ensure `tz` + `query_time` are attached into the router input (and therefore into `tide.forecast` args).
5. **Render:** add a compact tide formatter that prints `state_now`, durations to next/last extremes, and sun/moon snippets when available.

---

### Notes & Constraints

* Do **not** modify existing GHRSST behaviors or signatures; only add the metocean proxy + catalog entry.
* If `fastmcp.client.MCPClient` does not natively support headers, add a small transport wrapper (aiohttp session with `User-Agent`) or set a global default header.
* Keep argument validation and error mapping consistent with the GHRSST proxy patterns (data → structured → text → raw fallback). 

---

## 11) Example E2E (tide)

**User:** `(121.5,25.0) 現在是漲潮還是退潮？何時滿潮？日出？`
**CLI → server:** includes `tz="Asia/Taipei"`, `query_time="2025-11-13T16:05:00+08:00"`.
**Classifier plan:** `decision="mcp"`, `tool="tide.forecast"`, args filled (lon/lat/date/query_time/tz).
**Server:** calls local proxy → upstream metocean; receives JSON with `state_now`, extremes, sun/moon.
**Answer (one line):**
`現在屬於漲潮，距滿潮約 3小時31分（滿潮 18:23，乾潮 12:09）。日出 06:16、日落 17:07；今日月相：盈凸月(78%).`

---

This Δ-spec keeps the GHRSST path intact, adds the **metocean** endpoint, **User-Agent** policy, a new **`tide.forecast`** tool, and the minimal router/CLI changes so users can ask tide/sun/moon questions in natural language and get immediate answers via MCP.
