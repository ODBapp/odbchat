# ODBchat Router + Metocean MCP Spec (v0.3.7)

**Date:** 2025-11-14 (Asia/Taipei)  
**Scope:** Unified contract for the single-pass router, GHRSST proxying, and metocean (`tide.forecast`) integration. This replaces both `odbchat_mcp_tools_router_spec_v0_1.md` and `odbchat_mcp_tools_metocean_integration_spec_v0_2.md`.

---

## 1. Goals & Non-Goals

| Goal | Description |
| --- | --- |
| Router-first UX | Every free-text prompt runs through `router.answer`, which must decide `mcp_tools | explain | code | fallback` in one pass and extract normalized MCP arguments when applicable. |
| Deterministic MCP routing | GHRSST and tide calls execute even if the LLM mislabels the mode—our heuristics parse coordinates/time hints and override the decision. |
| Metocean integration | GHRSST + tide share the upstream `https://eco.odb.ntu.edu.tw/mcp/metocean` surface, UA `metocean-mcp`. New tide rendering handles zh/en, sun/moon, inland points, and future/past dates. |
| CLI alignment | CLI always supplies `tz` + `query_time`, buffers multi-line paste, and relays router responses verbatim (no translation on the client). |

Non-goals: adding new metocean tools beyond `tide.forecast`, supporting wave height/sea-state data, or implementing multi-step agents.

---

## 2. Tool Catalog & Upstream Surface

| Tool | Inputs | Notes |
| --- | --- | --- |
| `ghrsst.point_value` | `{longitude, latitude, date?, fields?, method?}` | Proxy to `/mcp/metocean`. `fields` defaults to `["sst","sst_anomaly"]`. `method` is `"exact"` unless auto-switched to `"nearest"`. |
| `ghrsst.bbox_mean` | `{bbox[4], date, fields?, method?}` | Same endpoint/UA as above. BBox order `[lon0, lat0, lon1, lat1]`. |
| `tide.forecast` | `{longitude?, latitude?, station_id?, date?, query_time?, tz?}` | `include_sun_moon` is no longer accepted (sun/moon returned automatically). Either lon/lat or station_id is required. |

Proxy modules (`server/api/ghrsst_mcp_proxy.py`, `server/api/metocean_mcp_proxy.py`) must:

- Use `StreamableHttpTransport` with UA header `metocean-mcp`.
- Preserve upstream JSON when `result.data` or `structured_content` exists.
- Fall back to concatenated text or return the upstream envelope when the payload is opaque.
- Raise `ToolError` with a helpful message (`Failed upstream metocean tool 'tide.forecast': …`).

---

## 3. Router Contract (`router.answer`)

### 3.1 Input

```json
{
  "query": "string (required)",
  "tz": "Asia/Taipei (optional — CLI always sets)",
  "query_time": "2025-11-14T11:00:00+08:00 (optional)",
  "debug": false
}
```

`tz` is normalized to an IANA name; `query_time` defaults to `now_iso_in_tz(tz)` when omitted.

### 3.2 Output Envelope

```json
{
  "mode": "mcp_tools|explain|code|fallback",
  "text": "…",               // optional when mode=code (then `code` is present)
  "code": "…",               // optional
  "plan": {...},             // optional normalized plan
  "citations": [ { "title": "...", "source": "...", "chunk_id": 0 } ],
  "tool": "ghrsst.point_value|tide.forecast",   // only when mode=mcp_tools
  "arguments": {...},        // normalized args passed to MCP tool
  "result": {...},           // upstream JSON when tool succeeded
  "error": "string",         // when tool failed
  "debug": {...}             // present when debug=true
}
```

### 3.3 Decision Flow

1. **LLM classification:** Router calls `run_onepass(query, today, tz, query_time)`; result contains `mode`, `text`, `code`, `plan`, `mcp`, `debug`.
2. **Parse heuristics independent of LLM decision:**
   - `_contains_code_request`: checks regex for `code|Python|畫圖|plot…` but respects `/不用程式/CLI tool` and CLI mentions so legitimate doc questions don’t get misrouted.
   - `_is_sea_state_only_query`: matches `海況|sea state` and short-circuits with a static explain response telling the user only SST/tide data are available.
   - `_looks_like_tide_query`: regex covers `潮`, `滿潮`, `乾潮`, `日出`, `日落`, `夕陽`, `曙光`, `暮光`, `月出`, `月落`, `moon rise/set`, `sunrise/set`, `tide`, `tidal`. `_query_requests_tide_details` is stricter (needs tide-specific words, not just sun/moon).
   - `_looks_like_ghrsst_query`: matches SST keywords + either coordinates/bbox or both time hints and digits.
3. **Auto MCP:**
   - When `mode=="mcp_tools"` but `mcp` block missing, router tries `_infer_tide_decision_from_query` then `_infer_ghrsst_decision_from_query`.
   - Even if LLM mode is `code/explain`, `needs_tide` or `gh_request` triggers the same inference logic and immediate MPC execution.
4. **MCP execution:** `_execute_mcp_tool` handles normalization, retries nearest, builds text strings, or returns error payloads.
5. **Fallback:** If no heuristic triggers, router returns the one-pass payload untouched (text/code/citations/plan/debug).

---

## 4. GHRSST Rules

- **Point vs BBox:** `_extract_point_from_query` parses `(lon,lat)` or `lon,lat` with optional NSEW suffix. `_extract_bbox_from_query` recognizes `[lon0, lat0, lon1, lat1]` ranges. The first match determines the tool.
- **Date defaults:** `date` default is `today_in_tz(tz)`. For bbox and point we store the requested date on the response so the answer can show `（原請求 YYYY-MM-DD）` when we had to fall back to an earlier date.
- **Fields/method:** `fields` defaults to `["sst","sst_anomaly"]`. `method` defaults to `"exact"` unless:
  1. User explicitly requested `"nearest"`, OR
  2. Query references “今天/今日/現在/current/today” and requested date == today, OR
  3. Upstream returned “data not exist / no data / available range” and we retry with `nearest`.
- **Text format:** `YYYY-MM-DD｜point [lon, lat]: SST ≈ 27.71 °C, Anomaly +1.09 °C（原請求 2025-11-11）` for points; `bbox […]` for ranges. Citations stay empty because upstream data is real-time.

---

## 5. Tide Rendering & Language Rules

### 5.1 Modes

| Mode | Trigger | Output |
| --- | --- | --- |
| **Relative** | `query_date == query_time.date()` | Include `state_now`, `Next high/low tide` sentences with durations (zh: `約 6小時31分 後`; en: `about 4h32m later`), `Previous … (已過 … / elapsed …)`, sun/moon segments, and a multi-line tide list. |
| **Summary** | `query_date != query_time.date()` | Omit state/relative text; render `滿潮 … 乾潮 …` sentences plus `資料日期 YYYY-MM-DD`/`Date YYYY-MM-DD`. |
| **Astro-only** | Query mentions sun/moon but not tide keywords | Still call `tide.forecast`, but show only sun/moon segments (and moon phase) and skip tide list + notes. |

### 5.2 Formatting

- **Language detection:** `_is_zh_query` (presence of any CJK). zh replies use `、`/`。` joiners; en uses `, ` and `. `, durations as `6小時31分` vs `6h31m`.
- **Sun/Moon segments:** Always include Civil dawn/dusk if provided. Format:
  - zh: `曙光 05:45、日出 06:08、日落 17:13、暮光 17:36。月出 00:53、月落 13:38、今日月相：殘月(月盈:31%)`
  - en: `Civil dawn 05:45, Sunrise 06:08, Sunset 17:13, Civil dusk 17:36. Moonrise … Moon phase: Waning Crescent (Illumination: 31%).`
- **Tide list:** Always appended on a new line when tide data is shown. zh prefix `潮位資訊：`, en `Tide list:`. Format `滿潮 06:08 高度62 cm、18:51 高度85 cm；乾潮 …` or `High tide 06:08 height 62 cm, …; Low tide …`.
- **Notes:** After the tide list (or after astro text if no tide list). Translate `Tide heights → 潮高`, `MSL → 平均海水面`. If `meta.status` contains upstream warnings, append them as well.
- **No tide data:** When `show_tide` and highs/lows absent, append `此點位並無潮汐資料。` / `No tide data is available for this location.` but still show sun/moon segments if available.
- **Inland astro-only:** If user asked only for sunrise/sunset at inland coordinates, omit tide warning entirely to avoid confusing users.

### 5.3 Translation Map

| English | zh-TW |
| --- | --- |
| Tide heights | 潮高 |
| MSL | 平均海水面 |
| Moon phases | `Waning Crescent → 殘月`, `Third Quarter → 下弦月`, `Waning Gibbous → 虧凸月`, `Full Moon → 滿月`, `Waxing Gibbous → 盈凸月`, `First Quarter → 上弦月`, `Waxing Crescent → 眉月`, `New Moon → 新月`. |
| Fraction Illumination | 月盈 |

---

## 6. CLI Requirements

1. **Router payload:** Always include `tz` (`tzlocal` fallback to `Asia/Taipei`) and `query_time` (ISO-8601 with offset).
2. **Multi-line input:** Buffer pasted blocks; submit after an empty newline.
3. **Rendering:** Print router `text` verbatim. When `mode="mcp_tools"` and `tool` exists, show `tool`/`arguments` in debug mode; otherwise just show human text. Do not restack translations.
4. **Error messaging:** On `error` field from router, show `目前無法取得資料：…` string and keep `tool/arguments` details behind `--debug`.

---

## 7. Error Handling & Logging

- All ToolErrors should bubble up through router as `mode="mcp_tools"`, `tool`, `arguments`, `error`, `text`.
- Router logs (INFO level):
  - `mode`, `needs_tide`, `gh_request`, `tool`, `error`.
  - Arguments sanitized to 2 decimal places for lon/lat.
  - Timing buckets: `t_classify_ms`, `t_mcp_ms`, `t_total_ms`.
- CLI logs SSE/HTTP failures (e.g., `HTTP 400` when upstream rejects) and suggests `/llm status`.

---

## 8. Acceptance Tests (Minimum)

| ID | Scenario | Expectation |
| --- | --- | --- |
| T1 | `現在台灣周遭(123,25)海溫多少？` | Router auto GHRSST point → `mode=mcp_tools`, `tool=ghrsst.point_value`, `text` with SST/Anomaly, `method="nearest"` if data missing. |
| T2 | `現在台灣周遭[118,20,123,25]海溫？` | Router auto GHRSST bbox. |
| T3 | `(123,37)今天何時滿潮? 是滿月嗎?` | Router auto tide relative text (state, next/prev, tide list, moon translation, note last). |
| T4 | `太麻里(22.55N, 120.95E)日出時間？` | Router still calls tide tool but returns only sun/moon (with Civil dawn/dusk) because query lacked tide keywords. No tide warning. |
| T5 | `Moon rise at (121.0045, 22.475)?` | English query still routes to tide; response in English, durations `h/m`, moon phrases in English. |
| T6 | `2025/11/25 座標約為 25.2079, 121.4286 看夕陽和潮汐？` | Summary mode (date differs), lists highs/lows with `資料日期 2025-11-25`, no “currently rising tide”. |
| T7 | `(121.5,25.0) 海況？` | Router returns explain text “目前僅提供海表溫度與潮汐資訊…”. |
| T8 | Inland tide query with explicit tide keyword | Response includes sun/moon plus `此點位並無潮汐資料。`. |
| T9 | CLI multi-line paste | Input of multiple lines is sent as one prompt (no premature submission). |
| T10 | Router `mcp` block missing (LLM bug) | Heuristics infer tool and still answer. Logged error `mcp mode missing tool specification`. |

Each test must have a pytest counterpart (see `tests/test_router_classifier.py` and `tests/test_cli_onepass.py`).

---

This spec governs the router heuristics, MCP argument normalization, tide rendering, and CLI obligations for ODBchat v0.3.7. All future changes to GHRSST or metocean routing must update this file and `specs/mcp.tools.schema.json` before implementation.
