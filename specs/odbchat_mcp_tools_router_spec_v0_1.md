# ODBchat: Single-Pass Classifier+Router Spec (for Codex)

**Date:** 2025-11-06 (Asia/Taipei)
**Owners:** ODBchat team (Server: `server/odbchat_mcp_server.py`; CLI: `cli/odbchat_cli.py`)
**Scope:** Route free-text user queries to `mcp | explain | code` **in one LLM call**, and (when `mcp`) extract normalized tool arguments (time, place → bbox/point, fields) to call GHRSST tools.

---

## 1) Problem Statement

Users ask: *“現在台灣周遭海溫多少？”* and similar. The system often:

* Falls into **RAG one-pass “code mode”** (emitting ad-hoc Python) instead of calling GHRSST MCP tools.
* Shows planner debug like *“Code may not reference plan param …”*
* Is **slow**, because free-text always goes through RAG’s planner/classifier, even when a simple MCP tool would suffice.

This happens despite GHRSST MCP tools being correctly exposed both upstream (`/mcp/ghrsst`) and locally via a proxy.

---

## 2) Root Cause

The **missing piece is an “intent → tool router.”**

* The current CLI sends all free-text to `rag.onepass_answer`.
* There is **no early decision** that a query is best answered via MCP tools (e.g., `ghrsst.bbox_mean`) and **no slot-filling** to derive `date`, `bbox`/`point`, `fields` from natural language.

---

## 3) Proposed Solution (Concept)

**Unify “router” and “classifier” into a single, fast LLM step**:

* Replace the 2-way classifier (`explain | code`) with a **3-way classifier**: `mcp | explain | code`.
* **If `mcp`**: in the **same LLM call**, require the model to select **exactly one** MCP tool (from a provided catalog) and return **normalized arguments** (date, bbox/point, fields).
* The server (not the LLM) **validates & normalizes** arguments, executes the MCP call, formats a concise answer, and returns it to clients.
* If not `mcp`, the output flows into the **existing one-pass** `explain` or `code` branches unchanged.

**Result:** one LLM hop per user query; correct tool usage; lower latency; cleaner behavior.

---

## 4) Implementation Plan (Steps, Skeleton, Examples)

### 4.1 Artifacts & Locations

* **New**: `server/router_classifier.py`

  * Function: `classify_and_route(query: str, debug: bool=False) -> dict`
  * Encapsulates the single-pass **classifier+router** logic using the local LLM (llama-cpp/ollama).
* **Server Registration**: `server/odbchat_mcp_server.py`

  * Register an MCP tool `router.answer` that calls `classify_and_route()`.
* **CLI Entry**: `cli/odbchat_cli.py`

  * Free-text chat should call `router.answer` (not `rag.onepass_answer`).

### 4.2 Interface Contracts

#### 4.2.1 Router Tool (exposed to clients)

**Tool name:** `router.answer`
**Input:**

```json
{
  "query": "string",
  "debug": false
}
```

**Output (uniform envelope):**

```json
{
  "mode": "tool|explain|code",
  "text": "User-visible final answer in the user's language",
  "source": "ghrsst.bbox_mean|ghrsst.point_value|rag.explain|rag.code",
  "confidence": 0.0,
  "raw": {
    "plan": { /* LLM JSON plan */ },
    "tool_result": { /* MCP result when mode=tool */ },
    "trace": { /* optional timing, retries, etc. */ }
  }
}
```

* `mode="tool"` means an MCP tool was executed successfully.
* `source` indicates the branch/tool used.
* `confidence` is the classifier confidence (propagated from the LLM plan or set by policy).

#### 4.2.2 Internal LLM Plan Schema (strict JSON)

The classifier must **only** output valid JSON like:

```json
{
  "decision": "mcp|explain|code",
  "confidence": 0.0,
  "tool": {
    "name": "ghrsst.bbox_mean",
    "arguments": {
      "bbox": [118.0, 20.0, 123.5, 26.5],
      "date": "2025-11-06",
      "fields": ["sst", "sst_anomaly"]
    }
  },
  "rationale": "short reason",
  "needs_followup": false,
  "followup_question": null
}
```

**Rules enforced by prompt & code:**

* If `decision="mcp"`, `tool.name` **must** be in `tool_catalog`.
* `arguments` keys **must** be a subset of the selected tool’s allowed schema.
* Time expressions are parsed relative to `now`, `tz`.
* Place names must map to `geo_registry` keys when possible; otherwise the model may propose a bbox (confidence may be lower).

### 4.3 Tool Catalog & Geo Registry (injected context)

**Tool catalog** (server-provided to LLM each call; minimal, task-oriented):

* `ghrsst.bbox_mean(bbox, date, fields?)`
* `ghrsst.point_value(longitude, latitude, date, fields?)`

**Geo registry** (YAML/Dict, server-side; examples):

```yaml
taiwan_nearby: [118.0, 20.0, 123.5, 26.5]
hawaiian_islands: [-162.5, 18.5, -153.5, 23.0]
taiwan_strait: [118.0, 21.5, 121.5, 26.0]
```

> Keep both lists small and curated to help the classifier be decisive and fast.

### 4.4 Server-Side Normalization & Fallback

Even if the LLM returns arguments, the server must **normalize/validate**:

* **Dates:** enforce `YYYY-MM-DD`. If absent, use **today** in `Asia/Taipei`.
  Policy (recommended): if `today` yields `NOT_FOUND`, auto-retry `today-1d`.
* **BBox/Point:** enforce order `(lon0<lon1, lat0<lat1)`, valid ranges, area > 0.
  If the LLM returned a registry key (string), replace with numeric bbox.
* **Fields:** default `["sst","sst_anomaly"]` when not provided.
* **Errors:** map upstream errors to standard codes `INVALID_ARGUMENT | NOT_FOUND | UNAVAILABLE` and render a compact user message.

### 4.5 Minimal Code Skeletons

#### 4.5.1 `server/router_classifier.py`

```python
# server/router_classifier.py
from typing import Dict, Any, List
from your_llm_client import chat_json  # wraps llama-cpp/ollama to enforce JSON output
from .utils.time import today_tpe, ensure_date, yyyymmdd_offset
from .utils.geo import GEO_REGISTRY, normalize_bbox
from .mcp_runtime import call_tool, call_onepass_explain, call_onepass_code, is_not_found

SYSTEM = """You are ODBchat's single-pass classifier+router.
Output ONLY valid JSON with keys: decision, confidence, tool?, rationale, needs_followup, followup_question.
Decide: 'mcp' | 'explain' | 'code'.
If 'mcp', pick EXACTLY ONE tool from tool_catalog and fill allowed arguments ONLY.
Parse time expressions relative to 'now' and 'tz'. Prefer place names in geo_registry.
"""

FEWSHOT = [
  # 1) Taiwan SST today -> bbox_mean (taiwan_nearby)
  # 2) 121.7E,24.0N today -> point_value
  # 3) Hawaiian Islands SST this week -> (v0: daily bbox_mean with date=now)
  # 4) "What is GHRSST?" -> explain
  # 5) "Write Python to ..." -> code
]

TOOL_CATALOG = [
  {"name":"ghrsst.bbox_mean",
   "args_schema":{"bbox":"[lon0,lat0,lon1,lat1]","date":"YYYY-MM-DD","fields?":"['sst','sst_anomaly']"},
   "description":"Daily mean SST/SSTA over a bbox."},
  {"name":"ghrsst.point_value",
   "args_schema":{"longitude":"float","latitude":"float","date":"YYYY-MM-DD","fields?":"['sst','sst_anomaly']"},
   "description":"Daily SST/SSTA at a point."}
]

DEFAULT_FIELDS = ["sst","sst_anomaly"]

def classify_and_route(query: str, debug: bool=False) -> Dict[str, Any]:
    now = today_tpe()  # YYYY-MM-DD (Asia/Taipei)
    plan = chat_json(
        {
            "system": SYSTEM,
            "fewshot": FEWSHOT,
            "user": {
                "query": query,
                "now": now,
                "tz": "Asia/Taipei",
                "tool_catalog": TOOL_CATALOG,
                "geo_registry": GEO_REGISTRY
            }
        },
        temperature=0, max_tokens=256
    )

    decision = plan.get("decision", "explain")
    if decision == "mcp":
        tool = plan["tool"]["name"]
        args = dict(plan["tool"]["arguments"])

        if tool.endswith("bbox_mean"):
            bbox = args.get("bbox")
            # allow registry key or numeric array
            if isinstance(bbox, str) and bbox in GEO_REGISTRY:
                bbox = GEO_REGISTRY[bbox]
            bbox = normalize_bbox(bbox)
            date = ensure_date(args.get("date"), now)
            fields = args.get("fields") or DEFAULT_FIELDS

            res = call_tool(tool, {"bbox": bbox, "date": date, "fields": fields})
            if is_not_found(res) and date == now:
                res = call_tool(tool, {"bbox": bbox, "date": yyyymmdd_offset(now, -1), "fields": fields})

            text = render_bbox_mean(res)  # implement concise °C message with date & bbox hint
            return wrap("tool", text, "ghrsst.bbox_mean", plan, res, debug)

        elif tool.endswith("point_value"):
            lon, lat = float(args["longitude"]), float(args["latitude"])
            date = ensure_date(args.get("date"), now)
            fields = args.get("fields") or DEFAULT_FIELDS

            res = call_tool(tool, {"longitude": lon, "latitude": lat, "date": date, "fields": fields})
            if is_not_found(res) and date == now:
                res = call_tool(tool, {"longitude": lon, "latitude": lat, "date": yyyymmdd_offset(now, -1), "fields": fields})

            text = render_point(res)
            return wrap("tool", text, "ghrsst.point_value", plan, res, debug)

        # unknown tool → degrade to explain
        res = call_onepass_explain(query)
        return wrap("explain", res["text"], "rag.explain", plan, res, debug)

    elif decision == "code":
        res = call_onepass_code(query)
        return wrap("code", res["text"], "rag.code", plan, res, debug)

    else:
        res = call_onepass_explain(query)
        return wrap("explain", res["text"], "rag.explain", plan, res, debug)


def render_bbox_mean(res: Dict[str, Any]) -> str:
    date = res.get("date", "")
    sst = res.get("sst")
    anom = res.get("sst_anomaly")
    if sst is None and anom is None:
        return f"No GHRSST data for {date} in the requested region."
    parts = []
    if isinstance(sst, (int, float)):
        parts.append(f"SST ≈ {sst:.1f} °C")
    if isinstance(anom, (int, float)):
        sign = "+" if anom >= 0 else ""
        parts.append(f"anomaly {sign}{anom:.2f} °C")
    return f"{date}: " + ", ".join(parts)

def render_point(res: Dict[str, Any]) -> str:
    return render_bbox_mean(res)

def wrap(mode, text, source, plan, tool_result, debug):
    out = {"mode": mode, "text": text, "source": source, "confidence": plan.get("confidence", 0.0)}
    raw = {"plan": plan}
    if tool_result is not None:
        raw["tool_result"] = tool_result
    if debug:
        raw["trace"] = {"debug": True}
    out["raw"] = raw
    return out
```

#### 4.5.2 Register the Router Tool

In `server/odbchat_mcp_server.py`, register an MCP tool:

```python
# inside server startup
from .router_classifier import classify_and_route

@mcp.tool(
  name="router.answer",
  description="Single-pass classifier+router: decide mcp|explain|code; when mcp, select tool and args; execute and render."
)
def router_answer(query: str, debug: bool = False):
    return classify_and_route(query, debug=debug)
```

#### 4.5.3 CLI Entry Change

In `cli/odbchat_cli.py`, route free-text to `router.answer`:

```diff
- result = await self.client.call_tool("rag.onepass_answer", {"query": text})
+ result = await self.client.call_tool("router.answer", {"query": text})
```

CLI prints `result["text"]`. If `--debug`, also display `result["raw"]["plan"]` & `["raw"]["tool_result"]`.

### 4.6 Acceptance Criteria

* **AC-1:** Prompt “現在台灣周遭海溫多少？” calls **`ghrsst.bbox_mean`** (visible in server logs), returns a one-line °C answer with the **used date** (today or yesterday).
* **AC-2:** Prompt “花蓮外海 121.7E,24.0N 今天呢？” calls **`ghrsst.point_value`**.
* **AC-3:** Prompt “GHRSST 是什麼？” follows **`explain`** branch (RAG).
* **AC-4:** Prompt “幫我把以下 JSON 轉 CSV 的 Python” follows **`code`** branch.
* **AC-5:** No extra LLM hops vs current classifier; total latency **≤** previous one-pass path on average.
* **AC-6:** Router output is always valid JSON; bad arguments are normalized server-side; errors map to `INVALID_ARGUMENT | NOT_FOUND | UNAVAILABLE`.

### 4.7 Observability & Logs

* Log: decision (`mcp|explain|code`), `tool.name`, normalized args, durations:
  `t_classify_ms`, `t_mcp_ms`, `t_total_ms`, retry flags (today→yesterday).
* Optional: sample structured logs to console + file.

### 4.8 Performance Notes

* Use the **same** local model for the classifier step (e.g., Gemma-3 4B/7B).
* `temperature=0`, `max_tokens≤256`, minimal few-shots (3–5), **small tool catalog**.
* Optionally enable **response cache** keyed by a minhash of the query → plan → result.

### 4.9 Error Handling

* MCP upstream timeout → `UNAVAILABLE` with a short, user-safe message.
* Empty region/day → `NOT_FOUND`, plus fallback to D-1 for “today”.
* Invalid bbox/latlon → `INVALID_ARGUMENT` with a one-line hint.

### 4.10 Future Extensions (v0.2+)

* Add `ghrsst.bbox_mean_timeseries(bbox, start, end, stat)` for “this week/last 3 days/this month”.
* Expand `geo_registry` and add a light **place-name resolver**.
* Per-tool guardrails (schema validators) and automatic unit annotations.

---

## 5) Work Breakdown (for Codex)

1. **Create** `server/router_classifier.py` (skeleton above).
2. **Register** MCP tool `router.answer` in `server/odbchat_mcp_server.py`.
3. **Switch** CLI free-text to call `router.answer`.
4. **Implement** helpers: `utils.time.ensure_date`, `utils.geo.normalize_bbox`, `mcp_runtime.call_tool/is_not_found`, simple renderers.
5. **Smoke tests** for AC-1…AC-4; measure latency before/after.
6. **Docs**: update README with the new entrypoint and example prompts.

---

## 6) Example End-to-End

**User:** `現在台灣周遭海溫多少？`
**Router plan (LLM):**

```json
{
  "decision": "mcp",
  "confidence": 0.83,
  "tool": {
    "name": "ghrsst.bbox_mean",
    "arguments": {
      "bbox": "taiwan_nearby",
      "date": "2025-11-06",
      "fields": ["sst","sst_anomaly"]
    }
  },
  "rationale": "SST question near Taiwan for today.",
  "needs_followup": false,
  "followup_question": null
}
```

**Server normalization & execution:**

* Resolve `bbox="taiwan_nearby"` → `[118.0, 20.0, 123.5, 26.5]`
* Call MCP tool; if `NOT_FOUND` for today, retry yesterday.
* Render: `2025-11-06: SST ≈ 26.4 °C, anomaly +0.21 °C.`

**CLI output:**

```
2025-11-06：台灣周遭海溫約 26.4 °C，距平 +0.21 °C。
```

---

**This spec is designed to be implemented with minimal diff to current code while removing the core failure mode (no intent→tool routing) and without adding an extra LLM hop.**
