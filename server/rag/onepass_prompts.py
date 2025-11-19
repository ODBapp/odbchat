from __future__ import annotations
import json, re
from typing import Dict, Any

ECO_BASE_URL = "https://eco.odb.ntu.edu.tw"

def _compact_whitelist(oas: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "paths": list(oas.get("paths") or []),
        "params": list(oas.get("params") or []),
        "append_allowed": list(oas.get("append_allowed") or oas.get("append") or []),
        "param_enums": oas.get("param_enums") or {},
        "servers": list(oas.get("servers") or []),
    }

def _sysrule_oas_whitelist(oas: Dict[str, Any]) -> str:
    o = _compact_whitelist(oas)
    enums_pretty = []
    for k, v in (o["param_enums"].items() if isinstance(o["param_enums"], dict) else []):
        if isinstance(v, (list, tuple)) and v:
            enums_pretty.append(f"  - {k} ∈ {{{', '.join(map(str, v))}}}")
    enums_line = "\n".join(enums_pretty) if enums_pretty else "(none)"
    return (
        "OAS whitelist (do NOT invent endpoints/params):\n"
        f"- endpoints: {', '.join(o['paths']) or '(none)'}\n"
        f"- params: {', '.join(o['params']) or '(none)'}\n"
        f"- append allowed: {', '.join(o['append_allowed']) or '(none)'}\n"
        f"- enums:\n{enums_line}\n"
    )

def _sysrule_classifier() -> str:
    return (
        "MODE classifier:\n"
        "- Decide one token: 'code', 'explain', 'fallback', or 'mcp_tools'.\n"
        "- Output 'mcp_tools' when the user explicitly wants actual SST numbers (單點或經緯度範圍的海溫) or tidal conditions at a particular date or uses words with time connotations such as now, recently, or today, etc.\n"
        "- If the question contains negative phrases like '不要程式', '不用 API', '除了程式以外', to ask for explanations, methods or tools but exclude using code, treat it as 'explain'.\n"
        "- If the user asks how to use a GUI/CLI tool (e.g., Hidy Viewer, ODBchat CLI) without requesting new code, treat it as 'explain'.\n"
        "- Output 'explain' for questions that ask to 描述/定義/比較/列舉/探索資訊、GUI/工具/平台、資料來源、可視化需求等。\n"
        "- Output 'code' when the user explicitly requests a script/programmatic step (e.g., '寫程式', '如何用程式…', 'Python code', 'API 範例', '畫圖', '時序圖/分析') OR clearly references modifying/continuing the previous code (phrases like '改變/畫', '更新/換', '再試一次', '上一個結果加上…', explicit parameter changes, etc.). If details (bbox/date) are missing, still choose MODE=code and make reasonable assumptions in the plan.\n"
        "- Output 'fallback' when the input is nonsense/typo, pure pleasantry (e.g., '謝謝', 'bye'), or otherwise lacks a clear technical request. Fallback replies should be short clarifications with no citations.\n"
        "- When uncertain between code or explain, prefer 'explain'.\n"
    )

def _sysrule_planner(oas_info: Dict[str, Any]) -> str:
    o = _compact_whitelist(oas_info)
    allowed_paths  = ", ".join(o["paths"]) or "(none)"
    allowed_params = ", ".join(o["params"]) or "(none)"
    allowed_append = ", ".join(o["append_allowed"]) or "(none)"
    append_usage   = f"Allowed values: {allowed_append}" if allowed_append != "(none)" else "Not allowed"
    return (
        "API planner rules:\n"
        "- Never invent endpoints or params; use ONLY the OAS whitelist.\n"
        "- If a previous plan is provided, reuse its endpoint/params unless the new question explicitly asks for a change (only adjust the fields mentioned by the user).\n"
        f"(1) Allowed endpoints: {allowed_paths}\n"
        f"(2) Allowed query params: {allowed_params}\n"
        f"(3) 'append' param {append_usage}\n\n"
        "- Use ONLY param names from the whitelist; do not invent aggregated params like 'bbox' or 'bounding_box'.\n"
        "- If PLAN has both 'start' and 'end' whitelisted, include both (never leave 'end' empty).\n\n"
        "- When the user omits spatial/temporal details, infer a sensible bbox/time window (document it in PLAN) instead of asking follow-up questions.\n"
        "CSV / JSON selection:\n"
        "- Prefer JSON if the user didn't explicitly ask CSV/file download.\n"
        "- For CSV (only when user requests to CSV/file download): choose '/csv' endpoint if present; otherwise ONLY if 'format' enum includes 'csv', set format='csv'.\n\n"
        "Spatial-related params guidance:\n"
        "- If region/place implied, estimate a relevant bbox or lon/lat (use ONLY whitelisted names like lon0,lon1,lat0,lat1 or lon,lat).\n"
        "- Never global extent by default.\n"
        "- Never cross antimeridian/0° in one box; if needed, plan split.\n\n"
        "Temporal params guidance (whitelisted names like start/end):\n"
        "- If a year implied → a full-year range; if month/day implied → tight window.\n"
        "- Always include BOTH 'start' and 'end' when allowed, covering the intended range.\n"
        "- Use ISO dates (YYYY-MM-DD).\n\n"
        "Param 'append' guidance:\n"
        f"- Allowed values: {allowed_append}\n"
        "- Infer variables from intent (e.g., '海溫'/'SST' → 'sst'; '距平'/'anomaly' → 'sst_anomaly'; '海洋熱浪'/'heatwave' without extra qualifiers → 'level'). Always set 'level' when the user just says “海洋熱浪分佈/heatwave map” (unless they explicitly ask for anomaly/SST instead).\n"
        "- If analysis/plotting requires variables and 'append' is allowed, DO NOT leave it empty.\n\n"
        "MHW chunking constraints (decide here for CODE to implement):\n"
        "- Pick one 'chunk_rule' for MHW endpoints:\n"
        "  • 'monthly' (bbox>90°×90°) • 'yearly' (bbox>10°×10°) • 'decade' (bbox≤10°×10° & span>10y) • '' (no chunk)\n\n"
        "Plotting decision (set 'plot_rule' for CODE to follow):\n"
        "  • 時序/趨勢→'timeseries'；地圖/分佈→'map'；否則 'none'.\n\n"
        "If usr intent to follow up or revise code from previous question, do NOT remove constraints/params in previous plan. Inherit values, or update them if the new question explicitly asks for altering values.\n"
        'Return JSON only: { "endpoint": "/<path>", "params": {"<k>":"<v>"}, "chunk_rule": "<monthly|yearly|decade|>", "plot_rule": "<timeseries|map|none|>" }\n'
    )

def _sysrule_code_assistant(oas_info: Dict[str, Any]) -> str:
    servers = oas_info.get("servers") or []
    base_hint = servers[0] if servers else ECO_BASE_URL
    return (
        "Your are a Python Coder. Rules:\n"
        "- Generate a single runnable Python script (only code, no prose).\n"
        "- Obey PLAN JSON exactly: EXACT endpoint and params (NO inventions). Do NOT send 'chunk_rule'/'plot_rule'/'bbox' as params.\n"
        f"- Determine BASE_URL from OAS servers (use '{base_hint}' unless PLAN specifies otherwise) and call requests.get(BASE_URL + endpoint, params=params, timeout=60).\n"
        "- NEVER handcraft query strings; start from PLAN['params'], copy into a dict, and pass it via the 'params' argument for every request.\n"
        "- Do NOT introduce authentication tokens or api_key parameters unless the PLAN explicitly contains them.\n"
        "- Read 'chunk_rule' from PLAN JSON and implement the required time-splitting loop to fetch API:\n"
        "    • 'monthly' → loop month by month between PLAN start/end (inclusive).\n"
        "    • 'yearly'  → loop year by year; each request must stay within a single calendar year.\n"
        "    • 'decade'  → request in ≤10-year windows (fallback to yearly for remainders <10).\n"
        "    • ''        → single request (no chunking).\n"
        "  Split ONLY via start/end dates; keep spatial params fixed; concat results at the end (pd.concat).\n"
        "  Respect PLAN['params']['start']/'end' when present; compute per-segment start/end strings and update params per request.\n"
        "  Required pattern: base_params = PLAN['params'].copy(); iterate periods, set params = base_params.copy(); params['start']=segment_start; params['end']=segment_end; requests.get(..., params=params, timeout=60); collect each chunk and pandas.concat at the end.\n"
        "- JSON: assign raw = r.json(); ALWAYS convert via pd.DataFrame(raw) before any filtering/plotting (never treat the raw JSON as the DataFrame). CSV: only for '/csv' or enum format='csv' using pandas.read_csv(io.StringIO(r.text)).\n"
        "- If bbox crosses 180°/0° split requests and concat.\n"
        "- If 'date' present → to_datetime and sort chronologically.\n"
        "- Honor PLAN['plot_rule']: 'timeseries' → use plt.figure(), plt.plot(df['date'], ...), label axes/title/legend, and plt.show(); 'map' → build a 2D grid: pivot_table or reshaping so that Z has shape (len(lat_unique), len(lon_unique)), create Lon, Lat via numpy.meshgrid, then plt.pcolormesh(Lon, Lat, Z, shading='auto') with colorbar and plt.show(); missing/empty → skip plotting.\n"
        "- For map plots: do NOT call pcolormesh directly on raw columns; you must pivot to a grid first and ensure lon/lat are sorted unique values.\n"
        "- Do NOT prefix blocks with language labels or markdown fences. Use ONLY the <<<PLAN>>> and <<<CODE>>> tags with raw JSON/code content.\n"
        "- Return the script in a code fence. CODE cannot be empty.\n"
    )

def _sysrule_mcp(today: str, tz: str, query_time: str) -> str:
    return (
        "MCP tool rules:\n"
        "GHRSST (SST):\n"
        "- Use 'ghrsst.point_value' for single lon/lat points; 'ghrsst.bbox_mean' for lon/lat ranges [lon0,lat0,lon1,lat1]. All floats in decimal degrees.\n"
        "- bbox MUST be a JSON array of exactly four numbers; do NOT wrap it in a string.\n"
        "- If 'date' missing, set date='{today}'. Default 'fields'=['sst','sst_anomaly']. Use method='nearest' for 今日/現在/最新, else 'exact'.\n"
        "\n"
        "Tide/Sun/Moon:\n"
        "- Use 'tide.forecast' for any question mentioning tides, sun/moon rise/set, or moon phase.\n"
        "- Always include 'tz' (IANA, e.g., '{tz}') AND 'query_time' (ISO-8601, e.g., '{query_time}') so the tool can compute 'state_now', 'since_extreme', 'until_extreme'.\n"
        "- Provide location via 'longitude'+'latitude' (floats) or 'station_id'. Set 'date' if user asks for a different day (default '{today}').\n"
        "\n"
        "- Output a <<<MCP>>> JSON block: {\"tool\":\"ghrsst.point_value|ghrsst.bbox_mean|tide.forecast\",\"arguments\":{...},\"confidence\":0.xx,\"rationale\":\"...\"}."
    )

def _sysrule_explain(force_zh: bool) -> str:
    lang = "- 回答請使用繁體中文。\n" if force_zh else "- Answer in the user's language.\n"
    return (
        "You are an ODB assistant. Provide a concise explanation that answers the question.\n"
        + lang +
        "- Use ONLY provided notes for specific resources; no code.\n"
        "- Do NOT include Python or any code fences when MODE=explain; respond with plain sentences/bullets only.\n"
        "- When the question asks about GUI/非程式工具, explicitly mention every relevant interactive option found in the notes (e.g., Hidy Viewer, ODBchat CLI) with a short description and URL. Never claim a tool is unavailable if the notes describe one.\n"
        "- Keep it short but complete.\n"
    )

def _fmt_user_template(tmpl: str, *, query: str, top_k: int) -> str:
    # support both {top_k} and {topK}
    return tmpl.format(query=query, top_k=top_k, topK=top_k)

def build_main_prompt(
    *,
    query: str,
    notes: str,
    whitelist: Dict[str, Any],
    top_k: int = 6,
    prev_plan: Dict[str, Any] | None = None,
    followup_hint: bool = False,
    today: str,
    tz: str,
    query_time: str,
    ghrsst_hint: bool = False,
    tide_hint: bool = False,
) -> str:
    strict_format = (
        "ABSOLUTE OUTPUT RULES:\n"
        "- Your FIRST characters MUST be exactly: '<<<MODE>>>' (no preface, no commentary).\n"
        "- NEVER output only a mode token. You MUST output ALL required blocks for the chosen MODE.\n"
        "If MODE=code (for STEP 2A and STEP 3):\n"
        "- You MUST output BOTH a <<<PLAN>>> JSON block and a <<<CODE>>> block, in this order.\n"
        "- If MODE=code and you do not include a non-empty <<<CODE>>> block, the output is INVALID.\n"
        "If MODE=explain:\n"
        "- Put the answer in <<<ANSWER>>> only (no PLAN/CODE there).\n"
        "If MODE=fallback:\n"
        "- Do NOT output PLAN/CODE; give a short clarification inside <<<ANSWER>>> only.\n"
        "- Never repeat PLAN or CODE contents outside their tagged blocks.\n"
        "- NEVER wrap any block in markdown fences (no ```json or ```python). Output the tags directly.\n"
        "If MODE=mcp_tools:\n"
        "- Do NOT output PLAN/CODE. Instead output a <<<MCP>>> JSON block containing tool name, arguments, and optional confidence/rationale.\n"
        "- The tool must be one of ghrsst.point_value, ghrsst.bbox_mean, or tide.forecast.\n"
        "\n"
        "STRICT OUTPUT FORMAT (use these exact tagged blocks):\n"
        "<<<MODE>>>{code|explain|fallback|mcp_tools}<<<END>>>\n"
        "<<<PLAN>>>{ \"endpoint\": \"/<path>\", \"params\": {\"<param>\": \"<value>\"}, \"chunk_rule\": \"<monthly|yearly|decade or empty>\", \"plot_rule\": \"<timeseries|map or empty>\"}<<<END>>>  # ONLY when MODE=code\n"
        "<<<CODE>>>\n<single runnable python script>\n<<<END>>>  # ONLY when MODE=code\n"
        "<<<MCP>>>\n{\"tool\":\"ghrsst.point_value|ghrsst.bbox_mean|tide.forecast\",\"arguments\":{...},\"confidence\":0.xx,\"rationale\":\"...\"}\n<<<END>>>  # ONLY when MODE=mcp_tools\n"
        "<<<ANSWER>>>\n<text answer>\n<<<END>>>\n"
    )

    classifier = _sysrule_classifier()
    planner    = _sysrule_planner(whitelist)
    coder      = _sysrule_code_assistant(whitelist)
    force_zh   = bool(re.search(r"[\u4e00-\u9fff]", query))
    explainer  = _sysrule_explain(force_zh)
    mcp_rules  = _sysrule_mcp(today, tz, query_time)
    user_tmpl  = "{query}"
    user       = _fmt_user_template(user_tmpl, query=query, top_k=top_k)
    prev_plan_blob = json.dumps(prev_plan, ensure_ascii=False) if prev_plan else "(none)"

    ghrsst_line = "yes" if ghrsst_hint else "no"
    tide_line = "yes" if tide_hint else "no"

    return (
        "You are an ODB assistant that does Classifier, Planner, Coder, and Explainer in ONE pass.\n"
        "Think step-by-step INTERNALLY, but NEVER print your reasoning.\n"
        "Do NOT ask the user for clarifications; make reasonable assumptions and proceed.\n"
        "Only print the tagged blocks defined in STRICT OUTPUT FORMAT.\n"
        "STEP 1 — " + classifier + "\n"
        "IF MODE=code THEN STEP 2A (Planner) — " + planner + "\n"
        "THEN STEP 3 (Coder) — " + coder + "\n"
        "ELSE IF MODE=explain THEN STEP 2B (Explainer) — " + explainer + "\n"
        "ELSE IF MODE=fallback THEN STEP 2C (Fallback) — give a short clarification (no PLAN/CODE, no citations).\n"
        "ELSE IF MODE=mcp_tools THEN STEP 2D (GHRSST MCP) — " + mcp_rules + "\n"
        + strict_format + "\n"
        "---- CONTEXT ----\n"
        f"RAG NOTES:\n{(notes or '').strip()}\n\n"
        f"TODAY (UTC+8): {today}\nCURRENT TZ: {tz}\nCURRENT QUERY TIME: {query_time}\n\n"
        f"PREVIOUS PLAN (inherit unless user explicitly changes values):\n{prev_plan_blob}\n\n"
        f"FOLLOW-UP HINT (based on user phrasing): {'yes' if followup_hint else 'no'}\n"
        "If hint=yes and a prior plan exists, prefer MODE=code and update only params explicitly mentioned in the new query.\n\n"
        f"GHRSST SST-REQUEST HINT: {ghrsst_line}\n"
        f"TIDE REQUEST HINT: {tide_line}\n"
        "If a hint is 'yes', default to MODE=mcp_tools unless the user explicitly requests code/program instructions.\n\n"
        f"OAS whitelist:\n{json.dumps(_compact_whitelist(whitelist), ensure_ascii=False)}\n\n"
        "EXAMPLES:\n"
        "1) Q: (121.5,25.0) 現在是漲潮嗎？何時滿潮？\n"
        "   → MODE=mcp_tools, <<<MCP>>> tool='tide.forecast', args include lon/lat, date=today, tz, query_time.\n"
        "2) Q: 今天台北日出日落？月相？\n"
        "   → MODE=mcp_tools, tide.forecast with lon/lat (or station), tz+query_time.\n"
        "3) Q: GHRSST 是什麼？\n"
        "   → MODE=explain with citations.\n\n"
        f"QUESTION:\n{user}\n"
    )

def build_continue_prompt(prev_raw: str, prev_code: str, query: str, whitelist: Dict[str, Any]) -> str:
    coder = _sysrule_code_assistant(whitelist)
    return (
        "You previously started answering but the code was incomplete. "
        "Continue ONLY the Python code. Do not repeat earlier lines. Keep the same constraints.\n"
        f"{_sysrule_oas_whitelist(whitelist)}\n"
        f"{coder}\n"
        f"<<<USER>>>\n{query}\n"
        "Earlier output:\n"
        f"{prev_raw}\n"
        "Return ONLY:\n"
        "<<<MODE>>>{code|explain|fallback}<<<END>>>\n"
        "<<<PLAN>>>{ \"endpoint\": \"/<path>\", \"params\": {\"<param>\": \"<value>\"}, \"chunk_rule\": \"<monthly|yearly|decade or empty>\", \"plot_rule\": \"<timeseries|map or empty>\"}<<<END>>>\n"
        "<<<CODE>>>\n<single runnable python script>\n<<<END>>>\n"
    )
