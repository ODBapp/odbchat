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
        "- Decide one token: 'code' or 'explain'.\n"
        "- Output 'code' if the user clearly asks for code/script/programmatic steps, API calls/downloads, or to continue/complete/follow up/revise previous code.\n"
        "- If the question contains negative phrases like 'without code', '不用程式', '不要 code', '不需寫程式', '不用 API', '除了程式以外', choose 'explain'.\n"
        "- Favor 'explain' when they ask to explain/define/list/compare/描述/解釋 without requesting code.\n"
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
        "CSV / JSON selection:\n"
        "- Prefer JSON if the user didn't explicitly ask CSV/file download.\n"
        "- For CSV: choose '/csv' endpoint if present; otherwise ONLY if 'format' enum includes 'csv', set format='csv'.\n\n"
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
        "- Infer variables from intent (e.g., SST→'sst'; anomaly→'sst_anomaly'; MHW等級→'level').\n"
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
        "- JSON: r.json() → DataFrame；CSV: only for '/csv' or enum format='csv' using pandas.read_csv(io.StringIO(r.text)).\n"
        "- If bbox crosses 180°/0° split requests and concat.\n"
        "- If 'date' present → to_datetime and sort chronologically.\n"
        "- Honor PLAN['plot_rule']: 'timeseries' → use plt.figure(), plt.plot(df['date'], ...), label axes/title/legend, and plt.show(); 'map' → build a 2D grid: pivot_table or reshaping so that Z has shape (len(lat_unique), len(lon_unique)), create Lon, Lat via numpy.meshgrid, then plt.pcolormesh(Lon, Lat, Z, shading='auto') with colorbar and plt.show(); missing/empty → skip plotting.\n"
        "- For map plots: do NOT call pcolormesh directly on raw columns; you must pivot to a grid first and ensure lon/lat are sorted unique values.\n"
        "- Do NOT prefix blocks with language labels or markdown fences. Use ONLY the <<<PLAN>>> and <<<CODE>>> tags with raw JSON/code content.\n"
        "- Return the script in a code fence. CODE cannot be empty.\n"
    )

def _sysrule_explain(force_zh: bool) -> str:
    lang = "- 回答請使用繁體中文。\n" if force_zh else "- Answer in the user's language.\n"
    return (
        "You are an ODB assistant. Provide a concise explanation that answers the question.\n"
        + lang +
        "- Use ONLY provided notes for specific resources; no code.\n"
        "- Keep it short but complete.\n"
    )

def _fmt_user_template(tmpl: str, *, query: str, top_k: int) -> str:
    # support both {top_k} and {topK}
    return tmpl.format(query=query, top_k=top_k, topK=top_k)

def build_main_prompt(*, query: str, notes: str, whitelist: Dict[str, Any], top_k: int = 6, prev_plan: Dict[str, Any] | None = None) -> str:
    strict_format = (
        "ABSOLUTE OUTPUT RULES:\n"
        "- Your FIRST characters MUST be exactly: '<<<MODE>>>' (no preface, no commentary).\n"
        "- NEVER output only 'code' or 'explain'. You MUST output ALL required blocks for the chosen MODE.\n"
        "If MODE=code (for STEP 2A and STEP 3):\n"
        "- You MUST output BOTH a <<<PLAN>>> JSON block and a <<<CODE>>> block, in this order.\n"
        "- If MODE=code and you do not include a non-empty <<<CODE>>> block, the output is INVALID.\n"
        "ELSE If MODE=explain (for STEP 2B):\n"
        "- Put the answer in <<<ANSWER>>> only (no PLAN/CODE there).\n"
        "- Never repeat PLAN or CODE contents outside their tagged blocks.\n"
        "- NEVER wrap any block in markdown fences (no ```json or ```python). Output the tags directly.\n"
        "\n"
        "STRICT OUTPUT FORMAT (use these exact tagged blocks):\n"
        "<<<MODE>>>{code|explain}<<<END>>>\n"
        "<<<PLAN>>>{ \"endpoint\": \"/<path>\", \"params\": {\"<param>\": \"<value>\"}, \"chunk_rule\": \"<monthly|yearly|decade or empty>\", \"plot_rule\": \"<timeseries|map or empty>\"}<<<END>>>\n"
        "<<<CODE>>>\n<single runnable python script>\n<<<END>>>\n"
        "<<<ANSWER>>>\n<text answer>\n<<<END>>>\n"
    )

    classifier = _sysrule_classifier()
    planner    = _sysrule_planner(whitelist)
    coder      = _sysrule_code_assistant(whitelist)
    force_zh   = bool(re.search(r"[\u4e00-\u9fff]", query))
    explainer  = _sysrule_explain(force_zh)
    user_tmpl  = "{query}"
    user       = _fmt_user_template(user_tmpl, query=query, top_k=top_k)
    prev_plan_blob = json.dumps(prev_plan, ensure_ascii=False) if prev_plan else "(none)"

    return (
        "You are an ODB assistant that does Classifier, Planner, Coder, and Explainer in ONE pass.\n"
        "Think step-by-step INTERNALLY, but NEVER print your reasoning.\n"
        "Only print the tagged blocks defined in STRICT OUTPUT FORMAT.\n"
        "STEP 1 — " + classifier + "\n"
        "IF MODE=code THEN STEP 2A (Planner) — " + planner + "\n"
        "THEN STEP 3 (Coder) — " + coder + "\n"
        "ELSE IF MODE=explain THEN STEP 2B (Explainer) — " + explainer + "\n"
        + strict_format + "\n"
        "---- CONTEXT ----\n"
        f"RAG NOTES:\n{(notes or '').strip()}\n\n"
        f"PREVIOUS PLAN (inherit unless user explicitly changes values):\n{prev_plan_blob}\n\n"
        f"OAS whitelist:\n{json.dumps(_compact_whitelist(whitelist), ensure_ascii=False)}\n\n"
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
        "<<<MODE>>>{code|explain}<<<END>>>\n"
        "<<<PLAN>>>{ \"endpoint\": \"/<path>\", \"params\": {\"<param>\": \"<value>\"}, \"chunk_rule\": \"<monthly|yearly|decade or empty>\", \"plot_rule\": \"<timeseries|map or empty>\"}<<<END>>>\n"
        "<<<CODE>>>\n<single runnable python script>\n<<<END>>>\n"
    )
