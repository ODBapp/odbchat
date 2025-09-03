#!/usr/#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
rag_cli.py — ODBchat RAG CLI (rev: 2025-09-03)
- 分類(code/explain) → OAS 規劃 →（必要時）最小補齊
- 嚴格遵守 OAS 白名單（端點/參數/enum/append）
- 程式碼生成保證：
  * JSON 一律 r.json() → DataFrame；嚴禁 pd.read_json / io.StringIO（除非 CSV）
  * CSV 僅在 /csv 或 format='csv' 下用 pandas.read_csv(io.StringIO(r.text))
  * 若有 'date' 欄位，轉 datetime；時序圖 x 軸優先用 'date'
  * 地圖請用 pcolormesh（禁止用 scatter 畫分佈地圖）
  * 嚴禁手刻 query string；必用 params=dict
- 追問承接：繼承上一輪的空間/時間/append、CSV 偏好（不自創）
- MHW 特例：OAS 時間跨度限制（<=10° → 10年；>10° → 1年；>90° → 1個月）→ 要求分段抓取 + concat
- 程式「沒寫完」：提供續寫路徑（帶入上次完整 code，請 LLM 產生合併後的完整單檔）
- 地圖任務守門失敗：提供安全地圖模板 fallback（pcolormesh + pivot）
"""

from __future__ import annotations

import os, re, sys, json, argparse, warnings
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import requests

try:
    import yaml
except Exception:
    yaml = None

# --------------------
# Config
# --------------------
QDRANT_HOST    = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION     = os.environ.get("QDRANT_COL", "odb_mhw_knowledge_v1")

EMBED_MODEL    = os.environ.get("EMBED_MODEL", "thenlper/gte-small")

OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "600"))

LLAMA_URL      = os.environ.get("LLAMA_URL", "http://localhost:8001/completion")
LLAMA_TIMEOUT  = float(os.environ.get("LLAMA_TIMEOUT", "600"))

BASE_URL       = "https://eco.odb.ntu.edu.tw"

# --------------------
# Init
# --------------------
qdrant   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

# 會話態
LAST_PLAN: Optional[Dict[str,Any]] = None
OAS_CACHE: Optional[Dict[str,Any]] = None
LAST_CODE_TEXT: Optional[str] = None  # 用於「續寫/補完」情境

# --------------------
# Intent hints
# --------------------
CSV_HINT_RE    = re.compile(r"\bcsv\b|下載|匯出|存檔", re.I)
YEAR_RE        = re.compile(r"\b(19|20)\d{2}\b")
MONTH_RE       = re.compile(r"\b(19|20)\d{2}-(0[1-9]|1[0-2])\b")
MAP_HINT_RE    = re.compile(r"地圖|分佈|分布|distribution\s*map|map", re.I)
CONTINUE_HINT  = re.compile(r"繼續|續寫|接著|補完|沒寫完|沒有寫完|未完成|continue|carry on", re.I)

# 變數輕量同義詞 cue（白名單約束下才納入）
VAR_SYNONYMS = [
    ("sst_anomaly", r"距平|異常|anomal(y|ies)"),
    ("level",       r"\blevel\b|等級|分級"),
    ("td",          r"熱位移|thermal\s*displacement"),
    ("sst",         r"\bsst\b|海溫|海表溫"),
]

def extract_user_requested_vars(question: str, allowed: List[str]) -> List[str]:
    out = []
    q = question.lower()
    for var, rx in VAR_SYNONYMS:
        if re.search(rx, q, re.I) and (not allowed or var in allowed):
            out.append(var)
    return sorted(set(out))

def month_hint_from_question(question: str) -> Optional[str]:
    m = MONTH_RE.search(question)
    return m.group(0) if m else None

# --------------------
# Helpers
# --------------------
def get_payload_text(payload: Dict[str, Any]) -> str:
    txt = payload.get("text")
    if isinstance(txt, str): return txt
    for k in ("content","body","raw"):
        v = payload.get(k)
        if isinstance(v, str): return v
    return ""

def get_title_url(payload: Dict[str, Any]) -> Tuple[str,str]:
    title = payload.get("title") or payload.get("source_file") or payload.get("canonical_url") or "(untitled)"
    url   = payload.get("canonical_url") or payload.get("source_file") or ""
    return title, url

def encode_query(text: str) -> List[float]:
    v = embedder.encode(text, show_progress_bar=False, normalize_embeddings=True)
    return v.tolist() if hasattr(v, "tolist") else list(v)

# --------------------
# Qdrant recall
# --------------------
def query_qdrant(question: str, topk: int=6, debug: bool=False) -> List[Dict[str, Any]]:
    vec = encode_query(question)
    resp = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=max(20, topk*4),
        with_payload=True,
        with_vectors=False,
        query_filter=None,
    )
    points = getattr(resp, "points", resp)
    hits = [{"text": get_payload_text(getattr(p, "payload", {}) or {}),
             "payload": getattr(p, "payload", {}) or {}} for p in points]

    # 再補 code/api 偏好（增加候選多樣性）
    code_filter = models.Filter(should=[
        models.FieldCondition(key="doc_type", match=models.MatchValue(value="code_snippet")),
        models.FieldCondition(key="doc_type", match=models.MatchValue(value="code_example")),
        models.FieldCondition(key="doc_type", match=models.MatchValue(value="cli_tool_guide")),
        models.FieldCondition(key="source_type", match=models.MatchText(text="code")),
    ])
    try:
        resp2 = qdrant.query_points(
            collection_name=COLLECTION,
            query=vec,
            limit=40,
            with_payload=True,
            with_vectors=False,
            query_filter=code_filter,
        )
        pts2 = getattr(resp2, "points", resp2)
        hits += [{"text": get_payload_text(getattr(p, "payload", {}) or {}),
                  "payload": getattr(p, "payload", {}) or {}} for p in pts2]
    except Exception:
        pass

    # 去重
    out, seen = [], set()
    for h in hits:
        p = h.get("payload", {}) or {}
        key = p.get("doc_id") or (p.get("title"), p.get("source_file"))
        if key in seen: continue
        seen.add(key)
        out.append(h)

    # 題意加權與過濾
    ql = question.lower()
    def boost(p: Dict[str,Any]) -> float:
        score = 0.0
        dt = (p.get("doc_type") or "").lower()
        title = (p.get("title") or "").lower()
        if dt in ("api_spec","code_snippet","code_example"): score += 1.5
        if dt == "cli_tool_guide":
            if re.search(r"\bcli\b|命令列|指令|工具", ql):
                score += 0.6
            else:
                score -= 0.6
        if ("enso" in title) and not re.search(r"enso|niño|la\s*niña", ql):
            score -= 0.8
        return score

    scored = [(boost(h["payload"]), h) for h in out]
    scored = [x for x in scored if x[0] >= -0.5]  # 丟掉負分太多的
    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [h for _,h in scored]

    # 輕度多樣化
    uniq, seen2 = [], set()
    for h in reranked:
        p = h["payload"]; k = (p.get("doc_type"), p.get("title"))
        if k in seen2: continue
        seen2.add(k); uniq.append(h)
        if len(uniq) >= topk: break

    if debug:
        from collections import Counter
        dist = Counter([(h["payload"].get("doc_type"), h["payload"].get("title")) for h in uniq])
        print(f"[DEBUG] diversified selection: {dict(dist)}")
        print(f"[DEBUG] total hits(after merge dedupe): {len(out)}, selected: {len(uniq)}")
    return uniq

# --------------------
# OAS parsing / harvest（含快取）
# --------------------
def _json_pointer(obj: Any, pointer: str) -> Any:
    if not pointer.startswith("#/"): return None
    cur = obj
    for p in pointer[2:].split("/"):
        if isinstance(cur, dict) and p in cur: cur = cur[p]
        else: return None
    return cur

_ALLOWED_DESC_RE = re.compile(r"Allowed fields\s*:\s*'([^']+)'(?:\s*,\s*'([^']+)')?(?:\s*,\s*'([^']+)')?(?:\s*,\s*'([^']+)')?", re.I)

def _parse_oas_text(raw: str) -> Dict[str, Any]:
    params_set, paths_list = set(), []
    append_set = set()
    enums_map: Dict[str, set] = {}

    def add_enum(name: str, enum_vals):
        if not name: return
        if enum_vals:
            s = enums_map.setdefault(str(name), set())
            for v in enum_vals: s.add(str(v))

    if yaml is not None:
        try:
            y = yaml.safe_load(raw)
        except Exception:
            y = None
        if isinstance(y, dict):
            # paths
            for p in (y.get("paths") or {}):
                paths_list.append(p)
            # components.parameters
            comp_params = (y.get("components") or {}).get("parameters") or {}
            for key, pobj in comp_params.items():
                name = (pobj or {}).get("name") or key
                if name: params_set.add(str(name))
                sch = (pobj or {}).get("schema") or {}
                enum = sch.get("enum") or (sch.get("items") or {}).get("enum") or []
                add_enum(name, enum)
                if str(name) == "append":
                    for v in enum: append_set.add(str(v))
                    m = _ALLOWED_DESC_RE.search(pobj.get("description") or "")
                    if m:
                        for g in m.groups():
                            if g: append_set.add(g)
            # in-path parameters
            for _, pbody in (y.get("paths") or {}).items():
                for _, mobj in (pbody or {}).items():
                    if not isinstance(mobj, dict): continue
                    for prm in (mobj.get("parameters") or []):
                        if not isinstance(prm, dict): continue
                        if "$ref" in prm:
                            pobj = _json_pointer(y, prm["$ref"])
                            if isinstance(pobj, dict):
                                name = pobj.get("name")
                                if name: params_set.add(str(name))
                                sch = (pobj or {}).get("schema") or {}
                                enum = sch.get("enum") or (sch.get("items") or {}).get("enum") or []
                                add_enum(name, enum)
                                if str(name) == "append":
                                    for v in enum: append_set.add(str(v))
                                    m = _ALLOWED_DESC_RE.search(pobj.get("description") or "")
                                    if m:
                                        for g in m.groups():
                                            if g: append_set.add(g)
                        else:
                            name = prm.get("name")
                            if name: params_set.add(str(name))
                            sch = prm.get("schema") or {}
                            enum = sch.get("enum") or (sch.get("items") or {}).get("enum") or []
                            add_enum(name, enum)
                            if str(name) == "append":
                                for v in enum: append_set.add(str(v))
                                m = _ALLOWED_DESC_RE.search(prm.get("description") or "")
                                if m:
                                    for g in m.groups():
                                        if g: append_set.add(g)

    # regex fallback
    for p in re.findall(r'(?m)^\s*(/api/[^\s:]+)\s*:', raw): paths_list.append(p.strip())
    for p in re.findall(r'(?m)^\s*-\s*name\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$', raw): params_set.add(p.strip())
    m = _ALLOWED_DESC_RE.search(raw)
    if m:
        for g in m.groups():
            if g: append_set.add(g)

    enums_dict = {k: sorted(list(v)) for k, v in enums_map.items()}
    return {
        "params": sorted(params_set),
        "paths": sorted(dict.fromkeys(paths_list)),
        "append_allowed": sorted(append_set),
        "param_enums": enums_dict,
    }

def _collect_api_specs_from_hits(hits: List[Dict[str,Any]]) -> List[str]:
    texts=[]
    for h in hits:
        p = h.get("payload",{}) or {}
        if (p.get("doc_type") or "").lower()=="api_spec":
            raw = get_payload_text(p)
            if raw: texts.append(raw)
    return texts

def _collect_api_specs_full_scan(limit: int = 1000) -> List[str]:
    texts=[]
    flt = models.Filter(must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="api_spec"))])
    try:
        points, next_off = qdrant.scroll(COLLECTION, flt, with_payload=True, with_vectors=False, limit=min(256, limit))
        while True:
            for p in points:
                pay = getattr(p,"payload",None) or {}
                raw = get_payload_text(pay)
                if raw: texts.append(raw)
            if not next_off or len(texts)>=limit: break
            points, next_off = qdrant.scroll(COLLECTION, flt, with_payload=True, with_vectors=False,
                                             limit=min(256, limit-len(texts)), offset=next_off)
    except TypeError:
        pass
    return texts

def harvest_oas_whitelist(hits: List[Dict[str,Any]], debug: bool=False) -> Optional[Dict[str,Any]]:
    global OAS_CACHE
    if OAS_CACHE is not None:
        if debug:
            print(f"[DEBUG] OAS cached: params={len(OAS_CACHE.get('params',[]))}, paths={len(OAS_CACHE.get('paths',[]))}")
        return OAS_CACHE

    texts = _collect_api_specs_from_hits(hits)
    P,A,Paths = set(), set(), set()
    Enums: Dict[str,List[str]] = {}

    for raw in texts:
        meta = _parse_oas_text(raw)
        P.update(meta["params"]); A.update(meta["append_allowed"]); Paths.update(meta["paths"])
        for k,v in (meta["param_enums"] or {}).items():
            Enums.setdefault(k, [])
            for x in v:
                if x not in Enums[k]: Enums[k].append(x)

    if (len(P)<=1) or (not Paths):
        for raw in _collect_api_specs_full_scan(limit=1000):
            meta = _parse_oas_text(raw)
            P.update(meta["params"]); A.update(meta["append_allowed"]); Paths.update(meta["paths"])
            for k,v in (meta["param_enums"] or {}).items():
                Enums.setdefault(k, [])
                for x in v:
                    if x not in Enums[k]: Enums[k].append(x)

    if debug:
        print(f"[DEBUG] OAS paths found: {sorted(Paths)}")
        print(f"[DEBUG] OAS params: {sorted(P)}")
        print(f"[DEBUG] OAS append allowed: {sorted(A)}")
        if Enums:
            print(f"[DEBUG] OAS enums: {json.dumps(Enums, ensure_ascii=False)}")

    if not P and not Paths: 
        return None

    OAS_CACHE = {"params": sorted(P), "append_allowed": sorted(A), "paths": sorted(Paths), "param_enums": Enums}
    return OAS_CACHE

# --------------------
# LLM calls
# --------------------
def call_ollama_raw(prompt: str, timeout: float = OLLAMA_TIMEOUT) -> str:
    resp = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=timeout)
    resp.raise_for_status(); data = resp.json()
    return data.get("response","").strip() or data.get("choices",[{}])[0].get("text","")

def call_llamacpp_raw(prompt: str, timeout: float = LLAMA_TIMEOUT) -> str:
    resp = requests.post(LLAMA_URL, json={"prompt": prompt, "n_predict": 512, "temperature": 0.0}, timeout=timeout)
    resp.raise_for_status(); data = resp.json()
    return (data.get("content") or data.get("choices",[{}])[0].get("text","")).strip()

# --------------------
# Mode classifier
# --------------------
def llm_choose_mode(question: str, llm: str) -> str:
    sysrule = (
        "You are a classifier. Read the question and output exactly one token: 'code' or 'explain'.\n"
        "- Output 'explain' when the user asks to explain/define/list/compare/describes something, or explicitly says not to use code.\n"
        "- Output 'code' only when the user asks for code, script, programmatic steps, API calls, downloads, plotting, or when they ask to continue/complete previous code.\n"
        "Do NOT include any other text or reasoning. Output exactly one token."
    )
    prompt = f"{sysrule}\n\nQuestion:\n{question}\n\nYour answer (one token):"
    try:
        raw = call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)
        ans = (raw or "").strip().lower()
        if ans not in ("code","explain"):
            ans = "explain"
        return ans
    except Exception:
        ql = question.lower()
        has_continue = bool(CONTINUE_HINT.search(ql))
        has_code_words = any(k in ql for k in ["python","code","程式","範例","sample","plot","畫圖","下載","api 呼叫","呼叫 api","requests"])
        has_no_code_neg = any(k in ql for k in ["不要程式","不用程式","除了程式","不用 code","不要 code","不是程式","不用寫程式","如果不用程式","除了程式以外"])
        if has_no_code_neg: return "explain"
        return "code" if (has_continue or has_code_words) else "explain"

# --------------------
# Planning
# --------------------
def llm_json_plan(question: str, oas: Dict[str,Any], llm: str,
                  prev_plan: Optional[Dict[str,Any]]=None,
                  debug: bool=False,
                  csv_requested: bool=False,
                  years_hint: Optional[Tuple[str,str]]=None,
                  user_append_hints: Optional[List[str]]=None) -> Optional[Dict[str,Any]]:
    allowed_paths   = oas.get("paths", [])
    allowed_params  = oas.get("params", [])
    allowed_append  = oas.get("append_allowed", [])
    param_enums     = oas.get("param_enums", {}) or {}

    sysrule = (
        "You are an API planner for ODB APIs (e.g., MHW API) which complies with OpenAPI Specifications (OAS). "
        "Your goal is to select exactly ONE endpoint and a parameter dict that are BOTH strictly within the OAS whitelists provided. "
        "Do NOT invent endpoints or parameters.\n\n"
        "Spatial-related params guidance:\n"
        "- If the question mentions a region or place, estimate an appropriate spatial range or constraint for that region/place by using available geographic parameters from the whitelist "
        "(e.g., any params whose names relate to lon/lat, or bbox: lon0,lat0,lon1,lat1), but use ONLY parameter names present in the whitelist. "
        "For region-level analysis, prefer a bounding box (two distinct values per axis) rather than a single point.\n"
        "Temporal-related params guidance:\n"
        "- If the question implies a specific year or range, set temporal-related params (e.g., start/end, or any params whose name relate to date/datetime but use ONLY parameter names present in the whitelist) accordingly (YYYY-MM-DD). "
        "You may deduce a 12-month range when the user mentions a single year.\n\n"
        "Endpoint or format-related query params for CSV request guidance:\n"
        "- Prefer an endpoint that outputs JSON if the user does not explicitly ask for CSV or file download. "
        "- If the user explicitly asks for CSV or file download, then return an endpoint path that clearly indicates CSV (e.g., contains '/csv') if such an endpoint exists in the whitelist.\n"
        "- Otherwise, if a 'format' parameter exists in the whitelist and its enum includes 'csv', set format='csv'.\n"
        "- Otherwise, return an endpoint that outputs JSON.\n\n"
        "The query parameter: append usage guidance:\n"
        "- If the user explicitly requested certain data variables (provided below as 'user_append_hints'), include ALL of them in 'append' (comma-separated) IF and ONLY IF 'append' exists in the whitelist and each value is allowed by the OAS.\n\n"
        "If a previous plan is provided, inherit its parameter values unless the user explicitly asked to change them; only modify what is required by the new question."
    )

    prev_line = f"Previous plan:\n{json.dumps(prev_plan, ensure_ascii=False)}" if prev_plan else "Previous plan: (none)"
    csv_line  = f"User {'requested' if bool(csv_requested) else 'NOT requested'} CSV in endpoint or query params."
    years_line= f"Years hint: {json.dumps(years_hint)}"
    vars_line = f"user_append_hints: {json.dumps(user_append_hints or [], ensure_ascii=False)}"

    oas_blob  = {
        "paths": allowed_paths,
        "params": allowed_params,
        "append_allowed": allowed_append,
        "param_enums": param_enums
    }

    prompt = (
        f"{sysrule}\n\n"
        f"Question:\n{question}\n\n"
        f"{prev_line}\n{csv_line}\n{years_line}\n{vars_line}\n\n"
        f"OAS whitelist (for you to choose from):\n{json.dumps(oas_blob, ensure_ascii=False)}\n\n"
        "Return JSON with EXACT shape:\n"
        "{\n"
        '  "endpoint": "<one path from whitelist>",\n'
        '  "params": {\n'
        '    "<param_name>": "<value>"\n'
        "  }\n"
        "}\n"
        "Output ONLY the JSON object, no prose, no code fences."
    )

    try:
        raw = call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)
        raw = re.sub(r"^```json\s*|\s*```$", "", (raw or "").strip(), flags=re.I)
        plan = json.loads(raw)
        if debug: print(f"[DEBUG] Plan(raw): {raw}")
        return plan
    except Exception as e:
        if debug: print(f"[DEBUG] Plan LLM failed: {e}")
        return None

# --------------------
# Plan validation & refine
# --------------------
def _split_csv(val: str) -> List[str]:
    return [x.strip() for x in str(val).split(",") if x.strip()]

def validate_plan(plan: Dict[str,Any], oas: Dict[str,Any]) -> Tuple[bool, str]:
    if not isinstance(plan, dict): return False, "plan is not a dict"
    endpoint = plan.get("endpoint"); params= plan.get("params")
    if not isinstance(endpoint, str) or not isinstance(params, dict):
        return False, "missing endpoint/params"
    if endpoint not in set(oas.get("paths", [])):
        return False, f"endpoint '{endpoint}' not in OAS whitelist"

    allowed_params = set(oas.get("params", []))
    for k in params.keys():
        if k not in allowed_params:
            return False, f"param '{k}' not allowed"

    # enum 檢查（如 format）
    enums = oas.get("param_enums", {}) or {}
    for k,v in params.items():
        if k in enums and enums[k]:
            if str(v) not in set(enums[k]):
                return False, f"param '{k}' value '{v}' not allowed (enum)"

    # append 值檢查
    if "append" in params:
        allowed_append = set(oas.get("append_allowed", []))
        if allowed_append:
            items = _split_csv(params["append"])
            if not items or any(it not in allowed_append for it in items):
                return False, f"append contains values not allowed: {params['append']}"

    # YYYY-MM → YYYY-MM-DD
    for key in ("start","end"):
        if key in params and re.match(r"^\d{4}-\d{2}$", str(params[key])):
            params[key] = str(params[key]) + "-01"
    return True, ""

def llm_refine_plan(question: str,
                    oas: Dict[str,Any],
                    plan: Dict[str,Any],
                    prev_plan: Optional[Dict[str,Any]],
                    llm: str,
                    debug: bool=False) -> Dict[str,Any]:
    allowed_paths  = oas.get("paths", [])
    allowed_params = oas.get("params", [])
    allowed_append = oas.get("append_allowed", [])
    param_enums    = oas.get("param_enums", {}) or {}

    sysrule = (
        "You are validating an API plan. Do NOT change the endpoint. "
        "Only propose minimal additions to the params dict if the current plan is missing obviously-required or strongly-implied fields.\n"
        "Rules:\n"
        "- Use ONLY parameters present in the whitelist.\n"
        "- If the question mentions a year but 'end' is missing and allowed, add the year's end date.\n"
        "- If the question mentions a region/place and geographic params are allowed but missing, add a reasonable bounding box (two distinct values per axis) instead of a single point.\n"
        "- If the user explicitly asked for certain variables and 'append' is allowed, ensure all are included (comma-separated) but only from the allowed values.\n"
        "- If a previous plan exists, inherit spatial/temporal params from it when appropriate (e.g., follow-up that says '改成下載 CSV').\n\n"
        "Return ONLY a JSON object with one of the following shapes:\n"
        '{\"action\":\"ok\"}\n'
        'or\n'
        '{\"action\":\"patch\",\"add\":{\"param1\":\"value\", ...}}\n'
        "No other text."
    )

    blob = {
        "whitelist": {
            "paths": allowed_paths,
            "params": allowed_params,
            "append_allowed": allowed_append,
            "param_enums": param_enums
        },
        "question": question,
        "current_plan": plan,
        "previous_plan": prev_plan or {},
    }

    prompt = f"{sysrule}\n\n{json.dumps(blob, ensure_ascii=False)}\n"
    try:
        raw = call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)
        raw = (raw or "").strip()
        data = json.loads(re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.I))
        if debug: print(f"[DEBUG] Refine(raw): {data}")

        if not isinstance(data, dict): return plan
        if data.get("action") == "patch" and isinstance(data.get("add"), dict):
            patched = dict(plan)
            params  = dict(patched.get("params", {}))
            enums   = oas.get("param_enums", {}) or {}
            for k,v in data["add"].items():
                if k in allowed_params:
                    if k in enums and enums[k]:
                        if str(v) not in set(enums[k]):
                            continue
                    params[k] = v
            patched["params"] = params
            return patched
        return plan
    except Exception as e:
        if debug: print(f"[DEBUG] Refine failed: {e}")
        return plan

def _needs_refine(plan: Dict[str,Any], req_vars: List[str], years_hint: Optional[Tuple[str,str]]) -> bool:
    params = plan.get("params", {})
    if req_vars:
        cur = set(_split_csv(params.get("append",""))) if "append" in params else set()
        if not set(req_vars).issubset(cur):
            return True
    if years_hint and ("start" in params and "end" not in params):
        return True
    if all(k in params for k in ("lat0","lat1","lon0","lon1")):
        try:
            if float(params["lat0"]) == float(params["lat1"]) and float(params["lon0"]) == float(params["lon1"]):
                return True
        except Exception:
            pass
    return False

def inherit_from_prev_if_reasonable(plan: Dict[str,Any], prev_plan: Optional[Dict[str,Any]], oas: Dict[str,Any], csv_requested: bool, debug: bool=False) -> Dict[str,Any]:
    if not prev_plan: return plan
    allowed_params = set(oas.get("params", []))
    patched = dict(plan)
    p_now = dict(patched.get("params", {}))
    p_prev= dict(prev_plan.get("params", {}))

    # 繼承：缺什麼補什麼（不覆蓋現值）
    for k,v in p_prev.items():
        if k in allowed_params and k not in p_now:
            p_now[k] = v

    # CSV 追問：若使用者要 CSV
    if csv_requested:
        paths = oas.get("paths", [])
        if patched.get("endpoint") in paths:
            csv_paths = [p for p in paths if p.endswith("/csv")]
            if csv_paths and not patched["endpoint"].endswith("/csv"):
                base = patched["endpoint"].rsplit("/",1)[0]
                cand = [p for p in csv_paths if p.startswith(base)]
                patched["endpoint"] = (cand[0] if cand else csv_paths[0])
            else:
                enums = oas.get("param_enums", {}) or {}
                if "format" in allowed_params and "format" in enums and ("csv" in set(enums["format"])):
                    p_now["format"] = "csv"

    patched["params"] = p_now
    if debug and patched != plan:
        print(f"[DEBUG] Inherit prev → {patched}")
    return patched

# --------------------
# MHW 特例：時間跨度限制 → 代碼生成提示（不直接改 plan）
# --------------------
def mhw_span_hint(plan: Dict[str,Any]) -> Optional[Dict[str,Any]]:
    ep = plan.get("endpoint","")
    if "/api/mhw" not in ep: return None
    p  = plan.get("params",{})
    try:
        lon0, lon1 = float(p.get("lon0")), float(p.get("lon1"))
        lat0, lat1 = float(p.get("lat0")), float(p.get("lat1"))
    except Exception:
        return None
    start, end = p.get("start"), p.get("end")
    if not start or not end: return None

    def parse_date(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")

    dx = abs(lon1 - lon0)
    dy = abs(lat1 - lat0)
    try:
        dur_days = (parse_date(end) - parse_date(start)).days
    except Exception:
        return None

    chunk = None
    if dx > 90 or dy > 90:
        chunk = {"mode": "monthly", "max_months": 1}
    elif dx > 10 or dy > 10:
        chunk = {"mode": "yearly", "max_years": 1}
    else:
        if dur_days > 365*10 + 3:
            chunk = {"mode": "decade", "max_years": 10}
    return chunk

# --------------------
# Code generation & guard
# --------------------
def code_from_plan_via_llm(question: str, plan: Dict[str,Any], llm: str) -> str:
    endpoint = plan["endpoint"]; params = plan["params"]
    params_json = json.dumps(params, ensure_ascii=False, indent=2)

    # MHW 分段提示（若需要）
    chunk_hint = mhw_span_hint(plan)
    chunk_line = ""
    if chunk_hint:
        if chunk_hint["mode"] == "monthly":
            chunk_line = ("因為此 BBox 超過 90°×90°，依 OAS 只能查 1 個月。\n"
                          "你必須以「逐月分段」迴圈抓取，將每段結果以 pandas.concat 串聯。")
        elif chunk_hint["mode"] == "yearly":
            chunk_line = ("因為此 BBox 超過 10°×10°，依 OAS 只能查 1 年。\n"
                          "你必須以「逐年分段」迴圈抓取，將每段結果以 pandas.concat 串聯。")
        elif chunk_hint["mode"] == "decade":
            chunk_line = ("此 BBox 在 10°×10° 以內，依 OAS 最長 10 年。\n"
                          "若時間區間超過 10 年，你必須以「每 10 年」或「逐年」分段，最後以 pandas.concat 串聯。")

    # 強化規則：由「自然語意 → OAS → append → DataFrame 欄位」自動決定要畫的變數
    sysrule = (
        "You are a Python assistant. Write a concise, runnable script that uses requests to call the ODB API (e.g., MHW API) and load data into pandas.\n"
        "MANDATORY rules:\n"
        f"- Base URL is {BASE_URL}. Use EXACT endpoint and params provided below.\n"
        "- Call with: requests.get(f\"{BASE_URL}{endpoint}\", params=params, timeout=60)\n"
        "- Do NOT manually concatenate query strings; must use params=dict.\n"
        "- Do NOT add headers, API keys, or any extra parameters.\n"
        "- If and only if the endpoint ends with '/csv' OR params include format='csv':\n"
        "    * read CSV via: pandas.read_csv(io.StringIO(r.text)) and import io.\n"
        "- Otherwise (JSON):\n"
        "    * parse JSON via r.json() (NOT pandas.read_json, NOT io.StringIO).\n"
        "    * build DataFrame with pandas.DataFrame(r.json()).\n"
        "- If the DataFrame has a 'date' column, convert it with pandas.to_datetime(df['date']).\n"
        "- All comments must be Traditional Chinese or English.\n"
        "- Variable choice:\n"
        "    * Infer the intended variable(s) from the USER QUESTION semantics (e.g., '海溫'→sst; '海溫距平/異常'→sst_anomaly; '海洋熱浪等級'→level; '熱位移'→td),\n"
        "      but only use variables that are (i) allowed by OAS/append and (ii) actually present in df.columns.\n"
        "    * If multiple variables are requested, include them all in append (already provided) and plot the most relevant one per figure.\n"
        "    * If a chosen variable is missing, print available columns and skip plotting the missing one (do not invent columns).\n"
        "- Plotting:\n"
        "    * Time series: use 'date' on x-axis when available.\n"
        "    * Maps: build lon/lat 2D grid and use matplotlib.pyplot.pcolormesh (NOT scatter). Prefer meshgrid-style fill (no averaging) unless duplicates force aggregation.\n"
        "    * If the variable is categorical (e.g., 'level'), use a discrete colormap; if the user provides color palettes, honor it.\n"
        "- NEVER use pandas.read_json; NEVER wrap JSON with io.StringIO.\n"
    )
    if chunk_line:
        sysrule += "\nCHUNKING constraint:\n- " + chunk_line + "\n"

    prompt = f"""{sysrule}

User question:
{question}

Endpoint: {endpoint}
Params JSON (use exactly as-is):
{params_json}

Wrap the script in a single ```python code fence.
"""
    try:
        return (call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)).strip()
    except Exception as e:
        return f"[LLM error during code generation] {e}"

def code_continue_via_llm(question: str, plan: Dict[str,Any], last_code: str, llm: str) -> str:
    """將上一輪 code 作為基底，要求 LLM 產生『合併後的完整單檔』"""
    endpoint = plan["endpoint"]; params = plan["params"]
    params_json = json.dumps(params, ensure_ascii=False, indent=2)

    sysrule = (
        "You are a Python assistant. The user says the previous script was incomplete or needs changes. "
        "Take the PREVIOUS_SCRIPT below as the base, and output ONE complete, runnable script (single file) that:\n"
        "- Keeps the correct parts of the previous code;\n"
        "- Applies the user's new instructions (e.g., change dates, switch variables, finish map plotting);\n"
        "- Strictly uses the EXACT endpoint and params provided below (via requests.get(..., params=params));\n"
        "- JSON must be parsed with r.json(); CSV only via pandas.read_csv(io.StringIO(r.text));\n"
        "- If plotting a map, use pcolormesh (not scatter), and grid the data by lon/lat (pivot) before plotting;\n"
        "- If the DataFrame has 'date', convert it to datetime and use 'date' on time-series x-axis.\n"
        "- Output ONLY the final full script inside a single ```python code fence."
    )
    prompt = f"""{sysrule}

EXACT endpoint: {endpoint}
EXACT params JSON:
{params_json}

PREVIOUS_SCRIPT:
```python
{last_code}
User request now:
{question}
"""
    try:
        return (call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)).strip()
    except Exception as e:
        return f"[LLM error during code-continue] {e}"

def static_guard_check(code: str, oas: Dict[str,Any], expect_csv: bool, expect_map: bool) -> Tuple[bool, str]:
    if not code or not isinstance(code, str):
        return False, "empty code"

    # 端點白名單
    endpoints = set(oas.get("paths", []))
    used_paths = set(re.findall(r'["\'](/api/[^"\']+)["\']', code))
    for p in used_paths:
        if p not in endpoints:
            return False, f"code uses endpoint '{p}' not in whitelist"

    # 必須使用 params=
    if "requests.get(" in code and "params=" not in code:
        return False, "requests.get is used without params= dict"

    # 禁手刻 query string
    if re.search(r'requests\.get\([^)]*\?[^)]*\)', code):
        return False, "code manually concatenates query string"

    # CSV / JSON 解析規則
    if expect_csv:
        # CSV 路徑或 format=csv → 必須用 read_csv + io.StringIO
        if "pd.read_csv(" not in code or "io.StringIO(" not in code:
            return False, "CSV mode requires pandas.read_csv(io.StringIO(r.text)) with import io"
    else:
        # JSON 路徑 → 禁用 read_json 與 io.StringIO(json)
        if re.search(r'pd\.read_json\s*\(', code):
            return False, "JSON must use r.json(); pd.read_json is forbidden"
        if re.search(r'io\.StringIO\s*\(\s*r\.text\s*\)', code) and ("pd.read_csv(" not in code):
            return False, "JSON must not use io.StringIO; parse via r.json()"

    # 參數鍵/append 值合法性
    m = re.search(r"params\s*=\s*\{([\s\S]*?)\}\s*", code)
    if m:
        body = m.group(1)
        keys = re.findall(r'["\']([A-Za-z_][A-Za-z0-9_]*)["\']\s*:', body)
        allowed_params = set(oas.get("params", []))
        for k in keys:
            if k not in allowed_params:
                return False, f"code uses param '{k}' not in whitelist"
        am = re.search(r'["\']append["\']\s*:\s*["\']([^"\']+)["\']', body)
        if am:
            vals = _split_csv(am.group(1))
            allowed_append = set(oas.get("append_allowed", []))
            if allowed_append and any(v not in allowed_append for v in vals):
                return False, f"append value(s) not allowed: {am.group(1)}"

    # 不得提及 API key/token
    if re.search(r'api[_-]?key|token|authorization|bearer', code, re.I):
        return False, "code mentions API key/token"

    # 地圖任務時，不得用 scatter；至少包含 pcolormesh
    if expect_map:
        if re.search(r'\.scatter\s*\(', code):
            return False, "map plotting must not use scatter"
        if "pcolormesh(" not in code and "plt.pcolormesh(" not in code:
            return False, "map plotting must use pcolormesh"

    # code fence 完整性（避免截斷）
    if "```python" in code and "```" not in code.strip().split("```python",1)[-1]:
        return False, "code fence likely truncated"

    return True, ""

def build_timeseries_fallback_template(plan: Dict[str, Any], expect_csv: bool) -> str:
    endpoint = plan["endpoint"]
    params = plan["params"]

    # 從 append 決定候選變數；若沒有 append 就在 df 欄位內挑（排除 lon/lat/date）
    append_vals = []
    if isinstance(params.get("append"), str):
        append_vals = [v.strip() for v in params["append"].split(",") if v.strip()]

    code = [
        "# 最小可執行範例（時序圖）：依需求自行調整參數與欄位",
        "import requests",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        f'BASE = "{BASE_URL}"',
        f'ENDPOINT = "{endpoint}"',
        f"params = {json.dumps(params, ensure_ascii=False, indent=2)}",
        "r = requests.get(f\"{BASE}{ENDPOINT}\", params=params, timeout=60)",
        "r.raise_for_status()",
    ]

    if expect_csv:
        code += [
            "import io",
            "df = pd.read_csv(io.StringIO(r.text))",
        ]
    else:
        code += [
            "df = pd.DataFrame(r.json())",
        ]

    code += [
        "if 'date' in df.columns:",
        "    df['date'] = pd.to_datetime(df['date'])",
        "",
        "# 依 append 推斷要畫的變數；若未指定，從欄位中挑選（略過 lon/lat/date）",
        f"append_hint = {append_vals!r}",
        "candidates = [c for c in append_hint if c in df.columns]",
        "if not candidates:",
        "    candidates = [c for c in df.columns if c not in ('lon','lat','date')]",
        "if not candidates:",
        "    raise RuntimeError(f'找不到可繪製的欄位；可用欄位：{list(df.columns)}')",
        "Y_COL = candidates[0]",
        "",
        "print(df.head())",
        "if 'date' in df.columns:",
        "    df = df.sort_values('date')",
        "    df.plot(x='date', y=Y_COL, figsize=(10,4), title=f'{Y_COL} timeseries')",
        "    plt.tight_layout()",
        "    plt.show()",
        "",
    ]
    return "\n".join(code)

def build_map_fallback_template(
    plan: Dict[str, Any],
    expect_csv: bool,
    month_hint: Optional[str]
) -> str:
    endpoint = plan["endpoint"]
    params = plan["params"]

    append_vals = []
    if isinstance(params.get("append"), str):
        append_vals = [v.strip() for v in params["append"].split(",") if v.strip()]

    code = [
        "# 最小可執行範例（地圖分佈）：meshgrid, 依需求自行調整參數",
        "import requests",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "from matplotlib.colors import ListedColormap",
        f'BASE = "{BASE_URL}"',
        f'ENDPOINT = "{endpoint}"',
        f"params = {json.dumps(params, ensure_ascii=False, indent=2)}",
        "r = requests.get(f\"{BASE}{ENDPOINT}\", params=params, timeout=60)",
        "r.raise_for_status()",
    ]

    if expect_csv:
        code += [
            "import io",
            "df = pd.read_csv(io.StringIO(r.text))",
        ]
    else:
        code += [
            "df = pd.DataFrame(r.json())",
        ]

    code += [
        "if 'date' in df.columns:",
        "    df['date'] = pd.to_datetime(df['date'])",
        "",
        "# 目標月份（若問題含 YYYY-MM 則使用；否則用 params['start'] 的月份）",
        f"target_month = {json.dumps(month_hint)}",
        "if target_month is None and 'start' in params:",
        "    try:",
        "        target_month = pd.to_datetime(params['start']).strftime('%Y-%m')",
        "    except Exception:",
        "        target_month = None",
        "",
        "if target_month and 'date' in df.columns:",
        "    mdf = df[df['date'].dt.strftime('%Y-%m') == target_month].copy()",
        "else:",
        "    mdf = df.copy()",
        "",
        "# 依 append 推斷要畫的變數；若未指定，從欄位中挑選（略過 lon/lat/date）",
        f"append_hint = {append_vals!r}",
        "candidates = [c for c in append_hint if c in mdf.columns]",
        "if not candidates:",
        "    candidates = [c for c in mdf.columns if c not in ('lon','lat','date')]",
        "if not candidates:",
        "    raise RuntimeError(f'找不到可繪製的欄位；可用欄位：{list(mdf.columns)}')",
        "VAR = candidates[0]",
        "",
        "for col in ['lon','lat', VAR]:",
        "    if col not in mdf.columns:",
        "        raise KeyError(f\"欄位 {col!r} 不在回應資料中。可用欄位：{list(mdf.columns)}\")",
        "",
        "# 轉為規則網格（逐點填入）",
        "lons = np.sort(mdf['lon'].unique())",
        "lats = np.sort(mdf['lat'].unique())",
        "Lon, Lat = np.meshgrid(lons, lats)",
        "Z = np.full((lats.size, lons.size), np.nan)",
        "lon_index = {v:i for i,v in enumerate(lons)}",
        "lat_index = {v:i for i,v in enumerate(lats)}",
        "for _, row in mdf.iterrows():",
        "    i = lat_index.get(row['lat']); j = lon_index.get(row['lon'])",
        "    if i is not None and j is not None:",
        "        Z[i, j] = row[VAR]",
        "",
        "# 類別變數（如 level）給離散色盤；否則使用連續色盤",
        "if VAR == 'level':",
        "    LEVEL_COLORS = ['#f5c268', '#ec6b1a', '#cb3827', '#7f1416']",
        "    cmap = ListedColormap(LEVEL_COLORS)",
        "    pcm = plt.pcolormesh(Lon, Lat, Z, shading='nearest', cmap=cmap, vmin=0.5, vmax=4.5)",
        "else:",
        "    pcm = plt.pcolormesh(Lon, Lat, Z, shading='nearest')",
        "plt.colorbar(pcm, label=VAR)",
        "plt.xlabel('Longitude'); plt.ylabel('Latitude')",
        "title = f\"{VAR} distribution\" + (f\" ({target_month})\" if target_month else '')",
        "plt.title(title)",
        "plt.tight_layout()",
        "plt.show()",
        "",
    ]
    return "\n".join(code)

# --------------------
# Non-code prompt
# --------------------
def build_prompt(question: str, ctx: List[Dict[str, Any]], oas_info: Optional[Dict[str,Any]]) -> str:
    sys_rule = (
        "你是 ODB（海洋學門資料庫）助理。請只根據『依據』內容作答，允許語意近似（例如「海洋系統≈海洋生態系統」），"
        "但科學內容需審慎、可驗證。嚴禁自創或改造學術名詞；若不確定，請改用標準術語（如「聖嬰（El Niño）」「反聖嬰（La Niña）」「ENSO」）。"
        "若提及 ODB API 的端點或參數，必須嚴格遵守 OAS 白名單（端點/方法/參數不可虛構）。"
        "正文不要使用 [1][2] 或「來源1」等編號；參考資料請我在文末整理。"
        "回答使用繁體中文（不要使用簡體字詞）；英文術語可保留。"
    )
    if oas_info:
        sys_rule += f"（OAS 參數白名單：{', '.join(oas_info.get('params', []))}；append 允許值：{', '.join(oas_info.get('append_allowed', []))}）"

    ctx_text = ""
    for i, h in enumerate(ctx, 1):
        title, _ = get_title_url(h.get("payload", {}) or {})
        text = h.get("text","")
        if len(text) > 4000: text = text[:4000] + "…"
        ctx_text += f"\n[來源 {i}] {title}\n{text}\n"

    return f"{sys_rule}\n\n問題：{question}\n\n依據：\n{ctx_text}"

def format_citations(hits: List[Dict[str,Any]], question: str) -> str:
    def pri(p: Dict[str,Any]) -> int:
        dt = (p.get("doc_type") or "").lower()
        if dt == "api_spec": return 0
        if dt in ("code_snippet","code_example"): return 1
        if dt == "web_article": return 2
        if dt == "cli_tool_guide": return 3
        return 4

    ranked = sorted(hits, key=lambda h: pri(h.get("payload", {}) or {}))
    items=[]; seen=set()
    for h in ranked:
        p=h.get("payload",{}) or {}
        t,u = get_title_url(p); k=(t,u)
        if k in seen: continue
        seen.add(k)
        items.append(f"- {t} — {u}" if u else f"- {t}")
        if len(items)>=5: break
    return "\n".join(items) if items else "（無）"

# --------------------
# Core
# --------------------
def answer_once(
    question: str,
    llm: str = "ollama",
    topk: int = 6,
    temp: float = 0.2,
    max_tokens: int = 800,
    strict_api: bool = True,
    debug: bool = False,
    history: Optional[List[Tuple[str,str]]] = None,
    ctx_only: bool = False,
) -> str:
    global LAST_PLAN, LAST_CODE_TEXT

    # 1) 分類
    mode = llm_choose_mode(question, llm=llm)

    # 2) 檢索 & OAS
    hits = query_qdrant(question, topk=max(6, topk), debug=debug)
    oas_info = harvest_oas_whitelist(hits, debug=debug)

    # 3) 非程式題：直接解說
    if mode == "explain":
        prompt = build_prompt(question, hits, oas_info)
        if ctx_only: return prompt
        out = call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)
        out = out or "（無法在已知資料中產生可靠答案）"
        cites = format_citations(hits, question)
        LAST_CODE_TEXT = None
        return f"{out}\n\n---\n參考資料：\n{cites}"


    # 4) 程式題但無 OAS → 解說
    if not oas_info:
        prompt = build_prompt(question, hits, None)
        if ctx_only: return prompt
        out = call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)
        out = out or "（無法在已知資料中產生可靠答案）"
        cites = format_citations(hits, question)
        LAST_CODE_TEXT = None
        return f"{out}\n\n---\n參考資料：\n{cites}"

    # CSV 意圖與年份 hint
    csv_requested = bool(CSV_HINT_RE.search(question))
    years = [m.group(0) for m in YEAR_RE.finditer(question)]
    years_hint = None
    if years:
        y = years[0]
        years_hint = (f"{y}-01-01", f"{y}-12-31")
    month_hint = month_hint_from_question(question)
    expect_map = bool(MAP_HINT_RE.search(question))

    # 5) 規劃
    prev_plan = LAST_PLAN
    plan = llm_json_plan(
        question=question,
        oas=oas_info,
        llm=llm,
        prev_plan=prev_plan,
        debug=debug,
        csv_requested=csv_requested,
        years_hint=years_hint,
        user_append_hints=None
    )
    if not plan:
        prompt = build_prompt(question, hits, oas_info)
        out = call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)
        out = out or "（無法在已知資料中產生可靠答案）"
        cites = format_citations(hits, question)
        LAST_CODE_TEXT = None
        return f"{out}\n\n---\n參考資料：\n{cites}"

    # 使用者明確點名的變數 → 寫回 append（白名單約束）
    req_vars = extract_user_requested_vars(question, oas_info.get("append_allowed", []))
    if req_vars:
        pp = plan.setdefault("params", {})
        cur = set()
        if "append" in pp and pp["append"]:
            cur = set([x.strip() for x in str(pp["append"]).split(",") if x.strip()])
        for v in req_vars: cur.add(v)
        if cur:
            pp["append"] = ",".join(sorted(cur))

    # 承接上一輪（補缺的空間/時間/append、CSV 切換）
    plan = inherit_from_prev_if_reasonable(plan, prev_plan, oas_info, csv_requested, debug=debug)

    # 第一次驗證
    ok, msg = validate_plan(plan, oas_info)
    if debug and not ok:
        print(f"[DEBUG] Plan invalid@first: {msg}")

    # 6) 覺得需要才 refine（減少一次呼叫）
    if (not ok) or _needs_refine(plan, req_vars, years_hint):
        plan = llm_refine_plan(question, oas_info, plan, prev_plan, llm, debug=debug)

    # 第二次驗證
    ok, msg = validate_plan(plan, oas_info)
    if debug and not ok:
        print(f"[DEBUG] Plan invalid@refined: {msg}")
    if not ok:
        prompt = build_prompt(question, hits, oas_info)
        out = call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)
        out = out or "（無法在已知資料中產生可靠答案）"
        cites = format_citations(hits, question)
        LAST_CODE_TEXT = None
        return f"{out}\n\n---\n參考資料：\n{cites}"

    # 保存 plan（供追問承接）
    LAST_PLAN = {"endpoint": plan["endpoint"], "params": dict(plan["params"])}

    # 7) 產 code + 守門（加強：JSON 禁用 pd.read_json / io.StringIO）
    expect_csv = plan["endpoint"].endswith("/csv") or (plan["params"].get("format") == "csv")
    wants_continue = bool(CONTINUE_HINT.search(question)) and bool(LAST_CODE_TEXT)

    if wants_continue:
        code = code_continue_via_llm(question, plan, LAST_CODE_TEXT or "", llm=llm) or ""
    else:
        code = code_from_plan_via_llm(question, plan, llm=llm) or ""
    
    ok2, msg2 = static_guard_check(code, oas_info, expect_csv=expect_csv, expect_map=expect_map)
    if not ok2 and debug:
        print(f"[DEBUG] Code violates guard: {msg2}")

    # 地圖守門失敗 → 地圖 fallback；否則通用時序 fallback
    if not ok2:
        code = (build_map_fallback_template(plan, expect_csv, month_hint)
                if expect_map else
                build_timeseries_fallback_template(plan, expect_csv))

    # 避免空答
    if not code.strip():
        code = (build_map_fallback_template(plan, expect_csv, month_hint)
                if expect_map else
                build_timeseries_fallback_template(plan, expect_csv))

    # 記住這次完整 code（限制長度以免爆記憶）
    LAST_CODE_TEXT = code
    if LAST_CODE_TEXT and len(LAST_CODE_TEXT) > 18000:
        LAST_CODE_TEXT = LAST_CODE_TEXT[-18000:]

    cites = format_citations(hits, question)
    return f"{code}\n\n=== 參考資料 ===\n{cites}"

# --------------------
# CLI
# --------------------
def main():
    ap = argparse.ArgumentParser(description="ODBchat RAG CLI")
    ap.add_argument("question", nargs="?", help="要問的問題")
    ap.add_argument("--llm", choices=["ollama","llama","off"], default="ollama")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--strict-api", action="store_true", default=True)
    ap.add_argument("--debug", action="store_true", default=False)
    ap.add_argument("--ctx", action="store_true", default=False, help="只輸出 prompt（檢查 context 用）")
    ap.add_argument("--chat", action="store_true", default=False, help="互動模式")
    args = ap.parse_args()

    if not args.question and not args.chat:
        print("用法：rag_cli.py '你的問題'  或  rag_cli.py --chat")
        return

    history: List[Tuple[str,str]] = []

    if args.chat:
        print("進入互動模式（輸入 /exit 離開）")
        while True:
            try:
                q = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                print("\nBye."); break
            if not q.strip(): continue
            if q.strip() in ("/exit","/quit"):
                print("Bye."); break

            use_llm = args.llm != "off"
            out = answer_once(q,
                              llm=("ollama" if args.llm=="ollama" else ("llama" if args.llm=="llama" else "ollama")),
                              topk=args.topk, temp=args.temp, max_tokens=args.max_tokens,
                              strict_api=args.strict_api, debug=args.debug,
                              history=(history if use_llm else None),
                              ctx_only=args.ctx)
            print("\n=== 答覆 ===\n")
            print(out)
            history.append((q, out))
    else:
        out = answer_once(args.question,
                          llm=("ollama" if args.llm=="ollama" else ("llama" if args.llm=="llama" else "ollama")),
                          topk=args.topk, temp=args.temp, max_tokens=args.max_tokens,
                          strict_api=args.strict_api, debug=args.debug, history=None, ctx_only=args.ctx)
        print("\n=== 答覆 ===\n")
        print(out)

if __name__ == "__main__":
    main()
    if not ok2:
        code = build_fallback_template(plan, expect_csv)
