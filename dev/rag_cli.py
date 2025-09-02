#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
rag_cli.py — ODBchat RAG CLI (rev: unified fast path)
- 以 LLM 做「分類(code/explain)」→「OAS 內規劃」→（必要時）「最小補齊」
- 預設只需 2 次 LLM 呼叫（分類 + 規劃/或解說）；只有規劃驗證失敗或明顯缺值才呼叫第 3 次補齊
- 嚴格遵守 OAS 白名單（端點/參數/enum/append）
- 程式碼生成使用 requests.get(..., params=...)；禁止手刻 query string
- CSV 預設不使用，除非使用者明確要求或 OAS 只提供 CSV
- 非程式題：走解說流（繁體中文），正文無 [1][2] 編號；參考資料統一列文末
"""

from __future__ import annotations

import os, re, sys, json, argparse, warnings
from typing import Any, Dict, List, Tuple, Optional

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
# 預設 600 秒，配合你家中較慢的環境
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
OAS_CACHE: Optional[Dict[str,Any]] = None   # 快取 OAS 白名單（本程式生命週期）

# --------------------
# Intent hints（僅作最後備援；平時交給 LLM）
# --------------------
CSV_HINT_RE = re.compile(r"\bcsv\b|下載|匯出|存檔", re.I)
YEAR_RE     = re.compile(r"\b(19|20)\d{2}\b")

# --- Minimal variable hints（不是大詞庫，只是 4 個穩健 cue）---
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

    # 加一輪 code/api 偏好檢索（非過濾，只是補充樣本）
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

    # 題意相關的 rerank boost
    ql = question.lower()
    def boost(p: Dict[str,Any]) -> float:
        score = 0.0
        dt = (p.get("doc_type") or "").lower()
        title = (p.get("title") or "").lower()

        if dt in ("api_spec","code_snippet","code_example"):
            score += 1.5
        if dt == "cli_tool_guide":
            if re.search(r"\bcli\b|命令列|指令|工具", ql):
                score += 0.6
            else:
                score -= 0.4
        if ("enso" in title) and not re.search(r"enso|niño|la\s*niña", ql):
            score -= 0.8
        return score

    scored = [(boost(h["payload"]), h) for h in out]
    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [h for _,h in scored]

    # 多樣化（輕度）
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
# Mode classifier（只回 'code' 或 'explain'）
# --------------------
def llm_choose_mode(question: str, llm: str) -> str:
    """
    嚴格的單詞分類：'code' 或 'explain'
    """
    sysrule = (
        "You are a classifier. Read the question and output exactly one token: 'code' or 'explain'.\n"
        "- Output 'explain' when the user asks to explain/define/list/compare/describes something, or explicitly says not to use code.\n"
        "- Output 'code' only when the user asks for code, script, programmatic steps, API calls, downloads, or plotting.\n"
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
        # 備援（盡量偏向 explain）
        ql = question.lower()
        has_code_words = any(k in ql for k in ["python","code","程式","範例","sample","plot","畫圖","下載","api 呼叫","呼叫 api","requests"])
        has_no_code_neg = any(k in ql for k in ["不要程式","不用程式","除了程式","不用 code","不要 code","不是程式","不用寫程式","如果不用程式"])
        if has_no_code_neg: return "explain"
        return "code" if has_code_words else "explain"

# --------------------
# Planning（你的最新版 sysrule 精簡整合）
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
        "(e.g., any params whose names relate to lon/lat, or bbox: lon0,lat0,lon1,lat1 which specify longitude/latitude range), but use ONLY parameter names present in the whitelist. "
        "For region-level analysis, prefer a bounding box (two distinct values per axis) rather than a single point.\n"
        "Temporal-related params guidance:\n"
        "- If the question implies a specific year or range, set temporal-related params (e.g., start/end, or any params whose name relate to date or datetime but use ONLY parameter names present in the whitelist) accordingly (YYYY-MM-DD). "
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
    """
    只在需要時呼叫：為當前 plan 做「最小增補」（不可改 endpoint）
    """
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
    """是否值得呼叫 refine（減少第 3 次 LLM）"""
    params = plan.get("params", {})
    # 使用者點名了變數，但 append 沒包含完整
    if req_vars:
        cur = set(_split_csv(params.get("append",""))) if "append" in params else set()
        if not set(req_vars).issubset(cur):
            return True
    # 只有 start 沒 end，且有年份提示
    if years_hint and ("start" in params and "end" not in params):
        return True
    # bbox 退化為單點（lat0==lat1 且 lon0==lon1）
    if all(k in params for k in ("lat0","lat1","lon0","lon1")):
        try:
            if float(params["lat0"]) == float(params["lat1"]) and float(params["lon0"]) == float(params["lon1"]):
                return True
        except Exception:
            pass
    return False

# --------------------
# Code generation & guard
# --------------------
def code_from_plan_via_llm(question: str, plan: Dict[str,Any], llm: str) -> str:
    endpoint = plan["endpoint"]; params = plan["params"]
    params_json = json.dumps(params, ensure_ascii=False, indent=2)

    sysrule = (
        "You are a Python assistant. Write a concise, runnable script that uses requests to call the ODB API (e.g., MHW API) which complies with OpenAPI Specifications(OAS) and get the fetched data from the response. "
        "You MUST use the EXACT endpoint and params provided. "
        "Call it with: requests.get(f\"{BASE_URL}{endpoint}\", params=params) within try-except block. "
        "Do NOT manually concatenate query strings; do NOT add headers, API keys, or extra parameters. "
        "If and only if the provided endpoint ends with '/csv' or params include format='csv', read the response as CSV via pandas.read_csv(io.StringIO(r.text)) with an 'import io'. "
        "Otherwise parse JSON into a pandas DataFrame to manipulate the fetched data. "
        "All comments (if any) must be in Traditional Chinese or English."
        "Keep the code minimal and runable. If plotting or analyzing data is required in the question, provide an appropriate plot code for the fetched data. You should make sure the response format in OAS to use the available fields for correctly plotting. You should try to understand the programming requirements in the question and use your programming expertise to provide the correct code."
    ) 

    prompt = f"""{sysrule}

User question:
{question}

Base URL: {BASE_URL}
Endpoint: {endpoint}
Params JSON (use exactly as-is):
{params_json}

Wrap the script in a single ```python code fence.
"""
    try:
        return (call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)).strip()
    except Exception as e:
        return f"[LLM error during code generation] {e}"

def static_guard_check(code: str, oas: Dict[str,Any]) -> Tuple[bool, str]:
    endpoints = set(oas.get("paths", []))
    used_paths = set(re.findall(r'["\'](/api/[^"\']+)["\']', code))
    for p in used_paths:
        if p not in endpoints:
            return False, f"code uses endpoint '{p}' not in whitelist"

    # 禁手刻 query string
    if re.search(r'\?.*=', code) and "params=" not in code:
        return False, "code manually concatenates query string"

    # 必須使用 params=
    if "requests.get(" in code and "params=" not in code:
        return False, "requests.get is used without params= dict"

    # CSV 檢查：/csv 或 format='csv' → 需 io.StringIO
    is_csv_mode = any(p.endswith("/csv") for p in used_paths) or re.search(r"params\s*=\s*\{[\s\S]*?['\"]format['\"]\s*:\s*['\"]csv['\"]", code)
    if is_csv_mode:
        if "pd.read_csv(" in code and "io.StringIO(" not in code:
            return False, "CSV must be read via pandas.read_csv(io.StringIO(r.text))"
        if "import io" not in code:
            return False, "CSV mode requires 'import io' for io.StringIO"

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

    return True, ""

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
        sys_rule += f"（OAS 參數白名單：{', '.join(oas_info.get('params', []))}; append 允許值：{', '.join(oas_info.get('append_allowed', []))}）"

    ctx_text = ""
    for i, h in enumerate(ctx, 1):
        title, _ = get_title_url(h.get("payload", {}) or {})
        text = h.get("text","")
        if len(text) > 4000: text = text[:4000] + "…"
        ctx_text += f"\n[來源 {i}] {title}\n{text}\n"

    return f"{sys_rule}\n\n問題：{question}\n\n依據：\n{ctx_text}"

def format_citations(hits: List[Dict[str,Any]], question: str) -> str:
    # 優先序：api_spec > code_snippet/code_example > web_article > cli_tool_guide > other
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
    global LAST_PLAN

    # 1) 分類（只回 code/explain）
    mode = llm_choose_mode(question, llm=llm)

    # 2) 檢索 & OAS
    hits = query_qdrant(question, topk=max(6, topk), debug=debug)
    oas_info = harvest_oas_whitelist(hits, debug=debug)

    # 3) 非程式題：直接解說
    if mode == "explain":
        prompt = build_prompt(question, hits, oas_info)
        if ctx_only: return prompt
        out = call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)
        cites = format_citations(hits, question)
        return f"{out}\n\n---\n參考資料：\n{cites}"

    # 4) 程式題：沒有 OAS 也不硬產 code → 解說
    if not oas_info:
        prompt = build_prompt(question, hits, None)
        if ctx_only: return prompt
        out = call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)
        cites = format_citations(hits, question)
        return f"{out}\n\n---\n參考資料：\n{cites}"

    # CSV 意圖與年份 hint
    csv_requested = bool(CSV_HINT_RE.search(question))
    years = [m.group(0) for m in YEAR_RE.finditer(question)]
    years_hint = None
    if years:
        y = years[0]
        years_hint = (f"{y}-01-01", f"{y}-12-31")

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
        cites = format_citations(hits, question)
        return f"{out}\n\n---\n參考資料：\n{cites}"

    # 把使用者明確點名的變數寫回 append（白名單約束）
    req_vars = extract_user_requested_vars(question, oas_info.get("append_allowed", []))
    if req_vars:
        pp = plan.setdefault("params", {})
        cur = set()
        if "append" in pp and pp["append"]:
            cur = set([x.strip() for x in str(pp["append"]).split(",") if x.strip()])
        for v in req_vars: cur.add(v)
        if cur:
            pp["append"] = ",".join(sorted(cur))

    # 第一次驗證
    ok, msg = validate_plan(plan, oas_info)
    if debug and not ok:
        print(f"[DEBUG] Plan invalid@first: {msg}")

    # 6) 只有在需要時才呼叫 refine（減少第 3 次 LLM）
    if (not ok) or _needs_refine(plan, req_vars, years_hint):
        plan = llm_refine_plan(question, oas_info, plan, prev_plan, llm, debug=debug)

    # 第二次驗證
    ok, msg = validate_plan(plan, oas_info)
    if debug and not ok:
        print(f"[DEBUG] Plan invalid@refined: {msg}")
    if not ok:
        prompt = build_prompt(question, hits, oas_info)
        out = call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)
        cites = format_citations(hits, question)
        return f"{out}\n\n---\n參考資料：\n{cites}"

    # 保存 plan（供追問承接：如「改成下載 CSV」）
    LAST_PLAN = {"endpoint": plan["endpoint"], "params": dict(plan["params"])}

    # 7) 產 code + 守門
    code = code_from_plan_via_llm(question, plan, llm=llm)
    ok2, msg2 = static_guard_check(code, oas_info)
    if not ok2 and debug:
        print(f"[DEBUG] Code violates guard: {msg2}")
    if not ok2:
        # 最小模板（仍完全遵守 plan）
        endpoint = plan["endpoint"]; params = plan["params"]
        code = (
            "```python\n"
            "# 最小可執行範例：依需求自行調整參數\n"
            "import requests\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import io\n\n"
            f'BASE = "{BASE_URL}"\n'
            f'ENDPOINT = "{endpoint}"\n'
            f"params = {json.dumps(params, ensure_ascii=False, indent=2)}\n\n"
            "r = requests.get(f\"{BASE}{ENDPOINT}\", params=params, timeout=60)\n"
            "r.raise_for_status()\n"
            "ctype = r.headers.get('content-type','')\n"
            "if 'text/csv' in ctype or ENDPOINT.endswith('/csv') or params.get('format')=='csv':\n"
            "    df = pd.read_csv(io.StringIO(r.text))\n"
            "else:\n"
            "    df = pd.DataFrame(r.json())\n"
            "if 'date' in df.columns:\n"
            "    df['date'] = pd.to_datetime(df['date'])\n"
            "print(df.head())\n"
            "```"
        )

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

