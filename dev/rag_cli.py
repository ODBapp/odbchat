#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
rag_cli.py — ODBchat RAG CLI
Goal:
- 讓 LLM 在 OAS 白名單下自行推論 endpoint 與參數（含空間範圍/格式），避免硬編對照表。
- 先由 LLM 判斷本題「需要程式」或「不需要程式」；若使用者明說不要程式/除了程式之外，就走解說流。
- CSV：不硬指定 /csv；要求 LLM 根據 OAS 選擇「含 csv 的 endpoint」或「format=csv（若 OAS 允許）」。
- append：若使用者同時點名多個變數，請 LLM 全納入；並以白名單檢核。
- 嚴禁虛構參數/端點；程式碼必須用 requests.get(..., params=...)；禁止手刻 query string。
- 正文用繁體中文，避免簡體用語；正文不使用 [1][2] 編號，參考資料統一列在文末。
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
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "600"))

LLAMA_URL      = os.environ.get("LLAMA_URL", "http://localhost:8001/completion")
LLAMA_TIMEOUT  = float(os.environ.get("LLAMA_TIMEOUT", "600"))

BASE_URL       = "https://eco.odb.ntu.edu.tw"

# --------------------
# Init
# --------------------
qdrant   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

# 全域保存上一題通過驗證的 plan（用於 follow-up 承接）
LAST_PLAN: Optional[Dict[str,Any]] = None

# --------------------
# Intent heuristics（極輕量，主要仍交給 LLM 判斷）
# --------------------
CODE_HINTS  = ["python","程式","code","範例","example","sample code","示範","下載","抓資料","畫圖","plot","時序","時間序列","繪圖"]
NO_CODE_NEG = ["不要程式","不要用程式","除了程式","非程式","不用 code","不要 code","不是程式","不用寫程式"]

CSV_HINT_RE = re.compile(r"\bcsv\b|下載|匯出|存檔", re.I)
YEAR_RE     = re.compile(r"\b(19|20)\d{2}\b")

def likely_code_intent(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in CODE_HINTS)

def negative_code_intent(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in [t.lower() for t in NO_CODE_NEG])

# --------------------
# Basic helpers
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

    # 簡單 rerank（偏好 api/code，但不剔除其他）
    ql = question.lower()
    def boost(p: Dict[str,Any]) -> float:
        score = 0.0
        dt = (p.get("doc_type") or "").lower()
        title = (p.get("title") or "").lower()

        # 正向：API/Code 類
        if dt in ("api_spec","code_snippet","code_example"):
            score += 1.5

        # CLI 手冊只有在題目提到 CLI/命令列/工具時才加分，否則略微降權
        if dt == "cli_tool_guide":
            if re.search(r"\bcli\b|命令列|指令|工具", ql):
                score += 0.6
            else:
                score -= 0.4

        # 與 ENSO 關聯：只有問題未涉及 ENSO/Niño/La Niña 才施加負分
        if ("enso" in title) and not re.search(r"enso|niño|la\s*niña", ql):
            score -= 0.8

        return score

    scored = [(boost(h["payload"]), h) for h in out]
    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [h for _,h in scored]

    # 多樣化
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
# OAS parsing / harvest
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
            for p in (y.get("paths") or {}): paths_list.append(p)
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

    # regex fallback（盡力而為）
    for p in re.findall(r'(?m)^\s*(/api/[^\s:]+)\s*:', raw): paths_list.append(p.strip())
    for p in re.findall(r'(?m)^\s*-\s*name\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$', raw): params_set.add(p.strip())
    m = _ALLOWED_DESC_RE.search(raw)
    if m:
        for g in m.groups():
            if g: append_set.add(g)

    # cast to lists
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
    texts = _collect_api_specs_from_hits(hits)
    P,A,Paths = set(), set(), set()
    Enums: Dict[str,List[str]] = {}
    for raw in texts:
        meta = _parse_oas_text(raw)
        P.update(meta["params"])
        A.update(meta["append_allowed"])
        Paths.update(meta["paths"])
        for k,v in (meta["param_enums"] or {}).items():
            Enums.setdefault(k, [])
            for x in v:
                if x not in Enums[k]: Enums[k].append(x)
    if (len(P)<=1) or (not Paths):
        for raw in _collect_api_specs_full_scan(limit=1000):
            meta = _parse_oas_text(raw)
            P.update(meta["params"])
            A.update(meta["append_allowed"])
            Paths.update(meta["paths"])
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

    if not P and not Paths: return None
    return {"params": sorted(P), "append_allowed": sorted(A), "paths": sorted(Paths), "param_enums": Enums}

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
# Mode selection by LLM ("code" or "explain")
# --------------------
def llm_choose_mode(question: str, llm: str) -> str:
    sysrule = (
        "You are an API planner for ODB MHW API. "
        "Your goal is to select exactly ONE endpoint and a parameter dict that are BOTH strictly within the OAS whitelists provided. "
        "Do NOT invent endpoints or parameters.\n\n"
        "Spatial & temporal guidance:\n"
        "- If the question mentions a region or place, estimate an appropriate spatial constraint using available geographic parameters from the whitelist "
        "(e.g., any params whose names relate to lon/lat/bbox/coordinates), but use ONLY parameter names present in the whitelist. "
        "For region-level analysis, prefer a bounding box (two distinct values per axis) rather than a single point.\n"
        "- If the question implies a specific year or range, set start/end accordingly (YYYY-MM-DD). If the user mentions a single year, you may set it to that year's full range.\n\n"
        "Output format guidance:\n"
        "- Prefer a JSON endpoint for programmatic analysis by default. Only choose a CSV endpoint if the user explicitly requests CSV/download OR if OAS provides only a CSV path. "
        "- If the user asks for CSV and no CSV endpoint exists, check whether a 'format' parameter (enum) allows 'csv'. If yes, set format='csv'. Otherwise fall back to JSON.\n\n"
        "append guidance:\n"
        "- If the user explicitly requested certain data variables (provided below as 'user_append_hints'), include ALL of them in 'append' (comma-separated) IF and ONLY IF 'append' exists in the whitelist and each value is allowed by the OAS.\n\n"
        "If a previous plan is provided, inherit its parameter values unless the user explicitly asked to change them; only modify what is required by the new question."
    )
    prompt = f"{sysrule}\n\nQuestion:\n{question}\n\nAnswer with only one token: code or explain"
    try:
        raw = call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)
        ans = raw.strip().lower()
        if "explain" in ans: return "explain"
        if "code" in ans: return "code"
        # fallback
        return "code" if likely_code_intent(question) else "explain"
    except Exception:
        return "code" if likely_code_intent(question) else "explain"

# --------------------
# Planning (LLM) → validated plan
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
        "You are an API planner for ODB APIs (e.g., MHW API) which complies with OpenAPI Specifications(OAS). "
        "Your goal is to select exactly ONE endpoint and a parameter dict that are BOTH strictly within the OAS whitelists provided. "
        "Do NOT invent endpoints or parameters.\n\n"
        "Spatial-related params guidance:\n"
        "- If the question mentions a region or place, estimate an appropriate spatial range or constraint for that region/place by using available geographic parameters from the whitelist (e.g., any params whose names relate to lon/lat, or bbox: lon0,lat0,lon1,lat1 which specify longitude/latitude range), but use ONLY parameter names present in the whitelist.\n"
        "Temporal-related params guidance:\n"       
        "- If the question implies a specific year or range, set temporal-related params (e.g., start/end, or any params whose name relate to date or datetime but use ONLY parameters names present in the whitelist) accordingly (YYYY-MM-DD). You may deduce a 12-month range when the user mentions a single year.\n\n"
        "Endpoint or format-related query params for CSV request guidance:\n"
        "- First prefer an endpoint that output JSON response if the user does not explicitly asks for CSV or file download."
        "- If the user explicitly asks for CSV or file download, then return an endpoint path that clearly indicates CSV (e.g., contains '/csv') if such an endpoint exists in the whitelist.\n"
        "- Otherwise, if a 'format' parameter exists in the whitelist and its enum includes 'csv', set format='csv'.\n"
        "- Otherwise, return an endpoint that output JSON response.\n\n"
        "The query parameter: append usage guidance:\n"
        "- If the user explicitly requested certain data variables (provided below as 'user_append_hints'), include ALL of them in 'append' (comma-separated) IF and ONLY IF 'append' exists in the whitelist and each value is allowed by the OAS.\n\n"
        "If a previous plan is provided, inherit its parameter values unless the user explicitly asked to change them; only modify what is required by the new question."
    )

    prev_line = f"Previous plan:\n{json.dumps(prev_plan, ensure_ascii=False)}" if prev_plan else "Previous plan: (none)"
    csv_line  = f"User {'' if bool(csv_requested) else 'NOT'} requested CSV in endpoint or query params. "
    years_line= f"Years hint: {json.dumps(years_hint)}"
    vars_line = f"user_append_hints: {json.dumps(user_append_hints or [], ensure_ascii=False)}"
    if debug: print(f"[DEBUG] rule for plan, allowed_paths: {allowed_paths}, and prev: {prev_line} , and csv: {csv_line} , and years: {years_line} , and vars: {vars_line}")

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
        raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.I)
        plan = json.loads(raw)
        if debug: print(f"[DEBUG] Plan(raw): {raw}")
        return plan
    except Exception as e:
        if debug: print(f"[DEBUG] Plan LLM failed: {e}")
        return None

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

    # enum 檢查（例如 format=csv）
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
        "Otherwise, the response should be in JSON, and use try-except to parse JSON response into a pandas DataFrame to manipulate the fetched data. "
        "All comments (if any) must be in Traditional Chinese or English."
        "Keep the code minimal and runable. If plotting or analyzing data is required in the question, provide an appropriate plot code for the fetched data"
    )

    prompt = f"""{sysrule}

User question:
{question}

Base URL: {BASE_URL}
Endpoint: {endpoint}
Params JSON (use exactly as-is):
{params_json}

Wrap the script in a single ```python code fence. Keep it short.
"""
    try:
        return (call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)).strip()
    except Exception as e:
        return f"[LLM error during code generation] {e}"

def static_guard_check(code: str, oas: Dict[str,Any]) -> Tuple[bool, str]:
    # endpoint 白名單
    endpoints = set(oas.get("paths", []))
    used_paths = set(re.findall(r'["\'](/api/[^"\']+)["\']', code))
    for p in used_paths:
        if p not in endpoints:
            return False, f"code uses endpoint '{p}' not in whitelist"

    # 禁手刻 query string
    if re.search(r'\?.*=', code) and "params=" not in code:
        return False, "code manually concatenates query string"

    # 必須使用 params= 呼叫
    if "requests.get(" in code and "params=" not in code:
        return False, "requests.get is used without params= dict"

    # CSV 端點不得 read_json
    if any(p.endswith("/csv") for p in used_paths) and re.search(r'\bread_json\b', code):
        return False, "CSV endpoint must not use read_json"

    # 檢查 params 鍵與 append 多值合法性
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

    # 不得提 API key/token
    if re.search(r'api[_-]?key|token|authorization|bearer', code, re.I):
        return False, "code mentions API key/token"
    
    # 若 endpoint 是 /csv 或 params format='csv'，必須以 io.StringIO(r.text) 給 pandas.read_csv
    is_csv_mode = any(p.endswith("/csv") for p in used_paths) or re.search(r"params\.get\(\s*['\"]format['\"]\s*\)\s*==\s*['\"]csv['\"]", code)
    if is_csv_mode:
        if "pd.read_csv(" in code and "io.StringIO(" not in code:
            return False, "CSV must be read via pandas.read_csv(io.StringIO(r.text)); do not pass raw bytes/strings as file path"
        if "import io" not in code:
            return False, "CSV mode requires 'import io' for io.StringIO"    

    return True, ""

# --------------------
# Build non-code prompt
# --------------------
def build_prompt(question: str, ctx: List[Dict[str, Any]], oas_info: Optional[Dict[str,Any]]) -> str:
    sys_rule = (
        "你是 ODB（海洋學門資料庫）助理。請只根據『依據』內容作答，允許語意近似（例如「海洋系統≈海洋生態系統」），"
        "但科學內容需審慎、可驗證。嚴禁自創或改造學術名詞；若不確定，請改用標準術語（如「聖嬰（El Niño）」「反聖嬰（La Niña）」「ENSO」）。"
        "若提及 ODB API 的端點或參數，必須嚴格遵守 OpenAPI specification (OAS) 白名單（端點/方法/參數不可虛構）。"
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
    # 不在正文放編號，這裡列出最相關的 5 則來源
    items=[]; seen=set()
    for h in hits:
        p=h.get("payload",{}) or {}
        t,u = get_title_url(p); k=(t,u)
        if k in seen: continue
        seen.add(k)
        items.append(f"- {t} — {u}" if u else f"- {t}")
        if len(items)>=5: break
    return "\n".join(items) if items else "（無）"

# --- Minimal variable hints (not a big lexicon; just 4 robust cues) ---
VAR_SYNONYMS = [
    ("sst_anomaly", r"距平|異常|anomal(y|ies)"),
    ("level",       r"\blevel\b|等級|分級|級數"),
    ("td",          r"\btd\b|熱位移|thermal\s*displacement"),
    ("sst",         r"\bsst\b|海溫|海表溫|海水表面溫度|sea\s*surface\s*temperature"),
]

def extract_user_requested_vars(question: str, allowed: List[str]) -> List[str]:
    out = []
    q = question.lower()
    for var, rx in VAR_SYNONYMS:
        if re.search(rx, q, re.I) and (not allowed or var in allowed):
            out.append(var)
    return sorted(set(out))

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

    # 先由 LLM 判斷模式；若使用者明說不要程式，直接走 explain
    mode = llm_choose_mode(question, llm=llm)
    if negative_code_intent(question):
        mode = "explain"

    # 檢索
    hits = query_qdrant(question, topk=max(6, topk), debug=debug)

    # OAS
    oas_info = harvest_oas_whitelist(hits, debug=debug)

    # 非程式題 → 直接總結（禁止虛構 API）
    if mode == "explain":
        prompt = build_prompt(question, hits, oas_info)
        if ctx_only: return prompt
        out = call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)
        cites = format_citations(hits, question)
        return f"{out}\n\n---\n參考資料：\n{cites}"

    # 程式題：沒有 OAS 也不強產 code（避免亂寫）
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

    # 由 LLM 產生計畫（可承接上一題 plan；append 由 LLM 決定且需包含使用者明說的變數）
    prev_plan = LAST_PLAN  # 若無 follow-up 要求，LLM 仍可視為參考
    plan = llm_json_plan(
        question=question,
        oas=oas_info,
        llm=llm,
        prev_plan=prev_plan,
        debug=debug,
        csv_requested=csv_requested,
        years_hint=years_hint,
        user_append_hints=None  # 交給 LLM 從問題語意判斷，不再硬寫字典
    )

    if not plan:
        # 讓 LLM 失敗時，退回解說模式，避免輸出錯 code
        prompt = build_prompt(question, hits, oas_info)
        out = call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)
        cites = format_citations(hits, question)
        return f"{out}\n\n---\n參考資料：\n{cites}"

    # -- 把使用者點名的變數全部納入 append（仍受 OAS 白名單約束） --
    req_vars = extract_user_requested_vars(question, oas_info.get("append_allowed", []))
    if req_vars:
        pp = plan.setdefault("params", {})
        cur = set()
        if "append" in pp and pp["append"]:
            cur = set([x.strip() for x in str(pp["append"]).split(",") if x.strip()])
        for v in req_vars:
            cur.add(v)
        if cur:
            pp["append"] = ",".join(sorted(cur))

    ok, msg = validate_plan(plan, oas_info)
    if not ok and debug:
        print(f"[DEBUG] Plan invalid: {msg}")
    if not ok:
        # 再退回解說模式
        prompt = build_prompt(question, hits, oas_info)
        out = call_ollama_raw(prompt) if llm=="ollama" else call_llamacpp_raw(prompt)
        cites = format_citations(hits, question)
        return f"{out}\n\n---\n參考資料：\n{cites}"

    # 計畫通過 → 保存供下一題承接
    LAST_PLAN = {"endpoint": plan["endpoint"], "params": dict(plan["params"])}

    # 交給 LLM 產 code + 守門
    code = code_from_plan_via_llm(question, plan, llm=llm)
    ok2, msg2 = static_guard_check(code, oas_info)
    if not ok2 and debug:
        print(f"[DEBUG] Code violates guard: {msg2}")

    if not ok2:
        # 以最小模板回覆（仍完全遵守 plan）
        endpoint = plan["endpoint"]; params = plan["params"]
        code = (
            "```python\n"
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
