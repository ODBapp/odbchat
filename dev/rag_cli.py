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
- 地圖任務守門失敗：提供安全地圖模板 fallback（pcolormesh）
"""

from __future__ import annotations

import os, re, sys, json, argparse, warnings, ast
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import Counter

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
    # sst_anomaly：僅針對「海表」溫度異常、或明確的 “SST anomaly” 字樣
    ("sst_anomaly", r"(海表)?溫.*(距平|異常)|\bSST[-_\s]*anom(al(y|ies))?\b"),
    # level：限定與 MHW/海洋熱浪語境同現，避免一般統計「等級」誤觸
    ("level",       r"(?:\bMHW\b|海洋?熱浪).*(?:等級|level)|(?:等級|level).*(?:\bMHW\b|海洋?熱浪)"),
    # td：熱位移
    ("td",          r"(?:熱位移|thermal\s*displacement)"),
    # sst：限定為海表溫度（SST），避免一般「海溫/海水溫度」概念混入其他 API（如 subsurface）
    ("sst",         r"\bSST\b|海表溫|海水表面溫度|海表.*溫度"),
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
    # 第一次：通用檢索
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

    # 第二次：code/api 偏好（保留原結構與回傳格式）
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

    # 去重（保留既有鍵與策略）
    out, seen = [], set()
    for h in hits:
        p = h.get("payload", {}) or {}
        key = p.get("doc_id") or (p.get("title"), p.get("source_file"))
        if key in seen:
            continue
        seen.add(key)
        out.append(h)

    # ------- 只改這裡：score / re-rank / diversify -------
    ql = (question or "").lower()

    def _norm(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.lower()
        # 常見重音/變體簡化（避免 "niño" 與 "nino" 錯過）
        s = s.replace("niño", "nino").replace("ni\u00f1o", "nino")
        s = s.replace("la niña", "la nina").replace("ni\u00f1a", "nina")
        return s

    # 信號詞（general, 不做「區域 cues」，只做語意關鍵詞對齊）
    SIGNAL_TOKENS = [
        "enso","nino","la nina","el nino",
        "mhw","marine heatwave","marine heatwaves","海洋熱浪","熱浪",
        "sst","anomaly","anomalies","距平","異常",
        "timeseries","time series","時序","趨勢","map","分佈","分布","distribution",
        "api","openapi","oas","endpoint","append","lon","lat","bbox",
    ]
    q_tokens = set(t for t in SIGNAL_TOKENS if _norm(t) in _norm(ql))

    def boost(p: Dict[str,Any], text: str) -> float:
        score = 0.0
        dt = (p.get("doc_type") or "").lower()
        title = (p.get("title") or "")
        tags = p.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        faq  = p.get("faq") or []

        # 基礎：doc_type 權重（適度，不壓死 web_article）
        if dt == "api_spec":
            score += 1.8
        elif dt in ("code_snippet","code_example"):
            score += 1.2
        elif dt == "web_article":
            score += 0.5
        elif dt == "cli_tool_guide":
            # 若題意不是 CLI，略降
            if not re.search(r"\bcli\b|命令列|指令|工具", ql):
                score -= 0.3

        # 標題 / tags / faq 與題意關鍵詞對齊（最多 +2.0）
        aligned = 0.0
        t_all = " ".join([str(title), " ".join(map(str, tags)), " ".join(map(str, faq))])
        nt = _norm(t_all)
        for tok in q_tokens:
            if _norm(tok) in nt:
                aligned += 0.6
        score += min(aligned, 2.0)

        # 內容前段（前 600 字）弱對齊（最多 +1.0）
        head = _norm(str(text)[:600])
        c_aligned = 0.0
        for tok in q_tokens:
            if _norm(tok) in head:
                c_aligned += 0.25
        score += min(c_aligned, 1.0)

        # API/程式實作的「可用性」微調
        if dt == "api_spec" and re.search(r"/api/\w+|paths\s*:", _norm(text)):
            score += 0.3
        if dt in ("code_snippet","code_example") and "requests.get(" in text:
            score += 0.3

        # 過度泛化的文章小扣分（避免長篇泛論壓過 code/api）
        if dt == "web_article" and len(text) > 4000 and not q_tokens:
            score -= 0.2

        # 輕微新鮮度（有 retrieved_at 就給 +0.1）
        if p.get("retrieved_at"):
            score += 0.1

        return score

    scored = []
    for h in out:
        p = h.get("payload", {}) or {}
        t = h.get("text", "") or ""
        scored.append((boost(p, t), h))

    # 移除分數極低（雜訊）
    scored = [x for x in scored if x[0] > -0.5]
    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [h for _, h in scored]

    # 輕度多樣化：保證類型覆蓋（api_spec / code_* / web_article），其餘按 reranked 補齊
    by_type: Dict[str, List[dict]] = {}
    for h in reranked:
        dt = (h.get("payload", {}) or {}).get("doc_type", "").lower()
        by_type.setdefault(dt, []).append(h)

    selected, used_keys = [], set()

    def _push_one(pool_name: str):
        pool = by_type.get(pool_name, [])
        for h in pool:
            p = h.get("payload", {}) or {}
            key = p.get("doc_id") or (p.get("title"), p.get("source_file"))
            if key in used_keys:
                continue
            selected.append(h)
            used_keys.add(key)
            return True
        return False

    # 先保證核心類型各一
    _push_one("api_spec")
    _push_one("code_snippet") or _push_one("code_example")
    _push_one("web_article")

    # 其餘依 reranked 順序補滿 topk
    for h in reranked:
        if len(selected) >= topk:
            break
        p = h.get("payload", {}) or {}
        key = p.get("doc_id") or (p.get("title"), p.get("source_file"))
        if key in used_keys:
            continue
        selected.append(h)
        used_keys.add(key)

    if debug:
        from collections import Counter
        dist = Counter([(h["payload"].get("doc_type"), h["payload"].get("title")) for h in selected])
        print(f"[DEBUG] diversified selection: {dict(dist)}")
        print(f"[DEBUG] total hits(after merge dedupe): {len(out)}, selected: {len(selected)}")

    return selected

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

def run_llm(llm: str, messages: list[dict], timeout: float = 600.0) -> str:
    """
    Minimal chat wrapper that composes a plain prompt from messages and calls local backends.
    - llm: "ollama" uses call_ollama_raw; anything else falls back to call_llamacpp_raw.
    - messages: [{"role":"system"|"user"|"assistant", "content": "..."}]
    """

    # Compose a simple chat-style prompt for local models
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{role}:\n{content}")
        prompt = "\n\n".join(parts).strip()

    llm_id = (llm or "").lower()
    if llm_id == "ollama":
        return call_ollama_raw(prompt, timeout=timeout)
    # default: llama.cpp style
    return call_llamacpp_raw(prompt, timeout=timeout)

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
        "For region-level analysis, prefer a bounding box (two distinct values per axis) rather than a single point. "
        "Spatial-related params defined in OAS/whitelist (e.g., lon/lat or lon0/lat0) are usually required. DO NOT leave it empty or undefined.\n"
        "Temporal-related params guidance:\n"
        "- If the question implies a specific year or range, set temporal-related params (e.g., start/end, or any params whose name relate to date/datetime but use ONLY parameter names present in the whitelist) accordingly (YYYY-MM-DD). "
        "- You may deduce a 12-month range when the user mentions a single year. "
        "- You may deduce to a specific month of a specific year if user ask for map plotting. Fetch once, no need for loops to fetch API data.\n"
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

def extract_code_from_markdown(md: str) -> str:
    """
    更健壯的擷取：優先抓第一個 ```python fence，
    其次抓第一個 ``` 任意語言。若沒有結尾 ```，取到文末。
    最後做一次剝殼與去掉 BOM/隱藏字元。
    """
    if not md:
        return ""
    # 優先 python fence
    m = re.search(r"```(?:python|py)\s*\n([\s\S]*?)\n```", md, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 允許缺少結尾 fence：從第一個 ```python 開始截到文末
    m2 = re.search(r"```(?:python|py)\s*\n([\s\S]*)$", md, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).strip()

    # 其次：任意 ```...``` fence
    m3 = re.search(r"```\s*\n([\s\S]*?)\n```", md)
    if m3:
        return m3.group(1).strip()

    # 允許任意 ``` 開頭到文末
    m4 = re.search(r"```\s*\n([\s\S]*)$", md)
    if m4:
        return m4.group(1).strip()

    # 沒有 fence，只能嘗試把整段當成 code（常見於 LLM 回覆純 code 無 fence）
    return md.strip()


# --------------------
# Code generation & guard
# --------------------
def _check_python_syntax(src: str) -> tuple[bool, str | None]:
    try:
        import ast
        ast.parse(src)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def _looks_like_complete_code(code: str) -> bool:
    """
    不呼叫 LLM 的快速完備度判斷：
    - 語法可解析（ast.parse OK）→ 視為完整。
    - 若解析失敗，但最後一行是未閉合括號/字串，才建議續寫。
    """
    ok, _ = _check_python_syntax(code)
    if ok:
        return True
    # 若你想更嚴格，可補充括號/引號配對檢查，但多半沒有必要。
    return False
    
def _trim_trailing_partial_line(code: str) -> str:
    """
    若最後一行疑似半句（例如不含右括號且包含 '('），就先移除該行，讓續寫器重印正確行。
    避免像 'plt.axhline(-0.5' 直接卡住。
    """
    lines = code.rstrip().splitlines()
    if not lines:
        return code
    last = lines[-1].rstrip()
    # 簡單啟發式：有 '(' 但右括號數量不足
    if last.count("(") > last.count(")") and not last.strip().endswith("\\"):
        # 刪掉最後一行，讓續寫器重印完整版本
        return "\n".join(lines[:-1]).rstrip() + "\n"
    return code

def _build_common_code_sysrule(chunk_line: str | None = None) -> str:
    base = [
        "You are a Python assistant. Return a single runnable script (or a pure continuation if explicitly asked).",
        f"- Use server URL BASE_URL = {BASE_URL}. Use the EXACT endpoint and params provided (whitelist from OAS).",
        "- Call the API with: requests.get(f\"{BASE_URL}{endpoint}\", params=params, timeout=60).",
        "- Do NOT manually join query strings; must use params=dict. Do NOT add headers/API keys/extra params.",
        "- Always parse JSON with r.json() → pandas.DataFrame (NOT pandas.read_json; NOT io.StringIO on r.text).",
        "- Use CSV parsing (pandas.read_csv(io.StringIO(r.text))) ONLY IF the endpoint endswith('/csv') OR params.format=='csv',",
        "  AND only if the user explicitly asked for CSV/download.",
        "- If 'date' exists, convert with pandas.to_datetime(df['date']). For time series, use 'date' on x-axis.",
        "- Decide plot type from intent: time series (trend/變化) vs. map (spatial distribution) of a data variable.",
        "- If you adapt from EXAMPLE snippets in RAG_NOTES, note that IMPORTS ARE REMOVED on purpose — add ONLY the imports you actually need.",
        # 單月地圖不 loop
        "- If the user asks for a spatial distribution at a specific month/day (e.g., '2002-01' or '2002-01-01'), fetch ONCE for that month only; DO NOT loop.",
        # map 正確值對齊方式：pivot 或 分組填格
        "- For map, build a gridded array aligned to lon/lat:",
        "  (1) extract unique sorted lon/lat from df columns (not synthetic linspace),",
        "  (2) create numpy.meshgrid from those unique lon/lat,",
        "  (3) align values to grid cells using either:",
        "      • pivot_table(index='lat', columns='lon', values=VAR, aggfunc='mean'), or",
        "      • explicit fill: create an empty 2D array Z[lat,lon] and fill by iterating df rows to the matching [lat,lon] cell.",
        "  (4) use matplotlib.pyplot.pcolormesh(Lon, Lat, Z, shading='auto') (NEVER scatter).",
        "- Treat 'level' (from ODB MHW API) as categorical; use a discrete colormap (e.g., ['#f5c268','#ec6b1a','#cb3827','#7f1416']).",
        "- Never invent endpoints/params/columns; only use those allowed by OAS and columns present in df.",
        "- Prefer bounded for-loops for chunking (e.g., for year in range(...)); DO NOT write open-ended while loops.",
        "- Include only the imports that are actually used in the generated code. Exclude all unused imports before finalizing."
    ]
    if chunk_line:
        base += [
            "CHUNKING constraint:",
            f"- {chunk_line}",
            "- 分段僅更新 start/end；固定空間參數；每段成功後 append 進 list，最後 pd.concat(ignore_index=True)。",
            "- monthly 建議用 pandas.date_range(..., freq='MS')；yearly/decade 也請用有界 for 迴圈。",
        ]
    return "\n".join(base)

def code_from_plan_via_llm(question: str, plan: dict, llm: str, debug: bool=False) -> str:
    """
    由 LLM 直接產出完整可執行的 Python 程式（優先）。
    - 嚴格遵守 endpoint/params；requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=60)
    - JSON → r.json()；CSV 僅限 /csv 或 format=csv
    - 地圖需先用資料中的 unique lon/lat 建網格，再 pcolormesh；'level' 用離散色盤
    - 注入 OAS CHUNK 限制（逐年/逐月/每10年）
    - 若回覆未閉合 code fence或為空，**只**呼叫一次 code_continue_via_llm
    """

    endpoint = plan["endpoint"]
    params   = plan["params"]

    # 取用全域 RAG hits
    rag_hits = globals().get("LAST_HITS") or []
    rag_notes = collect_rag_notes(rag_hits, max_chars=1000, debug=debug)

    # CHUNKING constraint
    chunk_hint = mhw_span_hint(plan)
    if debug:
        print(f"[DEBUG] chunk_hint: {chunk_hint}")
    chunk_line = ""
    if chunk_hint:
        if chunk_hint.get("mode") == "monthly":
            chunk_line = ("因為此 BBox 超過 90°×90°，依 OAS 只能查 1 個月。\n"
                          "你必須以「逐月分段」迴圈抓取，將每段結果以 pandas.concat 串聯。")
        elif chunk_hint.get("mode") == "yearly":
            chunk_line = ("因為此 BBox 超過 10°×10°，依 OAS 只能查 1 年。\n"
                          "你必須以「逐年分段」迴圈抓取，將每段結果以 pandas.concat 串聯。")
        elif chunk_hint.get("mode") == "decade":
            chunk_line = ("此 BBox 在 10°×10° 以內，依 OAS 最長 10 年。\n"
                          "若時間區間超過 10 年，你必須以「每 10 年」或「逐年」分段，最後以 pandas.concat 串聯。")

    sysrule = _build_common_code_sysrule(chunk_line)
    user = (
        f"QUESTION:\n{question.strip()}\n\n"
        f"ENDPOINT:\n{endpoint}\n\n"
        f"PARAMS:\n{json.dumps(params, ensure_ascii=False, indent=2)}\n\n"
        "RAG_HINTS (僅在需要時可作為註解參考；請依題意調整而非死抄)：\n"
        f"{rag_notes or '(none)'}\n"
        "IMPORTANT: Output ONLY one code block or plain code. No prose."
    )

    messages = [
        {"role": "system", "content": sysrule},
        {"role": "user",   "content": user},
    ]

    try:
        txt = run_llm(llm, messages, timeout=600.0)
    except Exception as e:
        if debug:
            print(f"[DEBUG] Plan LLM failed: {e}")
        return ""

    if debug:
        print(f"[DEBUG] sysrule chars: {len(sysrule)}; user chars: {len(user)}")
        print(f"[DEBUG] LLM raw reply chars: {len(txt or '')}")
        print(f"[DEBUG] LLM raw preview: {(txt or '')[:300].replace(chr(10),'\\n')}")

    code = extract_code_from_markdown(txt)
    if debug:
        print(f"[DEBUG] LLM reply contains code fence: {('```' in (txt or ''))}")
        print(f"[DEBUG] extracted code length: {len(code)}")
    
    unclosed = ("```" in (txt or "")) and not re.search(r"```(?:python|py)?\s*[\s\S]*?```", txt, flags=re.IGNORECASE)
    if unclosed:
        if debug:
            print("[DEBUG] Detected unclosed code fence; checking syntax to decide continue.")
        if _looks_like_complete_code(code):
            if debug:
                print("[DEBUG] Code parses fine; skip auto-continue despite unclosed fence.")
            stitched = code.strip()
        else:
            if debug:
                print("[DEBUG] Code not syntactically complete; will auto-continue once.")
            stitched = code.strip()
            more = code_continue_via_llm(
                question="請從上段未完成處續寫，補足缺漏行，完成繪圖與收尾；不要重複前段程式碼；保持縮排正確一致。",
                plan=plan,
                last_code=stitched,
                llm=llm,
                debug=debug
            )
            if more:
                stitched = stitched.rstrip() + "\n\n" + more.lstrip()
    else:
        stitched = code.strip()

    return stitched

def code_continue_via_llm(question: str, plan: dict, last_code: str, llm: str, debug: bool=False) -> str:
    """
    續寫或修正上一輪程式：
    - 若只要求「繼續/寫完」，輸出「續寫片段」（不得重複前段，保持縮排一致）；
    - 若要求修改參數/邏輯，輸出「完整單檔」；
    - 嚴格遵守 endpoint/params 與 OAS JSON/CSV 規則；
    - 注入 CHUNKING constraint（逐段抓取 + concat）。
    """

    endpoint = plan["endpoint"]
    params   = plan["params"]

    # CHUNKING constraint
    chunk_hint = mhw_span_hint(plan)
    chunk_line = ""
    if chunk_hint:
        if chunk_hint.get("mode") == "monthly":
            chunk_line = ("因為此 BBox 超過 90°×90°，依 OAS 只能查 1 個月。\n"
                          "你必須以「逐月分段」迴圈抓取，將每段結果以 pandas.concat 串聯。")
        elif chunk_hint.get("mode") == "yearly":
            chunk_line = ("因為此 BBox 超過 10°×10°，依 OAS 只能查 1 年。\n"
                          "你必須以「逐年分段」迴圈抓取，將每段結果以 pandas.concat 串聯。")
        elif chunk_hint.get("mode") == "decade":
            chunk_line = ("此 BBox 在 10°×10° 以內，依 OAS 最長 10 年。\n"
                          "若時間區間超過 10 年，你必須以「每 10 年」或「逐年」分段，最後以 pandas.concat 串聯。")

    sysrule = _build_common_code_sysrule(chunk_line)

    user = (
        f"QUESTION:\n{question.strip()}\n\n"
        f"ENDPOINT:\n{endpoint}\n\n"
        f"PARAMS:\n{json.dumps(params, ensure_ascii=False, indent=2)}\n\n"
        "PREVIOUS_SCRIPT (do NOT repeat these lines in continuation mode):\n"
        f"{last_code}\n\n"
        "IMPORTANT:\n"
        "- If user did NOT ask to modify logic/params, output ONLY the missing continuation (plain code or a single code block).",
        "- Do NOT repeat previous lines; keep indentation consistent so the snippet can be concatenated directly.",
        "- If user DID ask to modify, output ONE complete runnable single-file script (no prose)."
    )
    # 把 tuple -> str
    user = "\n".join(user)

    messages = [
        {"role": "system", "content": sysrule},
        {"role": "user",   "content": user},
    ]

    try:
        txt = run_llm(llm, messages, timeout=600.0)
    except Exception as e:
        if debug:
            print(f"[DEBUG] Continue LLM failed: {e}")
        return ""

    cont = extract_code_from_markdown(txt).strip() or (txt or "").strip()

    if debug:
        print(f"[DEBUG] (continue) LLM raw reply chars: {len(txt or '')}")
        print(f"[DEBUG] (continue) LLM raw preview: {(txt or '')[:300].replace(chr(10),'\\n')}...")
        print(f"[DEBUG] (continue) contains code fence: {('```' in (txt or ''))}")
        print(f"[DEBUG] (continue) extracted code length: {len(cont)}")

    return cont

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
        # CSV path or format=csv → 必須 read_csv + io.StringIO
        if ("pd.read_csv(" not in code) or ("io.StringIO(" not in code):
            return False, "CSV mode requires pandas.read_csv(io.StringIO(r.text)) with import io"
    else:
        # JSON path → 嚴禁 read_json / 嚴禁任何 io.StringIO(r.text) / 嚴禁 read_csv(on r.text)
        if re.search(r'pd\.read_json\s*\(', code):
            return False, "JSON must use r.json(); pd.read_json is forbidden"
        if re.search(r'io\.StringIO\s*\(\s*r\.text\s*\)', code, flags=re.I):
            return False, "JSON must not use io.StringIO; parse via r.json()"
        if re.search(r'pd\.read_csv\s*\(\s*io\.StringIO\s*\(\s*r\.text\s*\)\s*\)', code, flags=re.I):
            return False, "JSON must not use pd.read_csv(io.StringIO(r.text)); use r.json() instead"

    # 不得提及 API key/token
    if re.search(r'api[_-]?key|token|authorization|bearer', code, re.I):
        return False, "code mentions API key/token"

    # 禁止 Basemap（避免照抄外部樣本）
    # if re.search(r'mpl_toolkits\.basemap', code):
    #    return False, "Basemap is not allowed"

    # 地圖守門
    if expect_map:
        if re.search(r'\.scatter\s*\(', code):
            return False, "map plotting must not use scatter"
        if "pcolormesh(" not in code and "plt.pcolormesh(" not in code:
            return False, "map plotting must use pcolormesh"
        # 避免合成 linspace 網格（應使用資料中的 lon/lat unique）
        if re.search(r'np\.linspace\s*\(', code):
            return False, "avoid synthetic lon/lat via np.linspace; use unique lon/lat from the data"

    # code fence 完整性（避免截斷）
    if "```python" in code and "```" not in code.strip().split("```python",1)[-1]:
        return False, "code fence likely truncated"

    return True, ""

def build_timeseries_fallback_template(plan: Dict[str, Any], expect_csv: bool) -> str:
    endpoint = plan["endpoint"]
    params = plan["params"]

    lines = [
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
        lines += [
            "import io",
            "df = pd.read_csv(io.StringIO(r.text))",
        ]
    else:
        lines += [
            "df = pd.DataFrame(r.json())  # JSON → DataFrame（禁止 pandas.read_json）",
        ]

    lines += [
        "if 'date' in df.columns:",
        "    df['date'] = pd.to_datetime(df['date'])",
        "",
        "# Fallback 策略：以 append 的第一個變數作為 Y；若沒指定 append，改用第一個非 lon/lat/date 欄位。",
        "Y_COL = None",
        "if isinstance(params.get('append'), str) and params['append'].strip():",
        "    for _v in [v.strip() for v in params['append'].split(',') if v.strip()]:",
        "        if _v in df.columns:",
        "            Y_COL = _v; break",
        "if Y_COL is None:",
        "    for c in df.columns:",
        "        if c not in ('lon','lat','date'):",
        "            Y_COL = c; break",
        "print(df.head())",
        "if Y_COL and 'date' in df.columns:",
        "    df = df.sort_values('date')",
        "    df.plot(x='date', y=Y_COL, figsize=(10,4), title=f'{Y_COL} timeseries')",
        "    plt.tight_layout(); plt.show()",
        ""
    ]
    return "\n".join(lines)

def build_map_fallback_template(plan: Dict[str, Any], expect_csv: bool, month_hint: Optional[str]) -> str:
    endpoint = plan["endpoint"]
    params = plan["params"]

    lines = [
        "# 最小可執行範例（地圖分佈）：以資料中的 lon/lat 建網格 + pcolormesh",
        "import requests",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        f'BASE = "{BASE_URL}"',
        f'ENDPOINT = "{endpoint}"',
        f"params = {json.dumps(params, ensure_ascii=False, indent=2)}",
        "r = requests.get(f\"{BASE}{ENDPOINT}\", params=params, timeout=60)",
        "r.raise_for_status()",
    ]
    if expect_csv:
        lines += [
            "import io",
            "df = pd.read_csv(io.StringIO(r.text))",
        ]
    else:
        lines += [
            "df = pd.DataFrame(r.json())",
        ]

    lines += [
        "if 'date' in df.columns:",
        "    df['date'] = pd.to_datetime(df['date'])",
        "",
        "# 目標月份（若題目含 YYYY-MM 則用之；否則沿用 params['start'] 的月份）",
        f"target_month = {json.dumps(month_hint)}",
        "if target_month is None and 'start' in params:",
        "    try:",
        "        target_month = pd.to_datetime(params['start']).strftime('%Y-%m')",
        "    except Exception:",
        "        target_month = None",
        "",
        "mdf = df.copy()",
        "if target_month and 'date' in mdf.columns:",
        "    mdf = mdf[mdf['date'].dt.strftime('%Y-%m') == target_month].copy()",
        "",
        "# Fallback 變數選擇：以 append 的第一個變數；若無就取第一個非 lon/lat/date 欄位",
        "VAR = None",
        "if isinstance(params.get('append'), str) and params['append'].strip():",
        "    for _v in [v.strip() for v in params['append'].split(',') if v.strip()]:",
        "        if _v in mdf.columns:",
        "            VAR = _v; break",
        "if VAR is None:",
        "    for c in mdf.columns:",
        "        if c not in ('lon','lat','date'):",
        "            VAR = c; break",
        "if VAR is None:",
        "    raise RuntimeError(f\"無可用變數可繪圖；可用欄位: {list(mdf.columns)}\")",
        "",
        "# 規則化成 lon/lat 網格後以 pcolormesh 繪製（pivot）",
        "lons = np.sort(mdf['lon'].unique())",
        "lats = np.sort(mdf['lat'].unique())",
        "grid = mdf.pivot_table(index='lat', columns='lon', values=VAR, aggfunc='mean')",
        "grid = grid.reindex(index=lats, columns=lons)",
        "Lon, Lat = np.meshgrid(lons, lats)",
        "plt.figure(figsize=(10,6))",
        "pcm = plt.pcolormesh(Lon, Lat, grid.values, shading='auto')",
        "plt.colorbar(pcm, label=VAR)",
        "plt.xlabel('Longitude'); plt.ylabel('Latitude')",
        "title = f\"{VAR} distribution\" + (f\" ({target_month})\" if target_month else '')",
        "plt.title(title)",
        "plt.tight_layout(); plt.show()",
        ""
    ]
    return "\n".join(lines)

def _extract_title_from_yaml_front_matter(text: str) -> str | None:
    """
    從 YAML front-matter 或任意 YAML 片段取出 title。
    支援：
      - 有 '---' 的 front-matter 區塊
      - 沒有 '---' 時，掃描前幾十行找 'title:' 行
    """
    if not isinstance(text, str) or not text.strip():
        return None

    s = text.lstrip()
    # 情況 A：沒有 '---'，就直接在全文找 title:
    if not s.startswith('---'):
        m_any = re.search(r'^\s*title\s*:\s*(.+)$', text, flags=re.MULTILINE)
        if m_any:
            raw = m_any.group(1).strip()
            if len(raw) >= 2 and raw[0] in ("'", '"') and raw[-1] == raw[0]:
                raw = raw[1:-1]
            return raw or None
        return None

    # 情況 B：有 '---'，擷取第一段 header
    lines = s.splitlines()
    i = 1  # 跳過第一個 '---'
    buf = []
    while i < len(lines):
        line = lines[i]
        if line.strip() == '---':
            break
        # 有些檔案以 content: | 當成 header 與 body 的分水嶺
        if re.match(r'^\s*content\s*:\s*\|', line):
            break
        buf.append(line)
        i += 1

    header = "\n".join(buf)
    m = re.search(r'^\s*title\s*:\s*(.+)$', header, flags=re.MULTILINE)
    if not m:
        return None
    raw = m.group(1).strip()
    if len(raw) >= 2 and raw[0] in ("'", '"') and raw[-1] == raw[0]:
        raw = raw[1:-1]
    return raw or None


def _type_of(hit: dict) -> str:
    """
    推斷 doc_type/type，依序檢查：
      1) 頂層 doc_type/type
      2) payload (dict) 裡的 doc_type/type
      3) payload (str/YAML) 的 'doc_type:'
      4) 否則回傳 'note'
    """
    hit = hit or {}
    t = (hit.get("doc_type") or hit.get("type") or "").strip().lower()
    if t:
        return t
    payload = hit.get("payload")
    if isinstance(payload, dict):
        t = (payload.get("doc_type") or payload.get("type") or "").strip().lower()
        if t:
            return t
    elif isinstance(payload, str):
        m = re.search(r'^\s*doc_type\s*:\s*([A-Za-z0-9_+-]+)\s*$', payload, flags=re.MULTILINE)
        if m:
            return m.group(1).strip().lower()
    return "note"


def _best_title(hit: dict, idx: int, default: str = "") -> tuple[str, str]:
    """
    取得最佳標題與其來源（source），回傳 (title, source)。
    優先序：
      1) 頂層：title/doc_title/name/source_title/filename/basename/path/url/id
      2) 巢狀 meta/front/metadata/front_matter: {title}
      3) payload=dict 的 title
      4) payload=str 的 YAML 片段
      5) content/text 的 YAML 片段
      6) fallback: "<doc_type> #idx"
    """
    hit = hit or {}
    # 1) 平面鍵
    flat_keys = ("title", "doc_title", "name", "source_title", "filename", "basename", "path", "url", "id")
    for k in flat_keys:
        v = hit.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip(), f"flat.{k}"

    # 2) 巢狀鍵
    for nk in ("meta", "front", "metadata", "front_matter"):
        node = hit.get(nk)
        if isinstance(node, dict):
            v = node.get("title")
            if isinstance(v, str) and v.strip():
                return v.strip(), f"nested.{nk}.title"

    # 3) payload=dict
    payload = hit.get("payload")
    if isinstance(payload, dict):
        v = payload.get("title")
        if isinstance(v, str) and v.strip():
            return v.strip(), "payload.title"

    # 4) payload=str (YAML/原文)
    if isinstance(payload, str) and payload.strip():
        t = _extract_title_from_yaml_front_matter(payload)
        if t:
            return t.strip(), "payload.yaml"

    # 5) content/text 的 YAML front matter
    content = hit.get("content") or hit.get("text") or ""
    if isinstance(content, str) and content.strip():
        t = _extract_title_from_yaml_front_matter(content)
        if t:
            return t.strip(), "content.yaml"

    # 6) fallback
    htype = _type_of(hit)
    title = default or f"{htype} #{idx+1}"
    return title, "fallback"


def collect_rag_notes(
    hits: list,
    max_chars: int = 1200,
    code_snippet_char_limit: int = 500,
    debug: bool = False
) -> str:
    """
    收集 RAG 片段：
      - code_snippet：抽 code fence → 移除所有 imports → 截斷 → 以 ```text 包住
      - 其他：擷取前 ~400 字的摘要
      - 標題：用 _best_title (會回傳來源標示)
      - debug：列出 type、title、title_source、keys、content_len
    """
    def _strip_leading_imports(code: str) -> str:
        kept = []
        for ln in (code or "").splitlines():
            if re.match(r"^\s*(import|from)\s+\S+", ln):
                continue
            kept.append(ln)
        return "\n".join(kept).strip()

    notes_parts, titles = [], []
    used = 0

    if debug:
        print(f"[DEBUG] RAG hits total: {len(hits or [])}; type distribution: n/a")

    for idx, h in enumerate(hits or []):
        if used >= max_chars:
            break
        h = h or {}
        htype = _type_of(h)
        title, title_src = _best_title(h, idx, default=htype)
        raw_content = h.get("content") or h.get("text") or ""

        # 若 content/text 為空且 payload 是 dict，試著拿 payload.content
        if (not raw_content) and isinstance(h.get("payload"), dict):
            raw_content = h["payload"].get("content", "")

        # 確保是字串
        if not isinstance(raw_content, str):
            try:
                raw_content = json.dumps(raw_content, ensure_ascii=False)
            except Exception:
                raw_content = str(raw_content)

        if debug:
            keys_preview = sorted([k for k in h.keys() if isinstance(k, str)])
            print(f"[DEBUG] RAG pick[{idx}]: type={htype!r}, title={title!r}, title_src={title_src}, keys={keys_preview}, content_len={len(raw_content)}")

        block = ""
        if htype == "code_snippet":
            m = re.search(r"```(?:python|py)?\s*\n([\s\S]*?)\n```", raw_content, flags=re.IGNORECASE)
            if not m:
                m = re.search(r"```\s*\n([\s\S]*?)\n```", raw_content, flags=re.IGNORECASE)
            code = (m.group(1).strip() if m else raw_content.strip())

            code = _strip_leading_imports(code)
            if not code:
                if debug:
                    print(f"[DEBUG] RAG pick[{idx}]: code empty after import stripping → skipped")
                continue
            code = code[: min(code_snippet_char_limit, max_chars - used)]

            block = (
                f"[code_snippet] {title}\n"
                "EXAMPLE (imports removed — adapt patterns; add only the imports you actually need):\n"
                "```text\n" + code + "\n```"
            )
        else:
            # 文字摘要
            excerpt = re.sub(r"\s+", " ", raw_content).strip()
            if not excerpt:
                if debug:
                    print(f"[DEBUG] RAG pick[{idx}]: empty excerpt → skipped")
                continue
            excerpt = excerpt[: min(400, max_chars - used)]
            block = f"[{htype}] {title}\n{excerpt}"

        if not block:
            continue

        blk_len = len(block)
        if used + blk_len > max_chars:
            block = block[: max_chars - used]
            blk_len = len(block)

        notes_parts.append(block)
        titles.append(title)
        used += blk_len

    header = (
        "[RAG NOTES]\n"
        "EXAMPLE snippets below have IMPORTS REMOVED on purpose.\n"
        "They are reference patterns — do NOT copy verbatim.\n"
        "Add ONLY the imports you actually need when adapting.\n\n"
    )
    notes = header + "\n\n".join(notes_parts)

    if debug:
        print(f"[DEBUG] RAG notes selected: {len(notes_parts)} blocks; total chars={used}; titles={titles}")
    return notes

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
    globals()["LAST_HITS"] = hits

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
        code = code_continue_via_llm(question, plan, LAST_CODE_TEXT or "", llm=llm, debug=debug) or ""
    else:
        code = code_from_plan_via_llm(question, plan, llm=llm, debug=debug) or ""

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
