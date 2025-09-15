#!/usr/#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
rag_cli.py — ODBchat RAG CLI (rev: 2025-09-07)
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
- 本版更新：
  * Rerank/多樣化：移除 doc_type 基礎加分；以語意對齊為主，保持兩段檢索結構不變。
  * static_guard_check() 柔化：移除對 np.linspace 的強制阻擋（改由提示引導），其餘硬性規則保留。
"""

from __future__ import annotations

import os, re, json, argparse, warnings, ast, sys
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
from collections import Counter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import requests

warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

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
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "gemma3:12b") #gemma3:27b #"gpt-oss:20b"
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "720"))

LLAMA_URL      = os.environ.get("LLAMA_URL", "http://localhost:8201/completion")
LLAMA_TIMEOUT  = float(os.environ.get("LLAMA_TIMEOUT", "300"))

ECO_BASE_URL   = "https://eco.odb.ntu.edu.tw"

# 選擇性風險修補（避免變慢，預設關閉；設為 '1' 以啟用）
LLM_RISK_VALIDATE = os.environ.get("LLM_RISK_VALIDATE", "0") == "1"

# --------------------
# Init
# --------------------
qdrant   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

# 會話態
LAST_PLAN: Optional[Dict[str,Any]] = None
OAS_CACHE: Optional[Dict[str,Any]] = None
LAST_CODE_TEXT: Optional[str] = None  # 用於「續寫/補完」情境

# --- OAS fallback cache (only used if a function wasn't passed `oas`) ---
def _set_oas_cache(oas: dict | None) -> None:
    globals()["_OAS_CACHE"] = (oas or {}).copy()

def _get_oas_cache() -> dict:
    return (globals().get("_OAS_CACHE") or {}).copy()

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

# ---- Common regex & helpers (shared by one-pass & multi-steps) ----
CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*[\s\S]*?```", flags=re.IGNORECASE)
ONE_PASS_MARKER_RE = re.compile(r'<<<[^>]+>>>', flags=re.IGNORECASE)
EXPLAIN_TOKEN_RE   = re.compile(r'\{+\s*explain\s*\}+', flags=re.IGNORECASE)
CODE_LIKE_RE = re.compile(r'\b(import|from|def |class |requests\.get|pd\.|plt\.)\b')
MAP_LIKE_RE  = re.compile(r'\b(pcolormesh|imshow|meshgrid|cartopy|Basemap)\b', re.I)
TS_LIKE_RE   = re.compile(r'\bplot\s*\(|rolling\(|resample\(|to_datetime\(|groupby\(\s*["\']date', re.I)
CJK_LINE_RE  = re.compile(r'^[^\S\r\n]*[^\x00-\x7F]+.*$')   # 含 CJK 的整行（不含註解）
JSON_PLAN_ANY_RE = re.compile(r'\{\s*"endpoint"\s*:\s*"/api/[^"]+"\s*,\s*"params"\s*:\s*\{[\s\S]*?\}\s*\}', re.I)

# --- RAG helpers (doc_type normalize & OAS guarantee) ---

_DOC_TYPE_ALIASES = {
    "api", "oas", "openapi", "swagger", "api_specification", "openapi_spec", "openapi-3.1"
}

def _norm_doc_type(dt: str | None) -> str:
    if not isinstance(dt, str):
        return "n/a"
    s = dt.strip().lower()
    if s in _DOC_TYPE_ALIASES:
        return "api_spec"
    # 常見別名歸一
    if s.replace("-", "_") in {"api_spec", "code_snippet", "cli_tool_guide",
                               "web_article", "paper_note", "note"}:
        return s.replace("-", "_")
    return s

def _has_type(hits: list[dict], type_name: str) -> bool:
    for h in hits or []:
        p = (h.get("payload") or {})
        t = _norm_doc_type(p.get("doc_type"))
        if t == type_name:
            return True
    return False

def _force_include_api_spec(all_hits: list[dict], selected: list[dict]) -> tuple[list[dict], bool]:
    """若 selected 裡沒有 api_spec，但 all_hits 裡有，則強制補進第一篇；回傳 (selected, forced_flag)。"""
    if _has_type(selected, "api_spec"):
        return selected, False

    cand = None
    # 先從 all_hits 找 doc_type=api_spec
    for h in all_hits or []:
        p = (h.get("payload") or {})
        if _norm_doc_type(p.get("doc_type")) == "api_spec":
            cand = h; break

    # 次佳：找 title/source_file 含 swagger/openapi/oas
    if cand is None:
        for h in all_hits or []:
            p = (h.get("payload") or {})
            title = (p.get("title") or p.get("doc_id") or p.get("source_file") or "").lower()
            if any(k in title for k in ("swagger", "openapi", "oas", "api spec")):
                cand = h; break

    if cand is not None:
        # 插到最前面，並避免重複 (以 (doc_id) 或 (title, source_file) 去重)
        used_keys = set()
        def key_of(x):
            px = (x.get("payload") or {})
            return px.get("doc_id") or (px.get("title"), px.get("source_file"))
        used_keys.add(key_of(cand))
        out = [cand]
        for h in selected:
            k = key_of(h)
            if k not in used_keys:
                used_keys.add(k); out.append(h)
        return out, True

    return selected, False

def _strip_leading_punct(s: str) -> str:
    if not s: return s
    return re.sub(r'^[\s\.\,;:!?？、。，【】（）()「」『』《》]+', '', s)

def _ensure_fenced(code: str) -> str:
    c = (code or "").strip()
    if not c:
        return c
    if CODE_FENCE_RE.search(c):
        return c
    return f"```python\n{c}\n```"

def merge_code_fences(*parts: str) -> str:
    body = "\n".join([extract_code_from_markdown(p or "") for p in parts if p]).strip()
    return f"```python\n{body}\n```" if body else ""

def _clean_explain_text(txt: str) -> str:
    # 移掉任何 code fence / one-pass marker / {explain}
    s = re.sub(r'```[\s\S]*?```', ' ', txt or '')
    s = ONE_PASS_MARKER_RE.sub(' ', s)
    s = EXPLAIN_TOKEN_RE.sub(' ', s)
    # 保留 markdown 的基本結構：只做多餘空白收斂，**不破壞換行**
    # 把多行的超長連續空白縮成 1 個空白，但**保留換行**
    s = re.sub(r'[ \t]+', ' ', s)
    # 乾掉前導標點
    s = _strip_leading_punct(s)
    return s.strip()

def _extract_code_strict(txt: str) -> str:
    raw_code = extract_code_from_markdown(txt).strip()
    if not raw_code:
        # 與 one-pass 一致：去掉 <<<...>>> 再嘗試
        stripped = ONE_PASS_MARKER_RE.sub('', txt or '')
        raw_code = extract_code_from_markdown(stripped).strip() or stripped.strip()
    return _ensure_fenced(raw_code)

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

def _mhw_chunk_general_rule() -> str:
    """
    一般性（不看 BBox or 年限）的 MHW chunk 提醒，讓 one-pass LLM 也知道限制。
    真正的精確提示仍在 multi-step 用 mhw_span_hint(plan) 產生 chunk_line。
    """
    return (
        "If you choose the ODB MHW API ('/api/mhw' or '/api/mhw/csv'), enforce chunking limits by the following three rules: "
        "(1) If the bounding box is larger than 90° × 90°, fetch at MOST one MONTH per request, loop monthly, and pandas.concat the pieces. "
        "(2) If the bounding box is larger than 10° × 10°, fetch at MOST one YEAR per request, loop yearly, and pandas.concat the pieces. "
        "(3) If the bounding box is within 10° × 10°, you may fetch up to 10 YEARS per request; if the range is longer, split by decade or year, then concat.\n"
    )

def build_chunk_line_from_plan(plan: dict | None) -> str:
    """
    用目前的 plan 推導具體 chunk_line；沒有命中則回空字串。
    """
    if not plan or not isinstance(plan, dict):
        return ""
    hint = mhw_span_hint(plan) if 'mhw_span_hint' in globals() else None
    if not hint:
        return ""
    mode = hint.get("mode")
    if mode == "monthly":
        return ("因為此 BBox 超過 90°×90°，依 OAS 只能查 1 個月。\n"
                "你必須以「逐月分段」迴圈抓取，將每段結果以 pandas.concat 串聯。")
    if mode == "yearly":
        return ("因為此 BBox 超過 10°×10°，依 OAS 只能查 1 年。\n"
                "你必須以「逐年分段」迴圈抓取，將每段結果以 pandas.concat 串聯。")
    if mode == "decade":
        return ("此 BBox 在 10°×10° 以內，依 OAS 最長 10 年。\n"
                "若時間區間超過 10 年，你必須以「每 10 年」或「逐年」分段，最後以 pandas.concat 串聯。")
    return ""

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
        models.FieldCondition(key="doc_type", match=models.MatchValue(value="api_spec")),
        models.FieldCondition(key="doc_type", match=models.MatchValue(value="code_snippet")),
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

    # ------- 僅此處：score / re-rank / diversify（移除 doc_type 加分） -------
    ql = (question or "").lower()

    def _norm(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.lower()
        s = s.replace("niño", "nino").replace("ni\u00f1o", "nino")
        s = s.replace("la niña", "la nina").replace("ni\u00f1a", "nina")
        return s

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
        title = (p.get("title") or "")
        tags = p.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        faq  = p.get("faq") or []

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
        if re.search(r"/api/\w+|paths\s*:", _norm(text)):
            score += 0.15
        if "requests.get(" in text:
            score += 0.15

        # 過度泛化的長文小扣分（避免壓過 code/api）
        if len(text) > 4000 and not q_tokens:
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

    scored = [x for x in scored if x[0] > -0.5]
    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [h for _, h in scored]

    # 輕度多樣化：保證類型覆蓋（api_spec / code_* / web_article），其餘按 reranked 補齊
    by_type: Dict[str, List[dict]] = {}
    for h in reranked:
        dt = _norm_doc_type((h.get("payload", {}) or {}).get("doc_type"))
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
        dist = Counter([(h["payload"].get("doc_type"), h["payload"].get("title")) for h in selected])
        print(f"[DEBUG] diversified selection: {dict(dist)}")
        # ---- 保證至少一篇 api_spec ----
        # selected_before = list(selected)
        selected, forced_oas = _force_include_api_spec(out, selected)
        if forced_oas:
            if debug:
                print("[DEBUG] diversify | api_spec missing → forced insert from pool")
            # 更新 by_type 統計顯示
            # by_type = {}
            # for h in selected:
            #     p = (h.get("payload") or {})
            #     dt = _norm_doc_type(p.get("doc_type"))
            #     by_type.setdefault(dt, 0)
            #     by_type[dt] += 1        
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
    # if OAS_CACHE is not None:
    #     if debug:
    #         print(f"[DEBUG] OAS cached: params={len(OAS_CACHE.get('params',[]))}, paths={len(OAS_CACHE.get('paths',[]))}")
    #     return OAS_CACHE

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
def call_ollama_chat(messages: list[dict], timeout: float | None = None) -> str:
    """
    優先走 Ollama /api/chat（Chat Completions 風格），失敗時退回 /api/generate。
    參數對齊：temperature/top_p/top_k/min_p/repeat_penalty/num_ctx/num_predict/stop
    這些可用環境變數覆寫（都可選）：
      LLM_TEMP, LLM_TOP_P, LLM_TOP_K, LLM_MIN_P, LLM_REPEAT_PENALTY, LLM_NUM_CTX, LLM_MAX_TOKENS, LLM_STOP
    """
    temp = float(os.environ.get("LLM_TEMP", "0.7"))
    top_p = float(os.environ.get("LLM_TOP_P", "0.9"))
    top_k = int(os.environ.get("LLM_TOP_K", "40"))
    min_p = float(os.environ.get("LLM_MIN_P", "0.05"))
    rpt_p = float(os.environ.get("LLM_REPEAT_PENALTY", "1.05"))
    num_ctx = int(os.environ.get("LLM_NUM_CTX", "8192"))
    num_pred = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
    try:
        stops = json.loads(os.environ.get("LLM_STOP", "[]"))
        if not isinstance(stops, list):
            stops = []
    except Exception:
        stops = []

    chat_url = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
    gen_url  = OLLAMA_URL  # 你檔案原本已有（/api/generate）

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": m.get("role","user"), "content": m.get("content","")} for m in messages],
        "stream": False,
        "options": {
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": rpt_p,
            "num_ctx": num_ctx,
            "num_predict": num_pred,
            "stop": stops,
        }
    }
    try:
        resp = requests.post(chat_url, json=payload, timeout=timeout or OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # 新版 Ollama
        if isinstance(data, dict) and data.get("message"):
            return (data["message"].get("content") or "").strip()
        # 舊版/相容
        if "response" in data:
            return (data.get("response") or "").strip()
    except Exception:
        # 退回 /api/generate：把 messages 攤平成平面 prompt
        flat = "\n\n".join(f"{m.get('role','user').upper()}:\n{m.get('content','')}" for m in messages)
        resp = requests.post(gen_url, json={"model": OLLAMA_MODEL, "prompt": flat, "stream": False},
                             timeout=timeout or OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or data.get("choices",[{}])[0].get("text","") or "").strip()

    return ""

def call_ollama_raw(prompt: str, timeout: float = OLLAMA_TIMEOUT) -> str:
    resp = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=timeout)
    resp.raise_for_status(); data = resp.json()
    return data.get("response","").strip() or data.get("choices",[{}])[0].get("text","")

def call_llamacpp_chat(messages: list[dict], timeout: float | None = None) -> str:
    """
    優先走 llama.cpp 的 OpenAI 相容 /v1/chat/completions；失敗時退回 /completion。
    參數對齊同上；退回 /completion 時會把 messages 攤平成平面 prompt。
    """
    temp = float(os.environ.get("LLM_TEMP", "0.7"))
    top_p = float(os.environ.get("LLM_TOP_P", "0.9"))
    top_k = int(os.environ.get("LLM_TOP_K", "40"))
    min_p = float(os.environ.get("LLM_MIN_P", "0.05"))
    rpt_p = float(os.environ.get("LLM_REPEAT_PENALTY", "1.05"))
    num_pred = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
    try:
        stops = json.loads(os.environ.get("LLM_STOP", "[]"))
        if not isinstance(stops, list):
            stops = []
    except Exception:
        stops = []

    chat_url = os.environ.get("LLAMA_CHAT_URL", "http://localhost:8201/v1/chat/completions")
    comp_url = LLAMA_URL  # 你檔案原本已有（/completion）

    payload = {
        "model": "local",  # llama.cpp 忽略此值，但欄位需要存在
        "messages": [{"role": m.get("role","user"), "content": m.get("content","")} for m in messages],
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repeat_penalty": rpt_p,
        "max_tokens": num_pred,
        "stop": stops,
    }
    try:
        resp = requests.post(chat_url, json=payload, timeout=timeout or LLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("choices"):
            return (data["choices"][0].get("message",{}).get("content","") or "").strip()
    except Exception:
        # 退回 legacy /completion
        flat = "\n\n".join(f"{m.get('role','user').upper()}:\n{m.get('content','')}" for m in messages)
        resp = requests.post(comp_url, json={
            "prompt": flat,
            "n_predict": num_pred,
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": rpt_p,
            "stop": stops,
        }, timeout=timeout or LLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("content") or data.get("choices",[{}])[0].get("text","") or "").strip()

    return ""

def call_llamacpp_raw(prompt: str, timeout: float = LLAMA_TIMEOUT) -> str:
    resp = requests.post(LLAMA_URL, json={"prompt": prompt, "n_predict": 512, "temperature": 0.0}, timeout=timeout)
    resp.raise_for_status(); data = resp.json()
    return (data.get("content") or data.get("choices",[{}])[0].get("text"," ")).strip()

def run_llm(llm: str, messages: list[dict], timeout: float = 120.0) -> str:
    """
    統一用 Chat Completions 流程：
    - llm == "ollama" → call_ollama_chat(messages)
    - 其他（含 "llama"）→ call_llamacpp_chat(messages)
    """
    if not messages:
        return ""
    # 這裡不再把 messages 攤平成 "ROLE:\ncontent" 平面 prompt
    try:
        if llm == "ollama":
            return call_ollama_chat(messages, timeout=timeout) or ""
        else:
            return call_llamacpp_chat(messages, timeout=timeout) or ""
    except Exception as e:
        # 保底：回傳空字串，由上層邏輯決定 fallback
        return ""

def run_llm_raw(llm: str, messages: list[dict], timeout: float = 120.0) -> str:
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
        timeout = max(timeout, OLLAMA_TIMEOUT) or OLLAMA_TIMEOUT
        return call_ollama_raw(prompt, timeout=timeout)
    # default: llama.cpp style
    timeout = max(timeout, LLAMA_TIMEOUT) or LLAMA_TIMEOUT
    return call_llamacpp_raw(prompt, timeout=timeout)

# ========== Shared sysrules (single source of truth) ==========
""" 把 list/tuple/dict 等轉成乾淨字串，避免字串拼接時 TypeError。
def _as_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return "\n".join(_as_text(i) for i in x if i is not None)
    if isinstance(x, dict):
        try:
            return json.dumps(x, ensure_ascii=False, indent=2)
        except Exception:
            return str(x)
    return str(x)
"""
# ---------- OAS 摘要（供 planner / continue 共用） ----------
def _sysrule_oas_whitelist(oas: Dict[str, Any]) -> str:
    paths   = sorted(oas.get("paths", []) or [])
    params  = sorted(oas.get("params", []) or [])
    append_allowed = sorted(set((oas.get("append_allowed") or []) + (oas.get("append") or [])))
    enums   = oas.get("param_enums", {}) or {}

    enums_pretty = []
    for k, v in (enums.items() if isinstance(enums, dict) else []):
        if isinstance(v, (list, tuple)) and v:
            enums_pretty.append(f"  - {k} ∈ {{{', '.join(map(str, v))}}}")
    enums_line = "\n".join(enums_pretty) if enums_pretty else "(none)"

    return (
        "OAS whitelist (do NOT invent endpoints/params):\n"
        f"- endpoints: {', '.join(paths) or '(none)'}\n"
        f"- params: {', '.join(params) or '(none)'}\n"
        f"- append allowed: {', '.join(append_allowed) or '(none)'}\n"
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

def _sysrule_planner(oas_info: Dict[str, Any], debug: bool=True) -> str:
    allowed_paths  = ", ".join(sorted(oas_info.get("paths", []))) or "(none)"
    allowed_params = ", ".join(sorted(oas_info.get("params", []))) or "(none)"
    allowed_append = ", ".join(sorted(oas_info.get("append_allowed", []))) or "(none)"
    append_usage = f"Allowed values: {allowed_append}" if allowed_append and allowed_append != "(none)" else "Not allowed" 
    if debug:
        print(f"[DEBUG] planner append usage: {append_usage}")

    return (
        "API planner rules:\n"
        "- Never invent endpoints or params; use ONLY the OAS whitelist.\n"
        f"(1) Allowed endpoints: {allowed_paths}\n"
        f"(2) Allowed query params: {allowed_params}\n"
        f"(3) 'append' param {append_usage}\n\n"
        "CSV / JSON selection:\n"
        "- Prefer JSON if the user didn't explicitly ask CSV/file download.\n"
        "- For CSV: choose '/csv' endpoint if present; otherwise ONLY if 'format' param enum includes 'csv', set format='csv'.\n\n"
        "Spatial-related params guidance(use ONLY whitelisted param names, e.g., lon0,lon1,lat0,lat1 (bbox) or lon,lat):\n"
        "- If region (bbox) or place (single point) implied, estimate a most relevant spatial range or constraint for that region/place.\n"
        "- Never global extent by default (no lon:-180..180, lat:-90..90).\n"
        "- Never crossing longitude 180°/0° in one box; if intended, plan split ranges (merging is done in code stage).\n\n"
        "Temporal params guidance (only whitelisted param names, e.g., start/end):\n"
        "- If the question implies a year, translate to a 12-month range.\n"
        "- If it implies a month/day (e.g., map on a date), set a tight monthly/daily window.\n"
        "- Use ISO dates (YYYY-MM-DD).\n\n"
        "Param: 'append' guidance (if whitelisted and values allowed):\n"
        f"- Allowed values in 'append': {allowed_append} are used for variables selection as API fetching data\n"
        "- Decide and infer requested variables from intent, for examples:\n"
        "  • 海溫/SST（未提 anomaly）→ include 'sst' (or 'temperature', depends on which is allowed)\n"
        "  • 距平值/異常值/anomaly → include 'sst_anomaly' (if allowed)\n"
        "  • 海洋熱浪等級/MHW levels → include 'level' (if allowed)\n"
        "- If analysis/plotting requires variables and 'append' is whitelisted, DO NOT leave 'append' empty in the PLAN JSON; include ALL required variables (comma-separated) strictly from whitelist.\n\n"
        "MHW chunking constraints (planner awareness, and decide now for CODE to implement):\n"
        "- API max span depends on bbox size (<=10°: up to 10y; >10°: 1y; >90°: 1m). You just plan ONE canonical request; "
        "  actual looping/concat is implemented in the CODE stage.\n"
        "- If the chosen endpoint is '/api/mhw' or '/api/mhw/csv' (i.e. use MHW API), DECIDE the chunking strategy now and include it in the PLAN JSON as a top-level field 'chunk_rule'.\n"
        "  Use one of the following concise instructions (do NOT put into params):\n"
        "    • 'monthly'  (bbox>90° × 90° → loop monthly & concat)\n"
        "    • 'yearly'   (bbox>10° × 10° → loop yearly & concat)\n"
        "    • 'decade'   (bbox<=10° × 10° & span>10y → split by decade/year & concat)\n"
        "    • ''         (empty string, if no chunking is required or NOT use MHW API)\n\n"
        "Plotting decision (decide here for CODE to follow):\n"
        "- If the question implies plotting:\n"
        "    • (趨勢/變化/長期/時序/指數/時間/time series) → plot_rule='timeseries'\n"
        "    • (分佈/地圖/空間/網格/水平分佈/map) → plot_rule='map'\n"
        "- If no plotting is implied → plot_rule='none' (or empty).\n\n"
        "Plan Inheritance:\n"
        "- If usr intent to follow up or revise previous question, do NOT remove constraints/params in previous plan. Inherit values, or update them if the new question explicitly asks for altering values.\n"
        "- For example: if user ask for altering latitude range, then just alter whitelisted latitude-related params, keep other key/values unchanged.\n"
        "- Always output a full PLAN object that includes **all** the keys (endpoint, params, chunk_rule, plot_rule).\n"
        # "The PLAN is only a control object for the client; it is NOT user-visible prose. Strictly follow the following JSON format and do NOT treat it as the answer to the question.\n"
        'Return JSON only: { "endpoint": "<path>", "params": { "<param>": "<value>" }, "chunk_rule": "<monthly|yearly|decade|>", "plot_rule": "<timeseries|map|none|>" }\n'
    )

def _sysrule_code_assistant(chunk_hint_line: str = "", debug: bool=False) -> str:
    base = (
        "Your are a Python Coder. Your task and coding rules are:\n"
        "- Task — generate a single runnable Python script (only code, no prose).\n"
        "- Obey the PLAN JSON at Planner stage exactly:\n"
        "  • Use the EXACT endpoint and 'params' values from PLAN;\n"
        "  • If PLAN has 'append', include it verbatim in 'params' (no inventions);\n"
        "  • NEVER change lon/lat/time ranges unless user asked to;\n"
        "  • NEVER invent endpoints/params/columns.\n"
        "- Control keys 'chunk_rule' and 'plot_rule' MUST NOT be sent as query params; they only control client behavior.\n"    
        f"- Use server URL defined in OAS as BASE_URL (e.g., MHW API's BASE_URL = {ECO_BASE_URL})\n"
        "- requests.get(BASE_URL + endpoint, params=params, timeout=60)\n"
        "- JSON responses: r.json() → pd.DataFrame(...). DO NOT use io.StringIO unless CSV.\n"
        "- CSV responses: ONLY if endpoint contains '/csv' OR whitelist allows format='csv', use pandas.read_csv(io.StringIO(r.text)).\n"
        "- If bounding box (lon/lat or lon0/lon1/lat0/lat1) crosses antimeridian/0°, split into two requests and pandas.concat for the two responses.\n"
        "- Convert 'date' to datetime if present.\n"
        "- Keep indentation exactly consistent with the preceding code block if you are asked to continue code.\n"
    )

    base += (
        "- For Plotting behavior, read the PLAN JSON to get the decision ('plot_rule') at Planner stage:\n"
        "  • If plot_rule=='none' or empty → no plot, just fetch & prepare df.\n"
        "  • If plot_rule=='timeseries':\n"
        "      - use 'date' on x-axis; obey MHW CHUNKING constraints; bounded for-loops only; no open-ended while.\n"
        "  • If plot_rule=='map':\n"
        "      - build a gridded array aligned to lon/lat (NO scatter). Steps:\n"
        "        (1) unique sorted lon/lat from df columns (not synthetic linspace);\n"
        "        (2) create numpy.meshgrid from those unique lon/lat as Lon, Lat;\n"
        "        (3) align values to grid cells as Z using either pivot_table or explicit fill;\n"
        "        (4) use matplotlib.pyplot.pcolormesh(Lon, Lat, Z, shading='auto').\n"
        "- Treat 'level' as categorical when plotting; use a discrete ListedColormap.\n\n"
    )

    # if debug:
    #     print(f"[DEBUG] chunk_hint_line: {chunk_hint_line}")
    # if chunk_hint_line:
    base += (
        "Read the PLAN JSON to get the decision ('chunk_rule') of MHW CHUNKING constraints at Planner stage.\n" # general fallback if PLAN has no 'chunk_rule'):\n"
        # f"- {chunk_hint_line}\n"
        "- If 'chunk_rule' equals:\n" 
        "    • 'monthly: API requests should loop monthly and concat'\n"
        "    • 'yearly: API requests should loop yearly and concat'\n"
        "    • 'decade: API requests should split by decade or by year and concat'\n"
        "    • '' (empty string): No chunking is required\n"
        "- If the PLAN includes a top-level field 'chunk_rule', you MUST implement the described time-splitting loop accordingly (do NOT send 'chunk_rule' as a query param).\n"
        "- Split ONLY by time (start/end); keep spatial params fixed; concat at the end.\n\n"
    )

    base += (
        "Return a runnable Python script within code fence. CODE cannot be empty.\n"         
    )    
    return base
    
def _sysrule_explain(force_zh: bool) -> str:
    lang_line = ("- 回答請使用繁體中文。\n" if force_zh else "- Answer in the user's language.\n")
    return (
        "You are an ODB assistant. Provide a clear, concise explanation that directly addresses the user's question.\n"
        + lang_line +
        "- Use ONLY the information in the provided notes when citing specific options/resources.\n"
        "- DO NOT include any programming code, code fences, pseudo-code, or placeholders.\n"
        "- Avoid one-pass control markers like <<<...>>> and tokens like {explain}.\n"
        "- Prefer actionable, non-code options when relevant (e.g., web pages, dashboards, documented tools), and summarize what to do and where to find it.\n"
        "- Keep it short but complete: 3–8 short paragraphs or a tight bulleted list.\n"
        # "- If the notes include a CLI guide, describe what it enables at a high level (no commands).\n"
        "- If asked to explain/annotate previous code, explain critical steps clearly.\n"
    )

# ---------- Continue LLM：只補尾、不重寫；遵循 OAS / 計劃 / chunk 規則 ----------
def _sysrule_continue(oas: Dict[str, Any], chunk_hint_line: str = "") -> str:
    return "\n".join([
        "You are continuing an *incomplete* Python script. Do NOT repeat earlier code; only finish the missing tail.",
        _sysrule_oas_whitelist(oas or _get_oas_cache()),
        _sysrule_code_assistant(chunk_hint_line),
        "- Keep indentation exactly consistent with the preceding code block."

    ])

# --------------------
# Mode classifier
# --------------------
def llm_choose_mode(question: str, llm: str, debug: bool=False) -> str:
    sysrule = _sysrule_classifier()
    prompt  = f"{sysrule}\n\nQuestion:\n{question}\n\nYour answer (one token only):"
    txt = run_llm(llm, [{"role":"system","content":sysrule},{"role":"user","content":prompt}], timeout=LLAMA_TIMEOUT) or ""
    ans = (txt or "").strip().lower()
    mode = "code" if "code" in ans else "explain"
    if debug:
        print(f"[DEBUG] choose_mode → raw={ans!r} | mode={mode}")
    return mode    

# --------------------
# Planning
# --------------------
def _norm_append_values(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip().lower() for x in v.split(",") if x.strip()]
    if isinstance(v, (list, tuple, set)):
        return [str(x).strip().lower() for x in v]
    return [str(v).strip().lower()]

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
    if "append" in params and (oas.get("append_allowed") or []):
        allowed = {str(a).lower() for a in oas.get("append_allowed") or []}
        vals = _norm_append_values(params.get("append"))
        bad = [v for v in vals if v not in allowed]
        if bad:
            return False, f"append contains values not allowed: {bad}"

    # YYYY-MM → YYYY-MM-DD
    for key in ("start","end"):
        if key in params and re.match(r"^\d{4}-\d{2}$", str(params[key])):
            params[key] = str(params[key]) + "-01"
    return True, ""

def llm_json_plan(
    question: str,
    oas: Dict[str,Any],
    llm: str,
    prev_plan: Optional[Dict[str,Any]] = None,
    debug: bool=False,
) -> Optional[Dict[str,Any]]:
    sysrule = _sysrule_planner(oas)

    oas_blob = {
        "paths": oas.get("paths", []),
        "params": oas.get("params", []),
        "append_allowed": list(set((oas.get("append_allowed") or []) + (oas.get("append") or []))),
        "param_enums": oas.get("param_enums", {}) or {}
    }
    prev_line = f"Previous plan:\n{json.dumps(prev_plan, ensure_ascii=False)}" if prev_plan else "Previous plan: (none)"

    prompt = (
        f"{sysrule}\n\n"
        f"Question:\n{question}\n\n"
        f"{prev_line}\n\n"
        f"OAS whitelist (for you to choose from):\n{json.dumps(oas_blob, ensure_ascii=False)}\n\n"
        "Return JSON ONLY:\n"
        "{\n"
        '  "endpoint": "<one path from whitelist>",\n'
        '  "params": { "<param_name>": "<value>" }\n'
        "}\n"
        "Output ONLY the JSON (no fences)."
    )

    if debug:
        print("[DEBUG] planner | LLM entering")

    txt = run_llm(llm, [{"role":"system","content":sysrule},{"role":"user","content":prompt}], timeout=LLAMA_TIMEOUT) or ""
    s = re.sub(r"^```(?:json)?\s*", "", (txt or "").strip(), flags=re.I)
    s = re.sub(r"\s*```$", "", s, flags=re.I)

    if debug:
        print(f"[DEBUG] planner | raw chars={len(txt)} | cleaned chars={len(s)}")

    try:
        obj = json.loads(s)
        if validate_plan(obj, oas):
            return obj
        if debug:
            print("[DEBUG] planner | OAS validation failed")
    except Exception as e:
        if debug:
            print(f"[DEBUG] planner | JSON parse failed: {e!r}")
    return None

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
# Markdown code extraction
# --------------------
def _preclean_text_for_code(s: str) -> str:
    """
    在抽碼前的前清理：
    - 移除 BOM/零寬/全形空白等隱藏字元
    - 若在第一個 ``` 之前只有標點/空白，直接砍掉前綴（避免前導「？」等噪音造成 line 1 SyntaxError）
    """
    if not isinstance(s, str):
        return ""
    s = s.replace("\ufeff", "").replace("\u200b", "")
    s = s.lstrip("\u3000").lstrip()  # 全形空白與一般空白
    m = re.search(r"```", s)
    if m:
        prefix = s[:m.start()]
        if re.fullmatch(r"[\s\W\u3000-\u303F\uFF00-\uFFEF]*", prefix or ""):
            s = s[m.start():]
    return s

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
    return (md or "").strip()

# --------------------
# Code generation & guard
# --------------------
# 放在 code/LLM helpers 區塊
# 置於檔頭 imports 附近已有 re/json/ast 的前提下
FULLWIDTH_PUNCTS = "\uFF01\uFF1F\uFF0C\u3002\u3001"

def _preclean_code_for_parse(s: str, debug: bool=False, tag: str="") -> str:
    if not s:
        return ""
    # 清掉任何位置的 JSON PLAN（不只在檔頭）
    s = JSON_PLAN_ANY_RE.sub("\n", s)

    # 清掉獨立的 'code' token 與明顯非註解中文前言行（保留以 # 開頭者）
    out_lines = []
    seen_real = False
    for ln in (s.replace("\r","").split("\n")):
        if re.match(r'^\s*code\s*$', ln, flags=re.I):
            continue
        if not seen_real and CJK_LINE_RE.match(ln) and not ln.lstrip().startswith("#"):
            # 跳過純中文說明行（直到我們看到真正的程式語句）
            continue
        out_lines.append(ln)
        if not seen_real and (ln.strip().startswith(("import ","from ","BASE_URL","endpoint","def ","class ")) or CODE_LIKE_RE.search(ln)):
            seen_real = True
    s = "\n".join(out_lines).strip()

    # 若有多段 fence 的殘留，整體再抽一次 code → 確保單一段
    code = extract_code_from_markdown(s)
    if debug:
        print(f"[DEBUG] preclean({tag}): {len(s)}→{len(code)} chars after strip preface/json")

    return code

def _debug_dump_code(code: str, tag: str, debug: bool, max_chars: int = 8000) -> None:
    if not debug:
        return
    length = len(code or "")
    body = code if length <= max_chars else (code[:max_chars] + "\n...[truncated]...")
    print(f"[DEBUG] ==== BEGIN {tag} CODE DUMP ({length} chars) ====\n{body}\n[DEBUG] ==== END {tag} CODE DUMP ====")

def code_continue_via_llm(
    question: str,
    plan: Dict[str, Any],
    last_code: str,
    llm: str,
    debug: bool = False,
    chunk_line: Optional[str] = None,
    oas: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Continue an incomplete Python script. Uses the same common code rules as the code assistant,
    plus explicit OAS/append/spatial-temporal guidance. Does not require prev_plan; pass the
    current 'plan' (or previously stored plan when invoked as a standalone continue).
    """
    oas = oas or _get_oas_cache()
    local_chunk_line = build_chunk_line_from_plan(plan) or chunk_line

    sysrule = _sysrule_continue(oas=oas, chunk_hint_line=local_chunk_line)

    plan_blob = json.dumps(plan or {}, ensure_ascii=False)
    user_parts = []
    user_parts.append(
        "請從下列未完成的程式碼處繼續，補齊缺漏並完成繪圖與收尾。"
        "不要重複前段程式碼，保持縮排一致。"
    )
    user_parts.append(f"Plan (must be respected):\n{plan_blob}")
    user_parts.append("--- 上段程式碼 ---\n" + (last_code or ""))
    user_msg = "\n\n".join(user_parts)

    messages = [
        {"role": "system", "content": sysrule},
        {"role": "user", "content": user_msg},
    ]

    try:
        if debug:
            print("[DEBUG] use Continue LLM")
        txt = run_llm(llm, messages, timeout=600.0) or ""
    except Exception as e:
        if debug:
            print(f"[DEBUG] Continue LLM error: {e}")
        return ""

    txt = _preclean_text_for_code(txt or "")
    more = extract_code_from_markdown(txt).strip() or _strip_one_pass_markers(txt).strip()
    more = _preclean_code_for_parse(more, debug=debug, tag="continue")
    return _ensure_fenced(more) if more else ""

def code_from_plan_via_llm(
    question: str,
    plan: dict,
    llm: str,
    debug: bool = False,
    chunk_line: str | None = None,
    oas: dict | None = None,   # <— NEW
) -> str | None:
    oas = oas or _get_oas_cache()  # <— resolve    # 共用規則（與 one-pass 一致）
    sysrule = _sysrule_code_assistant(chunk_hint_line=chunk_line)

    user = (
        "Generate ONE runnable Python script using the EXACT endpoint & params from this plan:\n"
        + json.dumps(plan, ensure_ascii=False)
    )

    try:
        txt = run_llm(llm, [{"role":"system","content":sysrule},{"role":"user","content":user}], timeout=300.0) or ""
        
    except Exception as e:
        if debug:
            print(f"[DEBUG] LLM codegen error: {e}")
        return ""
    
    txt = _preclean_text_for_code(txt or "")
    code = _extract_code_strict(txt)
    # 未閉合/不完整 → 單次續寫
    unclosed = ("```" in (txt or "")) and not CODE_FENCE_RE.search(txt)
    looks_ok = _looks_like_complete_code(code, debug=debug)
    if unclosed or not looks_ok:
        local_chunk_line = build_chunk_line_from_plan(plan)
        if debug:
            print(f"[DEBUG] chunk_hint: {('n/a' if not local_chunk_line else 'present')}")
        _debug_dump_code(code, "ONE-PASS BEFORE CONTINUE", debug)
        if debug:
            print("[DEBUG] one-pass: need continuation")

        more = code_continue_via_llm(
            # question="請從上段未完成處繼續，補足缺漏行，完成繪圖與收尾，不要重複前段程式碼，但保持縮排正確一致。",
            question=question,
            plan=plan or {},
            last_code=code, 
            llm=llm,
            debug=debug,
            chunk_line=chunk_line,
            oas=oas
        )
        if more:
            code = merge_code_fences(code.rstrip() + "\n\n", more.lstrip())
    return code

CODE_FENCE_RE = re.compile(r"```(?:python|py)?[\s\S]*?```", re.IGNORECASE)

def _parse_one_pass_output(txt: str, debug: bool=False) -> Dict[str,Any]:
    out: Dict[str,Any] = {"raw": txt or ""}
    t = txt or ""

    m_mode = re.search(r'<<<MODE>>>\s*(code|explain)\s*<<<END>>>', t, flags=re.IGNORECASE)
    mode = (m_mode.group(1).lower() if m_mode else None)

    # 有明確 code fence/區塊才視為 code；否則用宣告的 MODE；再不然預設 explain
    has_code_block = bool(CODE_FENCE_RE.search(t) or re.search(r'<<<CODE\s+python>>>', t, flags=re.I))
    if mode is None:
        mode = "code" if has_code_block else "explain"
    out["mode"] = mode

    # 解析 PLAN（保留）
    m_plan = re.search(r'<<<PLAN>>>\s*([\s\S]*?)\s*<<<END>>>', t, flags=re.IGNORECASE)
    if m_plan:
        try:
            out["plan"] = json.loads(m_plan.group(1))
        except Exception:
            out["plan"] = None

    if mode == "code":
        m_code = re.search(r'<<<CODE\s+python>>>\s*([\s\S]*?)\s*<<<END>>>', t, flags=re.IGNORECASE)
        raw_code = (m_code.group(1).strip() if m_code else t)
        raw_code = _strip_one_pass_markers(raw_code)
        raw_code = _preclean_text_for_code(raw_code or "")
        code = extract_code_from_markdown(raw_code).strip() or raw_code
        code = _preclean_code_for_parse(code, debug=debug, tag="parse_one_pass")
        if not CODE_LIKE_RE.search(code or ""):
            # 這其實是文字，不當 CODE
            out["mode"] = "explain"
            s = _clean_explain_text(t)  # 你既有的清理（會去掉 <<<...>>> 與 {explain} 等）
            out["explain"] = _strip_leading_punct(s)
            out.pop("code", None)
        else:
            out["code"] = _ensure_fenced(code) if code else ""
    else:
        s = _clean_explain_text(t)
        # 清掉殘留標記
        s = re.sub(r'\bMODE\s*:\s*(code|explain)\b', ' ', s, flags=re.I)
        s = re.sub(r'\{code\}|\{plan\}', ' ', s, flags=re.I)
        s = re.sub(r'<<<[^>]+>>>', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        out["explain"] = _strip_leading_punct(s)

    if debug:
        keys = ", ".join(sorted(out.keys()))
        clen = len(out.get("code","") or "")
        print(f"[DEBUG] _parse_one_pass_output → keys: {keys} | mode={out['mode']} | code_len={clen}")
    return out

def _strip_one_pass_markers(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r'^\s*<<<[^>]+>>>\s*$', '', s, flags=re.MULTILINE)  # 整行標記
    s = re.sub(r'<<<[^>]+>>>', '', s)  # 殘留片段
    return s.strip()

# === restore: explain_via_llm (uses rerank+diversify first; falls back if needed) ===
def explain_via_llm(question: str, hits: list, llm: str, debug: bool = False) -> str:
    # 先嘗試你的 rerank / diversify；失敗就 fallback 簡單排序
    try:
        picked = rerank_and_diversify_hits(hits, intent="explain", topk=4, debug=debug)
    except Exception:
        priority = {"web_article": 0, "cli_tool_guide": 1, "manual": 2, "faq": 3, "paper_note": 4,
                    "api_spec": 8, "code_snippet": 9, "code_example": 9}
        def _rk(h):
            k = (h.get("payload", {}) or {}).get("doc_type", "").lower()
            return priority.get(k, 7)
        picked = sorted(hits or [], key=_rk)[:4]

    rag_notes = collect_rag_notes(picked, max_chars=1400, debug=debug)
    force_zh = bool(re.search(r"繁體|正體|繁體中文", question)) or bool(re.search(r"[\u4e00-\u9fff]", question))
    sysrule = _sysrule_explain(force_zh)
    user = "[RAG NOTES]\n" + (rag_notes or "(none)") + "\n\nQUESTION:\n" + question

    if debug:
        print("[DEBUG] explain | LLM(single-pass) entering")

    txt = run_llm(llm, [{"role":"system","content":sysrule},
                        {"role":"user","content":user}], timeout=300.0) or ""
    
    ans = _clean_explain_text(txt)   # 不破壞 markdown 的清理
    if debug:
        print(f"[DEBUG] explain | raw={len(txt)} | cleaned={len(ans)}")
    if not ans:
        # 保底：回傳精簡的 notes（不再做二次 LLM）
        raw = re.sub(r'```[\s\S]*?```', ' ', rag_notes or "")
        return re.sub(r'\s+', ' ', raw).strip()
    return ans

def _render_history_for_llm(history: Optional[List[Tuple[str, str]]],
                            max_pairs: int = 3,
                            max_chars: int = 800) -> str:
    """
    將最近的對話歷史壓縮成短文字供 LLM 參考。
    - 只取最後 max_pairs 對 Q/A
    - 全文上限 max_chars，必要時會截斷
    """
    if not history:
        return ""
    take = history[-max_pairs:]
    parts = []
    used = 0
    for q, a in take:
        seg = f"Q: {q.strip()}\nA: {str(a).strip()}\n"
        if used + len(seg) > max_chars:
            seg = seg[: max(0, max_chars - used)]
        if seg:
            parts.append(seg)
            used += len(seg)
        if used >= max_chars:
            break
    return "\n".join(parts).strip()

def build_timeseries_fallback_template(plan: Dict[str, Any], expect_csv: bool) -> str:
    endpoint = plan["endpoint"]
    params = plan["params"]

    lines = [
        "# 最小可執行範例（時序圖）：依需求自行調整參數與欄位",
        "import requests",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        f'BASE = "{ECO_BASE_URL}"',
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
        f'BASE = "{ECO_BASE_URL}"',
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


# ---- RAG notes helpers ----

def _extract_title_from_yaml_front_matter(text: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.lstrip()
    if not s.startswith('---'):
        m_any = re.search(r'^\s*title\s*:\s*(.+)$', text, flags=re.MULTILINE)
        if m_any:
            raw = m_any.group(1).strip()
            if len(raw) >= 2 and raw[0] in ("'", '"') and raw[-1] == raw[0]:
                raw = raw[1:-1]
            return raw or None
        return None
    lines = s.splitlines()
    i = 1
    buf = []
    while i < len(lines):
        line = lines[i]
        if line.strip() == '---':
            break
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
    hit = hit or {}
    flat_keys = ("title", "doc_title", "name", "source_title", "filename", "basename", "path", "url", "id")
    for k in flat_keys:
        v = hit.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip(), f"flat.{k}"
    for nk in ("meta", "front", "metadata", "front_matter"):
        node = hit.get(nk)
        if isinstance(node, dict):
            v = node.get("title")
            if isinstance(v, str) and v.strip():
                return v.strip(), f"nested.{nk}.title"
    payload = hit.get("payload")
    if isinstance(payload, dict):
        v = payload.get("title")
        if isinstance(v, str) and v.strip():
            return v.strip(), "payload.title"
    if isinstance(payload, str) and payload.strip():
        t = _extract_title_from_yaml_front_matter(payload)
        if t:
            return t.strip(), "payload.yaml"
    content = hit.get("content") or hit.get("text") or ""
    if isinstance(content, str) and content.strip():
        t = _extract_title_from_yaml_front_matter(content)
        if t:
            return t.strip(), "content.yaml"
    htype = _type_of(hit)
    title = default or f"{htype} #{idx+1}"
    return title, "fallback"

def rerank_and_diversify_hits(
    hits: list,
    query: str | None = None,
    intent: Optional[str] = None,
    topk: Optional[int] = None,
    debug: bool = False
) -> list:
    """
    輕量 re-rank + diversify（單一權威版本）：
    - intent in {"code","explain", None}：explain 時微調提升 web/guide，code 維持中性
    - query 目前只做輕度關鍵詞比對（保留擴充空間）
    - topk 若提供，最後裁切到該長度
    """
    if not hits:
        return []

    scored = []
    ql = (query or "").lower()

    for h in hits:
        base = 1.0
        p = (h.get("payload") or {})
        dt = (p.get("doc_type") or "").lower()
        title = (p.get("title") or h.get("title") or "").lower()

        # 意圖微調（溫和）
        if intent == "explain":
            if dt in ("web_article", "cli_tool_guide", "manual", "faq"):
                base += 0.35
            elif dt in ("api_spec", "code_snippet", "code_example"):
                base -= 0.15
        # intent == "code"：不特別加或扣，避免雙重偏向

        if any(k in title for k in ("enso", "marine heatwave", "nino", "mhw")):
            base += 0.1

        scored.append((base, h))

    scored.sort(key=lambda x: x[0], reverse=True)

    # 多樣性去重
    seen, out = set(), []
    def _key(h):
        p = h.get("payload") or {}
        return (
            p.get("doc_id")
            or p.get("canonical_url")
            or (p.get("title") or h.get("title") or "").split(" — ")[0].strip().lower()
            or p.get("source_file")
            or "blob::" + str(hash(h.get("content") or h.get("text") or ""))
        )
    for s, h in scored:
        k = _key(h)
        if k in seen: continue
        seen.add(k); out.append(h)

    if isinstance(topk, int) and topk > 0:
        out = out[:topk]

    if debug:
        from collections import Counter
        c = Counter(( (h.get("payload") or {}).get("doc_type") or "n/a").lower() for h in out)
        print(f"[DEBUG] rerank/diversify intent={intent!r} → type mix: {c}")

    return out

def collect_rag_notes(
    hits: list,
    max_chars: int = 1200,
    code_snippet_char_limit: int = 500,
    debug: bool = False
) -> str:
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
        # 與主流程一致：用 Counter 顯示型別分佈
        kinds = [ (_type_of(h) or "n/a") for h in (hits or []) ]
        print(f"[DEBUG] RAG hits total: {len(hits or [])}; type distribution: {Counter(kinds) or 'n/a'}")

    for idx, h in enumerate(hits or []):
        if used >= max_chars:
            break
        h = h or {}
        htype = _type_of(h)
        title, title_src = _best_title(h, idx, default=htype)
        raw_content = h.get("content") or h.get("text") or ""
        if (not raw_content) and isinstance(h.get("payload"), dict):
            raw_content = h["payload"].get("content", "")
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

def _format_chat_history(history: list[tuple[str, str]] | None, max_turns: int = 3, max_chars: int = 1200) -> str:
    """
    將最近幾輪 Q/A 壓成短上下文，供 LLM 參考（避免拖慢/洗版）。
    """
    if not history:
        return ""
    buf = []
    for q, a in history[-max_turns:]:
        buf.append(f"Q: {q}\nA: {a}\n")
    s = "\n".join(buf).strip()
    if len(s) > max_chars:
        s = s[-max_chars:]
    return s

def _norm_cite_key(payload: dict) -> str:
    """用於 citation 去重的 key：
       1) canonical_url 優先
       2) 否則 source_file
       3) 否則 doc_id
       4) 否則 title 的 stem（去掉『 — 之後』與副檔名、rag/ 前綴）
    """
    cu = (payload or {}).get("canonical_url") or ""
    sf = (payload or {}).get("source_file") or ""
    did = (payload or {}).get("doc_id") or ""
    title = (payload or {}).get("title") or ""

    def _stem(s: str) -> str:
        s = s.strip()
        s = re.sub(r"^rag/", "", s)
        s = re.sub(r"\.(yml|yaml|md|markdown)$", "", s, flags=re.IGNORECASE)
        return s.lower()

    # 視為同一份的條件：canonical_url 與 source_file 互為 yml/md 對映，或同名同 stem
    if cu:
        key = _stem(cu)
    elif sf:
        key = _stem(sf)
    elif did:
        key = f"id:{did}"
    else:
        # title 去掉 " — ..." 尾段，取主標題 stem
        t = title.split(" — ")[0].strip()
        key = _stem(t)
    return key

def _dedup_key_from_hit(h: dict) -> str:
    """為 hits 做去重用的 key，優先順序：canonical_url -> doc_id -> title(normalized) -> source_file/name。"""
    p = (h.get("payload") or {})
    url = (p.get("canonical_url") or "").strip().lower()
    if url:
        return f"url::{url}"
    did = (p.get("doc_id") or "").strip().lower()
    if did:
        return f"id::{did}"
    title = (p.get("title") or p.get("name") or "").strip().lower()
    if title:
        # 與 format_citations 的策略一致：去掉「 — 後面的路徑」再小寫
        title = re.split(r"\s+—\s+", title, maxsplit=1)[0].strip().lower()
        return f"title::{title}"
    src = (p.get("source_file") or "").strip().lower()
    if src:
        return f"src::{src}"
    # 毫無可用 meta 時，退回 content hash
    content = (h.get("content") or h.get("text") or "")
    return "blob::" + str(hash(content))  # 避免爆雷，弱去重

def _union_dedup_hits(primary: list, extra: list, limit: int | None = None, debug: bool = False) -> list:
    """把 primary 和 extra 聯集後依照 primary 優先、extra 次之做去重；可選上限 limit。"""
    seen, out = set(), []
    def push(seq):
        for h in (seq or []):
            k = _dedup_key_from_hit(h)
            if k in seen:
                continue
            seen.add(k)
            out.append(h)
            if isinstance(limit, int) and limit > 0 and len(out) >= limit:
                break
    push(primary)
    if not (isinstance(limit, int) and limit > 0 and len(out) >= limit):
        push(extra)
    if debug:
        kinds = [(h.get("payload", {}) or {}).get("doc_type", "n/a").lower() for h in out]
        print(f"[DEBUG] union hits count={len(out)}; type mix after union: {Counter(kinds) or 'n/a'}")
    return out

def format_citations(hits: list[dict], question: str) -> str:
    """
    以標題 — 連結的清單輸出，並做更強的去重：
    - 將 manuals/*.md 與 rag/manuals/*.yml 視為同一來源
    - 以 canonical_url / source_file / doc_id / title stem 多重規則去重
    """
    items = []
    seen = set()

    for h in hits or []:
        p = h.get("payload", {}) or {}
        title = p.get("title") or "Untitled"
        url = p.get("canonical_url") or p.get("source_file") or p.get("url") or ""
        if not url:
            # 退而求其次：允許顯示檔名
            url = p.get("path") or p.get("file_path") or ""

        key = _norm_cite_key(p)
        if key in seen:
            continue
        seen.add(key)

        # 顯示名稱：若標題後面曾附上一個路徑，避免重複的「兩個版本」
        items.append(f"- {title} — {url}")

    return "\n".join(items) if items else "- （無）"

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
        if ("pd.read_csv(" not in code) or ("io.StringIO(" not in code):
            return False, "CSV mode requires pandas.read_csv(io.StringIO(r.text)) with import io"
    else:
        if re.search(r'pd\.read_json\s*\(', code):
            return False, "JSON must use r.json(); pd.read_json is forbidden"
        if re.search(r'io\.StringIO\s*\(\s*r\.text\s*\)\s*\)', code, flags=re.I):
            return False, "JSON must not use io.StringIO; parse via r.json()"
        if re.search(r'pd\.read_csv\s*\(\s*io\.StringIO\s*\(\s*r\.text\s*\)\s*\)', code, flags=re.I):
            return False, "JSON must not use pd.read_csv(io.StringIO(r.text)); use r.json() instead"

    # 不得提及 API key/token
    if re.search(r'api[_-]?key|token|authorization|bearer', code, re.I):
        return False, "code mentions API key/token"

    # 地圖守門（必要的硬規則）
    # if expect_map:
        # if re.search(r'\.scatter\s*\(', code):
        #     return False, "map plotting must not use scatter"
        # if "pcolormesh(" not in code and "plt.pcolormesh(" not in code:
        #     return False, "map plotting must use pcolormesh"
        # np.linspace 不再當作硬阻擋（改由 maybe_llm_validate 作軟性修補）

    # code fence 完整性（避免截斷）
    if "```python" in code and "```" not in code.strip().split("```python",1)[-1]:
        return False, "code fence likely truncated"

    return True, ""

def prepare_rag_for_one_pass(
    base_hits: list,
    question: str,
    topk: int = 6,
    debug: bool = False
) -> list:
    """保持原 base_hits 不動；另外跑一輪 rerank/diversify（code 與 explain），再做聯集去重。"""
    # 先裁到 topk（維持你的主要邏輯）
    primary = list(base_hits[:topk]) if isinstance(topk, int) and topk > 0 else list(base_hits)

    # 嘗試各自意圖的 rerank；失敗就回空列
    def _safe_rerank(intent: str, k: int) -> list:
        try:
            return rerank_and_diversify_hits(primary, query=question, intent=intent, topk=max(2, k), debug=debug)
        except Exception:
            if debug:
                print(f"[DEBUG] rerank/diversify failed for intent='{intent}', fallback to [].")
            return []

    # 兩種意圖都做，因為 one-pass 會在同一次 LLM 內自行判斷 MODE
    r_code = _safe_rerank("code", topk//2 or 3)
    r_expl = _safe_rerank("explain", topk//2 or 3)

    if debug:
        kinds_primary = Counter([(h.get("payload") or {}).get("doc_type","n/a").lower() for h in primary])
        kinds_code    = Counter([(h.get("payload") or {}).get("doc_type","n/a").lower() for h in r_code])
        kinds_expl    = Counter([(h.get("payload") or {}).get("doc_type","n/a").lower() for h in r_expl])
        print(f"[DEBUG] prepare one-pass | primary mix={kinds_primary or 'n/a'}")
        print(f"[DEBUG] prepare one-pass | rerank(code) mix={kinds_code or 'n/a'}")
        print(f"[DEBUG] prepare one-pass | rerank(explain) mix={kinds_expl or 'n/a'}")

    # 聯集去重（不在這裡硬設上限，交給 collect_rag_notes 以 max_chars 截斷）
    union_hits = _union_dedup_hits(primary, r_code + r_expl, limit=None, debug=debug)
    return union_hits

def _looks_like_complete_code(code: str, debug: bool=False) -> bool:
    code = _preclean_code_for_parse(code, debug=debug, tag="looks_like")
    if not code:
        if debug:
            print("[DEBUG] Empty code body")
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        if debug:
            print(f"[DEBUG] Not complete code: SyntaxError at line {e.lineno}: {e.msg}")
        return False
    if not tree.body:
        if debug:
            print("[DEBUG] Empty code body")
        return False

    # 孤立 import alias 截斷偵測（保留你原先邏輯）
    imported_aliases = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_aliases.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_aliases.add(alias.asname or alias.name)

    last_stmt = tree.body[-1]
    if isinstance(last_stmt, ast.Expr) and isinstance(last_stmt.value, ast.Name) and last_stmt.value.id in imported_aliases:
        if debug:
            print("[DEBUG] Not complete code: Isolated import alias at end")
        return False
    return True

def grab_block_flexible(raw_text: str, tag: str) -> str:
    """
    從 <<<TAG>>> 開始擷取到：
      1) <<<END>>>，或
      2) 下一個 <<<SOME_TAG>>>，或
      3) 字串結尾。
    用於模型漏掉 <<<END>>> 時的寬鬆解析。
    """
    m = re.search(rf'<<<{tag}>>>', raw_text, flags=re.IGNORECASE)
    if not m:
        return ""
    start = m.end()
    # 先找 END
    m_end = re.search(r'<<<END>>>', raw_text[start:], flags=re.IGNORECASE)
    if m_end:
        return raw_text[start:start + m_end.start()].strip()
    # 再找下一個 TAG
    m_next = re.search(r'<<<[A-Z_]+>>>', raw_text[start:], flags=re.IGNORECASE)
    if m_next:
        return raw_text[start:start + m_next.start()].strip()
    # 都沒有就吃到結尾
    return raw_text[start:].strip()

def _grab_block(text: str, tag: str) -> str:
    pat = re.compile(rf'<<<{tag}>>>\s*([\s\S]*?)\s*<<<END>>>', flags=re.IGNORECASE)
    m = pat.search(text or "")
    return (m.group(1) if m else "").strip()

def extract_first_fenced_code(raw_text: str) -> str:
    """
    嘗試從輸出中擷取第一段 code fence。
    支援 ```lang\n...\n``` 與 ~~~lang\n...\n~~~，也容忍無語言標記。
    """
    # 三個反引號（有語言或無語言）
    m = re.search(r"```[^\n]*\n([\s\S]*?)```", raw_text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # 容忍同一行包起來的 ```code```
    m = re.search(r"```([\s\S]*?)```", raw_text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # 三個波浪線
    m = re.search(r"~~~[^\n]*\n([\s\S]*?)~~~", raw_text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return ""

def parse_strict_blocks(text: str) -> Dict[str, str]:
    """
    嚴格區塊解析 + 寬鬆補救：
      - 先抓 <<<TAG>>>...<<<END>>>
      - 缺 END 用 grab_block_flexible()
      - mode=code 但沒有 code → 從 fenced code 抽
      - 最後仍無 code → 轉 explain
      - 另外：PLAN 為空時，嘗試從 raw 恢復（recover_plan_from_text_or_code）
    """
    raw = (text or "").replace("\u200b", "").replace("\ufeff", "")

    def _grab_block(raw_text: str, tag: str) -> str:
        m = re.search(rf'<<<{tag}>>>\s*([\s\S]*?)\s*<<<END>>>', raw_text, flags=re.IGNORECASE)
        return (m.group(1) if m else "").strip()

    out = {
        "mode": _grab_block(raw, "MODE"),
        "plan": _grab_block(raw, "PLAN"),
        "code": _grab_block(raw, "CODE"),
        "answer": _grab_block(raw, "ANSWER"),
        "log": _grab_block(raw, "LOG"),
        "mhw_rule": _grab_block(raw, "MHW_RULE"),
        "raw": raw,
    }

    # 寬鬆補抓
    if not out["plan"]:
        soft = grab_block_flexible(raw, "PLAN")
        if soft:
            out["plan"] = soft
            print("[DEBUG] parse_blocks: PLAN recovered via flexible grab", file=sys.stderr)
    if not out["code"]:
        soft = grab_block_flexible(raw, "CODE")
        if soft:
            out["code"] = soft
            print("[DEBUG] parse_blocks: CODE recovered via flexible grab", file=sys.stderr)
    if not out["answer"]:
        soft = grab_block_flexible(raw, "ANSWER")
        if soft:
            out["answer"] = soft
            print("[DEBUG] parse_blocks: ANSWER recovered via flexible grab", file=sys.stderr)

    # 標準化 mode
    out["mode"] = (out["mode"] or "").strip().lower()
    if out["mode"] not in ("code", "explain"):
        out["mode"] = "code" if out["code"] else ("explain" if out["answer"] else "")

    # mode=code 但無 code → 從 fenced code 補救
    if out["mode"] == "code" and not out["code"]:
        fenced = extract_first_fenced_code(raw)
        if fenced:
            out["code"] = fenced
            print("[DEBUG] parse_blocks: CODE recovered from fenced block", file=sys.stderr)

    # PLAN 仍為空 → 嘗試從 raw 恢復（12b 常見）
    if not out["plan"]:
        rec = recover_plan_from_text_or_code(raw)
        if rec:
            try:
                out["plan"] = json.dumps(rec, ensure_ascii=False)
                print("[DEBUG] parse_blocks: PLAN recovered from text/code heuristics", file=sys.stderr)
            except Exception:
                pass

    # 仍無 code → 視為 explain
    if out["mode"] == "code" and not out["code"]:
        tmp = re.sub(r"```[\s\S]*?```", " ", raw)
        tmp = re.sub(r"~~~[\s\S]*?~~~", " ", tmp)
        tmp = re.sub(r"<<<[^>]+>>>", " ", tmp)
        ans = re.sub(r"\s+", " ", tmp).strip()
        if ans:
            out["mode"] = "explain"
            out["answer"] = out["answer"] or ans
            print("[DEBUG] parse_blocks: switched to explain (no code found)", file=sys.stderr)

    return out

# --------------------
# Core
# --------------------
def build_one_pass_sysrule(
    question: str,
    rag_notes: list,
    oas_info: dict,
    prev_plan: Optional[Dict[str, Any]],
    force_zh: bool,
    chunk_hint_line: Optional[str] = None,
    debug: bool = False
) -> str:
    """Compose one-pass sysrule from modular components for consistency with multi-step LLMs."""
    classifier_core = _sysrule_classifier()
    planner_core    = _sysrule_planner(oas_info)
    code_core       = _sysrule_code_assistant(chunk_hint_line, debug=debug)
    explain_core    = _sysrule_explain(force_zh)

    # Temporarily remove LOG block
    #    "Always include a final <<<LOG>>> block\n"
    #    "<<<LOG>>>\n<1-2 lines: why MODE and key constraints considered>\n<<<END>>>\n"
    #    + "STEP 4 — Assistant for Logging as <<<LOG>>> content with 1–2 lines explaining each STEP's decision.\n\n"

    strict_format = (
        # --- 絕對輸出規則（關鍵！防止 20b 輸出思考文字）---
        "ABSOLUTE OUTPUT RULES:\n"
        # "- Think silently; NEVER print analysis, steps, or rationale.\n"
        # "- Do NOT write any text outside the tagged blocks below. It means: Your FIRST characters must be exactly: '<<<MODE>>>' (no preface, no commentary).\n"
        # "- Do NOT use backticks or Markdown code fences anywhere.\n"
        "If MODE=code (for STEP 2A and STEP 3):\n"
        "- You MUST output BOTH a <<<PLAN>>> JSON block and a <<<CODE>>> block in this order.\n"
        "- If MODE=code and you do not include a non-empty <<<CODE>>> block, the output is INVALID.\n"
        "ELSE If MODE=explain (for STEP 2B):\n" 
        " - do NOT put the PLAN JSON in the <<<ANSWER>>> block. Put the answer to the user question in <<<ANSWER>>> block.\n"
        # "- Forbidden phrases outside blocks: 'We need to', 'Thus', 'So', 'Plan JSON', 'I will', 'Let us'.\n"
        "\n"
        "STRICT OUTPUT FORMAT (use these exact tagged blocks):\n"
        "<<<MODE>>>{code|explain}<<<END>>>\n"
        "<<<PLAN>>>{ \"endpoint\": \"/<path>\", \"params\": {\"<param>\": \"<value>\"}, \"chunk_rule\": \"<monthly|yearly|decade or empty>\", \"plot_rule\": \"<timeseries|map or empty>\"}<<<END>>>\n"
        "<<<CODE>>>\n<single runnable python script>\n<<<END>>>\n"
        "<<<ANSWER>>>\n<markdown or text answer>\n<<<END>>>\n"
    )
    notes_line = "RAG NOTES:\n" + f"{rag_notes or '(none)'}"
    prev_line  = "Previous plan (Keep the parameters if user not intent to vary them but want to follow up previous question):\n" + (json.dumps(prev_plan, ensure_ascii=False) if prev_plan else "(none)")
    oas_blob   = {
        "paths": oas_info.get("paths", []),
        "params": oas_info.get("params", []),
        "append_allowed": oas_info.get("append", []),
        "param_enums": oas_info.get("param_enums", {}),
    }

    return (
        "You are an ODB assistant that does Classifier, Planner, Coder, and Explainer in ONE pass.\n"
        + "Think step-by-step INTERNALLY, but NEVER print your reasoning.\n"
        + "Only print the tagged blocks and contents within the tagged blocks defined in STRICT OUTPUT FORMAT.\n"        
        + "You must decide and output the content of tagged blocks conditionally for each STEP:\n"
        + "STEP 1 (Classsifier for <<<MODE>>> content) — " + classifier_core  + "\n"
        + "(The following STEP 2A and STEP 2B are exclusive. The STEP 3 is following STEP 2A)\n"
        + "IF MODE=code THEN STEP 2A (Planner for <<<PLAN>>> content, ONLY when MODE=code) — " + planner_core + "\n"
        + "- THEN STEP 3 (Coder for <<<CODE>>> content, ONLY when MODE=code) — " + code_core + "\n"
        + "ELSE IF MODE=explain THEN STEP 2B (Explainer for <<<ANSWER>>> content, ONLY when MODE=explain) — " + explain_core + "\n"
        + strict_format
        + "\n"
        + "---- CONTEXT (reference for decisions) ----\n"
        + f"{notes_line}\n"
        + "\n"
        + f"Question:\n{question}\n"
        + "\n"
        + f"{prev_line}\n"
        + "\n"
        + "OAS whitelist (for planning ONLY):\n"
        + f"{json.dumps(oas_blob, ensure_ascii=False)}"
    )

def llm_one_pass_decide_plan_and_code(
    question: str,
    oas: Dict[str,Any],
    rag_notes: str,
    llm: str,
    chunk_line: Optional[str] = None,
    history: Optional[List[Tuple[str,str]]] = None,
    debug: bool = False,
    prev_plan: Optional[Dict[str, Any]] = None
) -> Dict[str,Any]:
    """
    單次 LLM：先判斷 MODE，再在同一回合給 PLAN / CODE 或 ANSWER。
    回傳欄位：
      - mode: "code" | "explain"
      - plan: dict | None
      - chunk_rule: str | None
      - plot_rule: str | None
      - code: str | None
      - explain: str | None
      - raw: 原始 LLM 輸出
      - log: LLM 自述判斷原因
    """
    force_zh = bool(re.search(r"繁體|正體|繁體中文", question)) or bool(re.search(r"[\u4e00-\u9fff]", question))
    sysrule = build_one_pass_sysrule(
        question=question,
        rag_notes=rag_notes,
        oas_info=oas,
        prev_plan=prev_plan,
        force_zh=force_zh,
        chunk_hint_line=chunk_line,
        debug=debug
    )

    user_parts: List[str] = []
    if rag_notes:
        user_parts.append("[RAG NOTES]\n" + (rag_notes or "(none)"))

    hist_text = ""
    try:
        if history and "_render_history_for_llm" in globals():
            hist_text = _render_history_for_llm(history, max_pairs=3, max_chars=800)
    except Exception:
        hist_text = ""
    if hist_text:
        user_parts.append("HISTORY (condensed):\n" + hist_text)

    user_parts.append("QUESTION:\n" + question)
    user_msg = "\n\n".join(user_parts)

    if debug:
        print("[DEBUG] use one-pass LLM")

    txt = run_llm(
        llm,
        [{"role":"system","content":sysrule},
         {"role":"user","content":user_msg}],
        timeout=600.0
    ) or ""

    blocks = parse_strict_blocks(txt)
    mode   = (blocks.get("mode") or "").strip().lower()
    planj  = blocks.get("plan", "") or ""
    code_b = blocks.get("code", "") or ""
    ans_b  = blocks.get("answer", "") or ""
    log_b  = blocks.get("log", "") or ""
    mhw_b  = blocks.get("mhw_rule", "") or "" 
    plot_b  = blocks.get("plot_rule", "") or "" 

    if debug:
        keys_present = ", ".join([k for k in ["mode","plan","code","answer","log","mhw_rule"] if blocks.get(k)])
        print(f"[DEBUG] parse_strict_blocks → keys: {keys_present} | mode={mode}")
        print(f"[DEBUG] MODE={mode} | plan_len={len(planj)} | code_len={len(code_b)} | answer_len={len(ans_b)} | log_len={len(log_b)}")
        if log_b:
            print(f"[DEBUG] LOG block:\n{log_b}")
        if planj:
            print(f"[DEBUG] PLAN block:\n{planj}")
        if mhw_b:
            print(f"[DEBUG] MHW_RULE block:\n{mhw_b}")
        if plot_b:
            print(f"[DEBUG] PLOT_RULE block:\n{plot_b}")

    out: Dict[str,Any] = {
        "mode": mode or "",
        "plan": None,
        "chunk_rule": None,
        "plot_rule": None,
        "code": None,
        "explain": None,
        "raw": txt,
        "log": log_b
    }

    out: Dict[str,Any] = {
        "mode": mode or "",
        "plan": None,
        "chunk_rule": None,   # NEW
        "code": None,
        "explain": None,
        "raw": txt,
        "log": log_b
    }

    if mode == "code":
        plan_obj: Optional[Dict[str,Any]] = None
        if planj:
            try:
                plan_obj = json.loads(planj)
            except Exception:
                m = re.search(r'\{[\s\S]*\}', planj)
                if m:
                    try:
                        plan_obj = json.loads(m.group(0))
                    except Exception:
                        plan_obj = None
        out["plan"] = plan_obj

        # NEW: 抓取 chunk_rule（PLAN 優先，否則用 MHW_RULE 區塊）
        plan_chunk = None
        if isinstance(plan_obj, dict):
            plan_chunk = plan_obj.get("chunk_rule")
            if isinstance(plan_chunk, (list, dict)):
                plan_chunk = json.dumps(plan_chunk, ensure_ascii=False)
        if not plan_chunk and mhw_b:
            plan_chunk = mhw_b
        if isinstance(plan_chunk, str):
            plan_chunk = plan_chunk.strip()
        out["chunk_rule"] = plan_chunk or None

        code_text = extract_code_from_markdown(code_b).strip()
        out["code"] = code_text
        return out

    elif mode == "explain":
        cleaned = ans_b
        if 'ONE_PASS_MARKER_RE' in globals() and isinstance(ONE_PASS_MARKER_RE, re.Pattern):
            cleaned = ONE_PASS_MARKER_RE.sub(' ', cleaned)
        if 'EXPLAIN_TOKEN_RE' in globals() and isinstance(EXPLAIN_TOKEN_RE, re.Pattern):
            cleaned = EXPLAIN_TOKEN_RE.sub(' ', cleaned)
        out["explain"] = cleaned.strip()
        return out

    # fallback → explain
    fallback = ans_b or txt
    if 'ONE_PASS_MARKER_RE' in globals() and isinstance(ONE_PASS_MARKER_RE, re.Pattern):
        fallback = ONE_PASS_MARKER_RE.sub(' ', fallback)
    if 'EXPLAIN_TOKEN_RE' in globals() and isinstance(EXPLAIN_TOKEN_RE, re.Pattern):
        fallback = EXPLAIN_TOKEN_RE.sub(' ', fallback)
    out["mode"] = "explain"
    out["explain"] = (fallback or "").strip()
    return out

def answer_once(
    question: str,
    llm: str,
    topk: int = 6,
    temp: float = 0.2,
    max_tokens: int = 800,
    strict_api: bool = True,
    debug: bool = False,
    history: List[Tuple[str, str]] | None = None,
    ctx_only: bool = False,
    force_mode: str | None = None,
    prev_plan: Optional[Dict[str, Any]] = None
) -> str:
    # 1) RAG
    hits = query_qdrant(question, debug=debug)
    globals()["LAST_HITS"] = hits
    if isinstance(topk, int) and topk > 0 and len(hits) > topk:
        hits = hits[:topk]
    if debug:
        kinds = [(h.get("payload", {}) or {}).get("doc_type", "n/a").lower() for h in hits]
        print(f"[DEBUG] RAG hits total: {len(hits)}; type distribution: {Counter(kinds) or 'n/a'}")

    # 2) OAS
    oas_info = harvest_oas_whitelist(hits, debug=debug)
    _set_oas_cache(oas_info)

    # 3) RAG 摘要 —— one-pass 使用「原始 hits ∪ rerank(code) ∪ rerank(explain)」
    hits_for_one_pass = prepare_rag_for_one_pass(hits, question, topk=topk, debug=debug)
    if debug:
        print(f"[DEBUG] RAG for one-pass: {len(hits_for_one_pass)} candidates after union (base+rerank).")
    rag_notes = collect_rag_notes(hits_for_one_pass, max_chars=1400, debug=debug)

    force_multi = (os.getenv("MULTI_STEPS", "0") == "1")
    init_chunk_line = _mhw_chunk_general_rule()
    if not force_multi:
        if debug:
            print("[DEBUG] path=one-pass | entering")
        op = llm_one_pass_decide_plan_and_code(
            question=question,
            oas=oas_info,
            rag_notes=rag_notes,
            llm=llm,
            chunk_line=init_chunk_line,
            history=history,
            debug=debug,
            prev_plan=prev_plan
        )
        if debug and op.get("mode") == "code" and op.get("code")=="":
           print("[DEBUG] one-pass | code missing → fallback to multi-step codegen")

        if op.get("mode") == "explain":
            if debug:
                print("[DEBUG] path=one-pass | success=explain")
            cites = format_citations(hits, question)
            return f"{op['explain']}\n\n=== 參考資料 ===\n{cites}"

        if op.get("mode") == "code" and (op.get("plan") and op.get("code") is not None):
            ok, msg = validate_plan(op["plan"], oas_info)
            if debug and not ok:
                print(f"[DEBUG] one-pass plan invalid: {msg} — falling back to multi-step.")
            else:
                code = op["code"] or ""
                # 若 code 不完整 → 續寫一次，使用 planner 給的 chunk_rule（若有）
                if not _looks_like_complete_code(code, debug=debug):
                    if debug:
                        print("[DEBUG] one-pass code incomplete → try single continuation")
                    prefer_chunk = op.get("chunk_rule") or build_chunk_line_from_plan(op["plan"]) or init_chunk_line
                    more = code_continue_via_llm(
                        question=question,
                        plan=op["plan"],
                        last_code=code, 
                        llm=llm,
                        debug=debug,
                        chunk_line=prefer_chunk,
                        oas=oas_info
                    )
                    if more:
                        code = merge_code_fences(code.rstrip() + "\n\n", more.lstrip())
                ok2, msg2 = static_guard_check(code, oas_info, expect_csv=False, expect_map=False)
                if debug and not ok2:
                    print(f"[DEBUG] one-pass code violates guard: {msg2} — falling back to multi-step.")
                else:
                    if debug:
                        print("[DEBUG] path=one-pass | success=code")
                    cites = format_citations(hits, question)
                    globals()["LAST_CODE_TEXT"] = code
                    return f"{code}\n\n=== 參考資料 ===\n{cites}"

        if debug:
            print("[DEBUG] path=one-pass | fallback to multi-step")

    # ====== Multi-Steps（僅當 MULTI_STEPS=1 或 one-pass 失敗時） ======
    if debug:
        print("[DEBUG] path=multi-step | entering")

    mode = force_mode or llm_choose_mode(question, llm=llm, debug=debug)
    if debug:
        print(f"[DEBUG] mode: {mode} (csv_requested=False)")

    if mode == "explain":
        last_code = globals().get("LAST_CODE_TEXT") or ""
        if last_code and re.search(r"(解釋|說明).*(程式|code)|加.*註解|comment", question, flags=re.I):
            if debug:
                print("[DEBUG] explain->annotate | use Continue LLM to add comments")
            annotated = code_continue_via_llm(
                question="請在現有程式碼中加入中文註解，逐段解釋每個步驟與參數的用途，保持原有程式不變並僅加註解。",
                plan=prev_plan or {},
                last_code=last_code,
                llm=llm,
                debug=debug,
                chunk_line=init_chunk_line,
                oas=oas_info                
            ) or ""
            if annotated.strip():
                cites = format_citations(hits, question)
                return f"{annotated}\n\n=== 參考資料 ===\n{cites}"

        ans = explain_via_llm(question, hits, llm=llm, debug=debug)
        cites = format_citations(hits, question)
        if debug:
            print("[DEBUG] path=multi-step | success=explain")
        return f"{ans}\n\n=== 參考資料 ===\n{cites}"

    # mode == code
    plan = llm_json_plan(question, oas_info, llm=llm, prev_plan=prev_plan, debug=debug)
    plan = inherit_from_prev_if_reasonable(plan, prev_plan, oas_info, csv_requested=False, debug=debug)

    ok, msg = validate_plan(plan, oas_info)
    if debug and not ok:
        print(f"[DEBUG] Plan invalid@second: {msg}")

    code = code_from_plan_via_llm(
        question, plan, llm=llm, debug=debug,
        chunk_line=(build_chunk_line_from_plan(plan) or init_chunk_line),
        oas=oas_info
    ) or ""
    ok2, msg2 = static_guard_check(code, oas_info, expect_csv=False, expect_map=False)
    if not ok2 and debug:
        print(f"[DEBUG] Code violates guard: {msg2}")

    globals()["LAST_CODE_TEXT"] = code
    cites = format_citations(hits, question)
    if debug:
        print("[DEBUG] path=multi-step | success=code")
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
    ap.add_argument("--chat", action="store_true", default=False, help="互動模式")
    ap.add_argument("--multi-steps", action="store_true", default=False, help="強制使用舊式多步驟管線（除錯用）")
    args = ap.parse_args()

    if not args.question and not args.chat:
        print("用法：rag_cli.py '你的問題'  或  rag_cli.py --chat")
        return

    history: List[Tuple[str,str]] = []
    if args.multi_steps:
        os.environ["MULTI_STEPS"] = "1"

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
                              ctx_only=False)
            print("\n=== 答覆 ===\n")
            print(out)
            history.append((q, out))
    else:
        out = answer_once(args.question,
                          llm=("ollama" if args.llm=="ollama" else ("llama" if args.llm=="llama" else "ollama")),
                          topk=args.topk, temp=args.temp, max_tokens=args.max_tokens,
                          strict_api=args.strict_api, debug=args.debug, history=None, ctx_only=False)
        print("\n=== 答覆 ===\n")
        print(out)

if __name__ == "__main__":
    main()
