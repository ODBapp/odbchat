#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import math
import argparse
import asyncio
import warnings
from typing import Any, Dict, List, Tuple, Optional, Iterable
from collections import Counter, defaultdict

warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# =========================
# Config (env)
# =========================
QDRANT_HOST    = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION     = os.environ.get("QDRANT_COL", "odb_mhw_knowledge_v1")

EMBED_MODEL    = os.environ.get("EMBED_MODEL", "thenlper/gte-small")
EMBED_DIM      = 384

# 兩種 LLM 後端：ollama / llama.cpp server
OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "300"))   # 放寬

LLAMA_URL      = os.environ.get("LLAMA_URL", "http://localhost:8001/completion")

# =========================
# Init
# =========================
qdrant   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

# =========================
# Lightweight heuristics
# =========================
API_KEYWORDS   = ["api","openapi","oas","endpoint","/api/","參數","規格","swagger","parameters","應用程式介面"]
CODE_KEYWORDS  = ["code","程式","python","範例","example","下載","csv","plot","畫圖","抓資料","implement","實做","方法","時序圖","時間序列","地圖","繪圖","資料","分析","趨勢"]
ECOSYS_KEYS    = ["ecosystem","生態","生態系","生態系統"]
ENSO_TOKENS    = ["enso","聖嬰","反聖嬰","el niño","la niña","enso basics"]

def _lower(s: Optional[str]) -> str:
    return (s or "").lower()

def detect_api_intent(q: str) -> bool:
    ql = q.lower()
    if any(k in ql for k in API_KEYWORDS): return True
    if "python" in ql: return True
    return False

def detect_code_intent(q: str) -> bool:
    ql = q.lower()
    if any(k in ql for k in CODE_KEYWORDS): return True
    # 合併中文字中間空白：避免「畫 時序圖」、「畫 地圖」抓不到
    nospace = ql.replace(" ", "")
    for kw in ["畫圖","畫地圖","時序圖","時間序列","繪圖"]:
        if kw in nospace: return True
    return False

def detect_ecosys(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ECOSYS_KEYS)

def detect_enso(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ENSO_TOKENS)

# -------- follow-up --------
FOLLOWUP_TOKENS = [
    "再做一次","改成","改為","只要json","只要 json","換成","接續","繼續","延續",
    "上一題","上述","剛剛","同樣","同上","follow-up","不要 csv","改用 json"
]

def detect_followup(question: str, history: Optional[List[Tuple[str,str]]]) -> Optional[Dict[str,str]]:
    if not history:
        return None
    ql = question.lower()
    if not any(tok in ql for tok in [t.lower() for t in FOLLOWUP_TOKENS]):
        # 即使沒有明顯關鍵詞，只要問題很短且含「json/再次/改/只要」也可能是追問
        if len(question) > 32:
            return None
    # 取上一輪
    prev_q, prev_a = history[-1]
    # 嘗試抓出上一輪是否使用了 API endpoint
    ep = None
    m = re.search(r'https?://[^\s"\'<>]+/api/[^\s"\'<>]+', prev_a)
    if m:
        ep = m.group(0)
    # 壓縮上一輪答案避免過長
    prev_a_compact = re.sub(r"```[\s\S]*?```", "[code omitted]", prev_a)
    prev_a_compact = re.sub(r"\s+", " ", prev_a_compact).strip()
    if len(prev_a_compact) > 600:
        prev_a_compact = prev_a_compact[:600] + "…"
    return {"prev_q": prev_q, "prev_a": prev_a_compact, "prev_ep": ep or ""}

# =========================
# Helpers
# =========================
def get_payload_text(payload: Dict[str, Any]) -> str:
    return payload.get("text") or payload.get("content") or ""

def approx_tokens(s: str) -> int:
    return max(1, math.ceil(len(s)/3))

def clamp_context(chunks: List[Dict[str, Any]], max_ctx_tokens: int, reserve: int=512) -> List[Dict[str, Any]]:
    out, used = [], 0
    budget = max(256, max_ctx_tokens - reserve)
    for h in chunks:
        t = h.get("text", "")
        tk = approx_tokens(t)
        if used + tk > budget:
            break
        out.append(h); used += tk
    return out

def dedupe_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for h in hits:
        p = h.get("payload", {}) or {}
        key = p.get("doc_id") or p.get("canonical_url") or p.get("source_file") or p.get("title") or get_payload_text(p)[:64]
        if key in seen: continue
        seen.add(key); out.append(h)
    return out

def encode_query(q: str) -> List[float]:
    return embedder.encode([q], normalize_embeddings=True)[0].tolist()

def _title_tags_content(payload: Dict[str, Any]) -> Tuple[str, List[str], str]:
    title = _lower(payload.get("title"))
    tags  = [_lower(t) for t in (payload.get("tags") or [])]
    content = _lower(get_payload_text(payload))
    return title, tags, content

def keyword_boost(q: str, payload: Dict[str, Any], debug: bool=False) -> float:
    score = 0.0
    title, tags, content = _title_tags_content(payload)
    ql = q.lower()

    # 主題（MHW）
    for kw in ["marine heatwaves","mhw","海洋熱浪","marine heatwaves (mhw)"]:
        if (kw in (title or "")) or any(kw in t for t in tags) or (kw in content):
            score += 0.6; break

    # 生態系問題 → 偏好含生態關鍵詞
    if detect_ecosys(q):
        for kw in ECOSYS_KEYS:
            if (kw in (title or "")) or any(kw in t for t in tags) or (kw in content):
                score += 0.6; break

    # API / 程式 / 方法 / 繪圖
    if detect_api_intent(q) or detect_code_intent(q):
        #if debug: print(f"[DEBUG] hits for detecting code needed in question")
        if (payload.get("doc_type") in ("api_spec","code_snippet","code_example","cli_tool_guide")):
            score += 1.2

    # 分級
    if any(kw in ql for kw in ["level","分級","category","categorize","等級"]):
        if any(k in content for k in ["hobday","海洋熱浪分級標準","categories of severity","90th percentile",
                                      "extreme","severe","moderate","strong","極端","嚴重","中等","強烈"]):
            score += 0.8

    # 問題不含 ENSO → 強降權 ENSO 文件
    if not detect_enso(q):
        if any(t in ["enso","el niño","la niñas","la niña","enso basics"] for t in tags) or ("enso" in (title or "")):
            score -= 2.0

    # CLI 工具文件：只有問題真的提 CLI/指令才加分，否則降權
    if payload.get("doc_type") == "cli_tool_guide":
        if any(k in ql for k in ["cli","指令","命令","mhw_plot","odbchat","工具","不用寫程式","no code"]):
            score += 0.5
        else:
            score -= 1.0
            
    if debug: print(f"[DEBUG] hits in rerank: {payload.get('title')} with score: {score}")         
    return score

def rerank_with_boost(q: str, hits: List[Dict[str, Any]], debug: bool=False) -> List[Dict[str, Any]]:
    alpha = 10.0
    scored = []
    n = len(hits)
    for i, h in enumerate(hits):
        base = (n - i)
        b = keyword_boost(q, h.get("payload", {}) or {}, debug=debug)
        scored.append((base + alpha*b, i, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [x[2] for x in scored]
    return out

def source_category(p: Dict[str, Any]) -> str:
    dt = (p.get("doc_type") or "").lower()
    st = (p.get("source_type") or "").lower()
    if dt == "api_spec": return "api"
    if dt in ("code_snippet","code_example") or "code" in st: return "code"
    if dt in ("paper_note","paper"): return "paper"
    if dt in ("manual","manuals","api_guide","api_spec_guide","tutorial","cli_tool_guide"): return "guide"
    if dt in ("web_article","web"): return "web"
    return "other"

def diversify_sources(question: str, hits: List[Dict[str, Any]], topk: int, debug: bool=False) -> List[Dict[str, Any]]:
    """限制某些類別的過度壟斷（避免永遠是 web_docs 那幾篇）"""
    need_api  = detect_api_intent(question)
    need_code = detect_code_intent(question)
    no_enso   = not detect_enso(question)

    # 類別上限（適度保守）
    caps = {"web": 2, "guide": 1}
    # 若是 API/程式意圖，至少保證 api/code 能進來（如果命中有）
    min_need = []
    if need_api:  min_need.append("api")
    if need_code: min_need.append("code")

    buckets = defaultdict(list)
    for h in hits:
        cat = source_category(h.get("payload", {}) or {})
        buckets[cat].append(h)

    # 先放必要類別
    out = []
    used = set()
    for mcat in min_need:
        if buckets[mcat]:
            out.append(buckets[mcat].pop(0))
            used.add(id(out[-1]))

    # 再輪流放其他類別，遵守 caps
    counts = Counter()
    for h in hits:
        if id(h) in used:
            continue
        p = h.get("payload", {}) or {}
        cat = source_category(p)
        # 問題不含 enso → 首輪跳過 enso 標籤（後面若不足再補）
        tags = [ (t or "").lower() for t in (p.get("tags") or []) ]
        if no_enso and any(t in ["enso","el niño","la niña","enso basics"] for t in tags):
            continue
        if counts[cat] >= caps.get(cat, 10):  # 類別達上限
            continue
        out.append(h)
        counts[cat] += 1
        if len(out) >= topk:
            break

    # 如果因為排除 ENSO 或 caps 導致不足，補齊
    # if debug: print(f"[DEBUG] hits in diversify: {counts} with no_enso: {no_enso} and result: {out}")
    topm = topk-2
    if len(out) >= topk:
        return out[:topk]
    elif len(out) < topm:
        for h in hits:
            if id(h) in used or h in out:
                continue
            out.append(h)
            if len(out) >= topm:
                break
    return out

# =========================
# Qdrant query
# =========================
def query_qdrant(question: str, topk: int=5, debug: bool=False) -> List[Dict[str, Any]]:
    vec = encode_query(question)
    resp = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=max(20, topk*3),           # 放寬：提升候選多樣性
        with_payload=True,
        with_vectors=False,
        query_filter=None,
    )
    points = getattr(resp, "points", resp)
    hits = [{"text": get_payload_text(getattr(p, "payload", {}) or {}),
             "payload": getattr(p, "payload", {}) or {}} for p in points]
    hits = dedupe_hits(hits)

    # 重排 + 多樣化（含你要的 debug）
    reranked = rerank_with_boost(question, hits, debug=debug)
    if debug:  print(f"[DEBUG] hits after rerank: {[h['payload'].get('title') for h in reranked]}")
    diversified = diversify_sources(question, reranked, topk, debug=debug)
    if debug:  print(f"[DEBUG] hits after diversify: {[h['payload'].get('title') for h in diversified]}")

    if debug:
        dist = Counter([(h["payload"].get("title") or h["payload"].get("source_file") or "Unknown") for h in diversified])
        print(f"[DEBUG] source distribution: {json.dumps(dict(dist), ensure_ascii=False)}")
    return diversified

# =========================
# Code block extraction
# =========================
FENCE_ANY   = re.compile(r"```[a-zA-Z]*\s*([\s\S]*?)```", re.MULTILINE)
INDENT_CODE = re.compile(r"(?m)^(?: {4}|\t).+$")

def _extract_after_codeblock_header(text: str) -> List[str]:
    results = []
    m = re.search(r"(?im)^\s*#{1,6}\s*Code\s+block\s*$", text or "")
    if not m:
        return results
    start = m.end()
    tail = text[start:]
    stop = re.search(r"(?im)^\s*#{1,6}\s+\S+", tail)
    segment = tail[: stop.start()] if stop else tail
    fences = FENCE_ANY.findall(segment)
    if fences:
        results.extend([b.strip() for b in fences if b.strip()])
    else:
        lines = segment.splitlines()
        buf, cur = [], []
        for ln in lines:
            if INDENT_CODE.match(ln):
                cur.append(ln[4:] if ln.startswith("    ") else ln.lstrip())
            else:
                if cur:
                    buf.append("\n".join(cur).strip()); cur = []
        if cur:
            buf.append("\n".join(cur).strip())
        results.extend([b for b in buf if b])
    return results

def extract_code_blocks(texts: Iterable[str], payloads: Iterable[Dict[str, Any]]) -> List[str]:
    out = []
    for t, p in zip(texts, payloads):
        t = t or ""
        for m in FENCE_ANY.finditer(t):
            block = m.group(1).strip()
            if block:
                out.append(block)
        out.extend(_extract_after_codeblock_header(t))
        tags = [ (p.get("doc_type") or "").lower() ] + [ (x or "").lower() for x in (p.get("tags") or []) ]
        if "code" in (p.get("source_type") or "").lower() or any(k in tags for k in ["code_snippet","code","examples","example"]):
            if not FENCE_ANY.search(t):
                lines = t.splitlines()
                buf, cur = [], []
                for ln in lines:
                    if INDENT_CODE.match(ln):
                        cur.append(ln[4:] if ln.startswith("    ") else ln.lstrip())
                    else:
                        if cur:
                            buf.append("\n".join(cur).strip()); cur = []
                if cur:
                    buf.append("\n".join(cur).strip())
                out.extend([b for b in buf if b])
    return out

def debug_scan_code_blocks(hits: List[Dict[str, Any]], debug: bool=False):
    texts = [h.get("text","") for h in hits]
    payloads = [h.get("payload", {}) or {} for h in hits]
    blocks = extract_code_blocks(texts, payloads)
    if debug:
        code_docs = sum(1 for p in payloads if (p.get("doc_type") == "code_snippet") or ("code" in (p.get("source_type") or "").lower()))
        print(f"[DEBUG] code docs scanned: {code_docs}, code blocks collected: {len(blocks)}")

# =========================
# OAS parsing / harvesting
# =========================
PATH_KEY_RE = re.compile(r'(?m)^\s{2,}(/[^:\s]+)\s*:\s*$')
PATH_BLOCK_FOR_TEMPLATE = (
    r'(?ms)^\s{{2,}}{path}\s*:\s*\n'
    r'(\s{{4,}}.*?)(?=^\s{{2,}}/|\Z)'
)
PARAM_NAME_RE = re.compile(r'(?m)^\s*-\s*name:\s*([A-Za-z0-9_]+)\s*$')
APPEND_ALLOWED_ANY_RE = re.compile(
    r"Allowed\s+fields\s*:\s*([\"'][^\"']+[\"'](?:\s*,\s*[\"'][^\"']+[\"'])*|\w+(?:\s*,\s*\w+)*)",
    re.IGNORECASE
)

def _unquote(s: str) -> str:
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    return s

def _split_append_allowed(s: str) -> List[str]:
    s = s.replace(" and ", ",")
    parts = [p.strip() for p in s.split(",")]
    return [_unquote(p) for p in parts if p.strip()]

def parse_oas_params(oas_yaml: str) -> Dict[str, Any]:
    if not oas_yaml:
        return {"params": [], "append_allowed": [], "paths": []}
    params, allowed, paths = [], [], []

    for m in PATH_KEY_RE.finditer(oas_yaml):
        p = m.group(1).strip()
        if p not in paths:
            paths.append(p)

    for p in paths:
        pat = PATH_BLOCK_FOR_TEMPLATE.format(path=re.escape(p))
        m = re.search(pat, oas_yaml)
        if not m:
            continue
        body = m.group(1)

        for name in PARAM_NAME_RE.findall(body):
            if name not in params:
                params.append(name)

        if "name: append" in body.lower():
            for ma in APPEND_ALLOWED_ANY_RE.finditer(body):
                for it in _split_append_allowed(ma.group(1)):
                    if it and it not in allowed:
                        allowed.append(it)

    if not params:
        for n in PARAM_NAME_RE.findall(oas_yaml):
            if n not in params:
                params.append(n)
    if not allowed:
        for ma in APPEND_ALLOWED_ANY_RE.finditer(oas_yaml):
            for it in _split_append_allowed(ma.group(1)):
                if it and it not in allowed:
                    allowed.append(it)

    return {"params": sorted(params), "append_allowed": sorted(allowed), "paths": sorted(paths)}

def collect_api_specs(debug: bool=False) -> List[Dict[str, Any]]:
    flt = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="api_spec"))])
    points, _ = qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=flt,
        with_payload=True,
        with_vectors=False,
        limit=1000
    )
    out = []
    for p in points:
        payload = getattr(p, "payload", {}) or {}
        out.append({"text": get_payload_text(payload), "payload": payload})
    if debug:
        print(f"[DEBUG] Scanned api_spec docs: {len(out)}")
    return out

def harvest_oas_whitelist(api_specs: List[Dict[str, Any]], hits: List[Dict[str, Any]], debug: bool=False) -> Dict[str, Any]:
    all_params, all_append, all_paths = set(), set(), set()
    for spec in api_specs:
        meta = parse_oas_params(spec.get("text",""))
        all_params.update(meta["params"])
        all_append.update(meta["append_allowed"])
        all_paths.update(meta["paths"])

    rebuilt = 0
    if not all_params:
        for h in hits:
            txt = h.get("text","")
            if "openapi:" in txt or "paths:" in txt:
                meta = parse_oas_params(txt)
                if meta["params"] or meta["append_allowed"]:
                    all_params.update(meta["params"])
                    all_append.update(meta["append_allowed"])
                    all_paths.update(meta["paths"])
                    rebuilt += 1
    if debug:
        print(f"[DEBUG] OAS params: {sorted(all_params)}")
        print(f"[DEBUG] OAS append allowed: {sorted(all_append)}")
        if all_paths:
            print(f"[DEBUG] OAS paths found: {sorted(all_paths)}")
        print(f"[DEBUG] api_spec docs scanned (rebuilt): {rebuilt}")
    return {"params": sorted(all_params), "append_allowed": sorted(all_append), "paths": sorted(all_paths)}

# =========================
# Prompt & LLM
# =========================
def pick_need_code(question: str) -> bool:
    return detect_code_intent(question)

def build_prompt(question: str,
                 ctx: List[Dict[str, Any]],
                 need_code: bool,
                 oas_info: Optional[Dict[str, Any]],
                 strict_api: bool,
                 followup: Optional[Dict[str,str]]=None) -> str:
    sys_rule = (
        "你是 ODB（海洋學門資料庫）助理。請只根據『依據』內容回答；沒有依據就回答「無法在已知資料中找到答案。」"
        "不得虛構 API 參數或值。不要在正文貼『內部檢索片段』、『引用清單』或『=== 引用 ===』。不要重述問題，直接作答。"
    )
    if need_code:
        sys_rule += "若提供程式，請用 ```python 包覆，並依 OpenAPI 規格使用正確的 query 參數。"
    if strict_api and oas_info:
        sys_rule += f"【API 白名單】只能使用：{', '.join(oas_info.get('params', []))}；append 可用：{', '.join(oas_info.get('append_allowed', []))}。"

    ctx_text = ""
    for i, h in enumerate(ctx, 1):
        m = h.get("payload", {}) or {}
        title = m.get("title") or m.get("source_file") or "Unknown"
        ctx_text += f"\n[來源 {i}] {title}\n{h.get('text','')}\n"

    follow = ""
    if followup:
        follow = "\n[前文（供參考，若不相干可忽略）]\n"
        follow += f"上一題：{followup['prev_q']}\n"
        if followup["prev_ep"]:
            follow += f"上一答使用 API：{followup['prev_ep']}\n"
        follow += f"上一答（節錄）：{followup['prev_a']}\n"

    return f"{sys_rule}\n{follow}\n問題：{question}\n\n依據：{ctx_text}\n\n請作答："

async def call_ollama(prompt: str, temperature: float=0.2, max_tokens: int=512) -> str:
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "options": {"temperature": temperature},
                    "stream": False
                }
            )
            if resp.status_code != 200:
                return f"[ERROR] Ollama HTTP {resp.status_code}: {resp.text}"
            data = resp.json()
            return (data.get("response","") or "").strip()
    except httpx.ReadTimeout:
        return "[ERROR] Ollama ReadTimeout，請檢查模型服務或提高 OLLAMA_TIMEOUT。"

async def call_llama(prompt: str, temperature: float=0.2, max_tokens: int=512) -> str:
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        resp = await client.post(
            LLAMA_URL,
            json={"prompt": prompt, "n_predict": max_tokens, "temperature": temperature}
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "content" in data:
            return data["content"]
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("text","")
        return str(data)

# =========================
# Post-processing
# =========================
CLEAN_PATTERNS = [
    (re.compile(r'(?im)^\s*<\s*/?(system|user|assistant)\s*>\s*$'), ""),
    (re.compile(r'(?im)^\s*引用清單\s*[:：]\s*$'), ""),
    (re.compile(r'(?im)^\s*===\s*引用\s*===\s*$'), ""),
    (re.compile(r'(?im)^===\s*內部檢索片段.*$', re.S), ""),
    (re.compile(r'(?im)^\s*append\s*[:：].*$'), ""),  # 移除「append：原問題」這種多餘行
]

def sanitize_answer(text: str) -> str:
    if not text: return text
    for pat, rep in CLEAN_PATTERNS:
        text = pat.sub(rep, text).strip()
    text = re.sub(r'(?is)\n+===\s*引用\s*===\s*\n+.*$', "", text).strip()
    text = re.sub(r'\n{3,}', "\n\n", text)

    # 去重
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    seen, out = set(), []
    for p in paras:
        key = p[:200]
        if key in seen: continue
        seen.add(key); out.append(p)

    # 避免只重述問題
    final = "\n\n".join(out).strip()
    return final

def post_fix_strict_api(text: str, oas_info: Dict[str, Any]) -> str:
    allowed = set(oas_info.get("params", []))
    allowed_append = set(oas_info.get("append_allowed", []))
    m = re.search(r"params\s*=\s*\{([\s\S]*?)\}", text)
    if not m: return text

    body = m.group(1)
    pairs = re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^\n,]+)', body)
    new_lines = []
    for k, v in pairs:
        line = f'    "{k}": {v},'
        if k not in allowed:
            line = f'    # "{k}": {v},  # removed (not in OpenAPI)'
        if k == "append":
            m2 = re.search(r'"([^"]*)"', v.strip())
            if m2:
                vals = [x.strip() for x in m2.group(1).split(",") if x.strip()]
                filtered = [x for x in vals if x in allowed_append]
                if not filtered:
                    line = f'    # "append": {v},  # removed (values not allowed)'
                else:
                    line = f'    "append": "{",".join(filtered)}",'
        new_lines.append(line)
    body_fixed = "\n".join(new_lines)
    return text[:m.start(1)] + body_fixed + text[m.end(1):]

def uniq_sources(hits: List[Dict[str, Any]], limit: int=5) -> List[Tuple[str,str]]:
    seen, outs = set(), []
    for h in hits:
        p = h.get("payload", {}) or {}
        title = p.get("title") or p.get("source_file") or "Unknown"
        url   = p.get("canonical_url") or p.get("source_file") or "Unknown"
        key = (title, url)
        if key in seen: continue
        seen.add(key); outs.append(key)
        if len(outs) >= limit: break
    return outs

def format_citations(hits: List[Dict[str, Any]], question: str = "") -> str:
    q = (question or "").lower()
    q_tokens = set(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", q))
    cites = []
    scored = []

    for h in hits:
        p = h.get("payload", {}) or {}
        title = p.get("title") or p.get("source_file") or "Unknown"
        url   = p.get("canonical_url") or p.get("source_file") or "Unknown"
        text  = (get_payload_text(p) or "").lower()
        tags  = " ".join([ (t or "").lower() for t in (p.get("tags") or []) ])
        bag   = f"{(title or '').lower()} {tags} {text}"
        bag_tokens = set(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", bag))

        overlap = len(q_tokens & bag_tokens)
        score = overlap

        if not detect_enso(question):
            if any(t in tags for t in ["enso","el niño","la niña","enso basics"]):
                score -= 3

        if p.get("doc_type") == "cli_tool_guide" and not any(k in q for k in ["cli","指令","命令","mhw_plot","odbchat"]):
            score -= 2

        scored.append((score, title, url))

    scored.sort(key=lambda x: x[0], reverse=True)
    uniq = []
    seen = set()
    for s, t, u in scored:
        key = (t, u)
        if key in seen: continue
        seen.add(key)
        if s < 0:
            continue
        uniq.append((t,u))
        if len(uniq) >= 5: break

    if not uniq:
        return "（無）"
    return "\n".join([f"[{i}] {t} — {u}" for i, (t,u) in enumerate(uniq, 1)])

# =========================
# Core QA
# =========================
async def answer_once(
    question: str,
    llm: str = "llama",
    topk: int = 5,
    temp: float = 0.2,
    max_tokens: int = 512,
    strict_api: bool = False,
    debug: bool = False,
    history: Optional[List[Tuple[str,str]]] = None,
    max_ctx_tokens: int = 3072,
    max_chunks: int = 5,
) -> str:

    # 1) 檢索
    hits = query_qdrant(question, topk=topk, debug=debug)

    # 2) 偵測追問（帶入簡短前文）
    follow = detect_followup(question, history)

    # 3) Debug: code blocks
    debug_scan_code_blocks(hits, debug=debug)

    # 4) 只有在 strict_api 或 問題帶 API/程式意圖時，才去掃 OAS
    need_code = detect_code_intent(question)
    need_api  = detect_api_intent(question)
    oas_info: Optional[Dict[str, Any]] = None

    if strict_api or need_api or need_code:
        api_specs = collect_api_specs(debug=debug)
        oas_info  = harvest_oas_whitelist(api_specs, hits, debug=debug)

    # 5) 控制上下文長度
    reserve = max_tokens + 800
    hits_ctx = clamp_context(hits, max_ctx_tokens=max_ctx_tokens, reserve=reserve)[:max_chunks]

    # 6) Prompt
    prompt = build_prompt(question, hits_ctx, need_code, oas_info if (strict_api or need_api or need_code) else None, strict_api, follow)

    # 7) Call LLM（context 過長 400 → 縮半重試一次）
    try:
        if llm == "ollama":
            raw = await call_ollama(prompt, temperature=temp, max_tokens=max_tokens)
        else:
            raw = await call_llama(prompt, temperature=temp, max_tokens=max_tokens)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400 and "context" in (e.response.text or "").lower():
            if debug: print("[DEBUG] llama 400; rebuild with smaller context and retry once …")
            hits_ctx = hits_ctx[: max(1, len(hits_ctx)//2) ]
            prompt = build_prompt(question, hits_ctx, need_code, oas_info if (strict_api or need_api or need_code) else None, strict_api, follow)
            if llm == "ollama":
                raw = await call_ollama(prompt, temperature=temp, max_tokens=max_tokens)
            else:
                raw = await call_llama(prompt, temperature=temp, max_tokens=max_tokens)
        else:
            raise

    # 8) 後處理
    answer = sanitize_answer(raw)
    if strict_api and oas_info:
        answer = post_fix_strict_api(answer, oas_info)

    # 9) 引用（依語義重疊過濾）
    cites = format_citations(hits_ctx, question)
    return f"{answer}\n\n=== 引用 ===\n\n{cites}"

# =========================
# Chat loop
# =========================
async def chat_loop(llm: str, topk: int, temp: float, max_tokens: int,
                    strict_api: bool, debug: bool, max_ctx_tokens: int, max_chunks: int):
    print("Enter '/exit' to quit. Ask your MHW questions.\n")
    history: List[Tuple[str,str]] = []
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if q in ("/exit","/quit"): break
        if not q: continue

        ans = await answer_once(q, llm, topk, temp, max_tokens,
                                strict_api, debug, history, max_ctx_tokens, max_chunks)
        print("\n=== 答覆 ===\n")
        print(ans); print()
        history.append((q, ans))

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="ODB MHW RAG CLI")
    ap.add_argument("question", nargs="?", default="", help="你的問題（--chat 時可省略）")
    ap.add_argument("--llm", choices=["ollama","llama"], default="llama")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-chunks", type=int, default=5, help="最多合併多少個 RAG 片段進 prompt")
    ap.add_argument("--ctx", type=int, default=3072, help="模型 context window 預算（用於裁切 RAG 片段）")
    ap.add_argument("--strict-api", action="store_true", help="嚴格依 OAS params/append 白名單修正 code 片段中的 params")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()

    if args.chat:
        if args.debug: print("[DEBUG] chat mode on")
        asyncio.run(chat_loop(args.llm, args.topk, args.temp, args.max_tokens,
                              args.strict_api, args.debug, args.ctx, args.max_chunks))
        return

    if not args.question:
        print("請輸入問題或改用 --chat")
        return

    out = asyncio.run(
        answer_once(args.question, args.llm, args.topk, args.temp, args.max_tokens,
                    args.strict_api, args.debug, None, args.ctx, args.max_chunks)
    )
    print("\n=== 答覆 ===\n")
    print(out)

if __name__ == "__main__":
    main()
