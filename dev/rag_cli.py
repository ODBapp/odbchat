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
from collections import Counter

# 避免舊驅動 CUDA 噴警告干擾
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# ----------------------
# Config (env)
# ----------------------
QDRANT_HOST   = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT   = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION    = os.environ.get("QDRANT_COL", "odb_mhw_knowledge_v1")

EMBED_MODEL   = os.environ.get("EMBED_MODEL", "thenlper/gte-small")
EMBED_DIM     = 384

OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT= float(os.environ.get("OLLAMA_TIMEOUT", "120"))

LLAMA_URL     = os.environ.get("LLAMA_URL", "http://localhost:8001/completion")

# ----------------------
# Init
# ----------------------
qdrant   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

# ----------------------
# Heuristics (minimal rules; focus on retrieval quality)
# ----------------------
API_KEYWORDS   = ["api", "openapi", "oas", "endpoint", "/api/", "參數", "規格", "swagger", "parameters", "應用程式介面"]
CODE_KEYWORDS  = ["code", "程式", "python", "範例", "example", "下載", "csv", "plot", "畫圖", "抓資料", "implement", "實做"]
ECOSYS_KEYS    = ["ecosystem", "生態", "生態系", "生態系統"]
ENSO_TOKENS    = ["enso", "聖嬰", "反聖嬰", "el niño", "la niña", "enso basics"]

def _lower(s: Optional[str]) -> str:
    return (s or "").lower()

def detect_api_intent(q: str) -> bool:
    ql = q.lower()
    if any(k in ql for k in API_KEYWORDS): return True
    if any(k in ql for k in CODE_KEYWORDS): return True
    if "python" in ql or "給 python" in ql: return True
    return False

def detect_code_intent(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in CODE_KEYWORDS) or "python" in ql

def detect_ecosys(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ECOSYS_KEYS)

def detect_enso(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ENSO_TOKENS)

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
    """小加分/扣分，避免『工人智慧』式 if-else；只調檢索排序，不影響 LLM 成文。"""
    score = 0.0
    title, tags, content = _title_tags_content(payload)
    ql = q.lower()

    # 主題命中（MHW）
    for kw in ["marine heatwaves", "mhw", "海洋熱浪", "marine heatwaves (mhw)"]:
        if (kw in (title or "")) or any(kw in t for t in tags) or (kw in content):
            score += 0.6; break

    # 生態系問題 → 偏好含生態關鍵詞的文
    if detect_ecosys(q):
        for kw in ECOSYS_KEYS:
            if (kw in (title or "")) or any(kw in t for t in tags) or (kw in content):
                score += 0.6; break

    # API / 程式問題 → 偏好 api_spec / code_snippet
    if detect_api_intent(q):
        if (payload.get("doc_type") in ("api_spec", "code_snippet")):
            score += 1.2

    # 偏好「分級」內涵（僅當問題提到）
    if any(kw in ql for kw in ["level", "分級", "category", "categorize", "等級"]):
        if any(k in content for k in ["hobday", "海洋熱浪分級標準", "categories of severity", "90th percentile",
                                      "extreme", "severe", "moderate", "strong", "極端", "嚴重", "中等", "強烈"]):
            score += 0.8

    # 若問題與 ENSO 無關，減少 ENSO 文件權重（避免常錯配）
    if not detect_enso(q):
        if any(t in ["enso", "el niño", "la niña", "enso basics"] for t in tags):
            score -= 1.0

    # 對 CLI 工具說明，只有問題真的提 CLI/指令時才升權重
    if payload.get("doc_type") == "cli_tool_guide":
        if any(k in ql for k in ["cli", "指令", "mhw_plot", "odbchat"]):
            score += 0.5
        else:
            score -= 1.0
    return score

def rerank_with_boost(q: str, hits: List[Dict[str, Any]], debug: bool=False) -> List[Dict[str, Any]]:
    """保留向量檢索的原順序，再加上 keyword_boost 作輕量重排，避免只打開 if-else。"""
    alpha = 10.0  # boost 權重（原順序為 base）
    scored = []
    n = len(hits)
    for i, h in enumerate(hits):
        base = (n - i)     # 原順序（越前越大）
        b = keyword_boost(q, h.get("payload", {}) or {}, debug=debug)
        scored.append((base + alpha * b, i, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[2] for x in scored]

def query_qdrant(question: str, topk: int=5, debug: bool=False) -> List[Dict[str, Any]]:
    vec = encode_query(question)
    resp = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=topk*2,            # 先抓寬一點，給 booster 重排空間
        with_payload=True,
        with_vectors=False,
        query_filter=None,
    )
    points = getattr(resp, "points", resp)
    hits = [{"text": get_payload_text(getattr(p, "payload", {}) or {}),
             "payload": getattr(p, "payload", {}) or {}} for p in points]
    hits = dedupe_hits(hits)
    hits = rerank_with_boost(question, hits, debug=debug)[:topk]

    if debug:
        dist = Counter([(h["payload"].get("title") or h["payload"].get("source_file") or "Unknown") for h in hits])
        print(f"[DEBUG] source distribution: {json.dumps(dict(dist), ensure_ascii=False)}")
    return hits

# --- Code block extraction (robust) ---

# ```python / ```py / ```（無語言）皆可
FENCE_ANY = re.compile(r"```[a-zA-Z]*\s*([\s\S]*?)```", re.MULTILINE)

# 四縮排 code（常見於轉義後的 Markdown）
INDENT_CODE = re.compile(r"(?m)^(?: {4}|\t).+$")

def _extract_after_codeblock_header(text: str) -> List[str]:
    # 從 "# Code block" 或 "## Code block" 往下抓到下一個標題或文末
    results = []
    m = re.search(r"(?im)^\s*#{1,6}\s*Code\s+block\s*$", text)
    if not m:
        return results
    start = m.end()
    tail = text[start:]
    stop = re.search(r"(?im)^\s*#{1,6}\s+\S+", tail)
    segment = tail[: stop.start()] if stop else tail
    # 優先抓反引號，否則抓四縮排
    fences = FENCE_ANY.findall(segment)
    if fences:
        results.extend([b.strip() for b in fences if b.strip()])
    else:
        # 合併連續的縮排行
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
        # 1) 先抓泛用 fenced
        for m in FENCE_ANY.finditer(t):
            block = m.group(1).strip()
            if block:
                out.append(block)
        # 2) 專抓 # Code block 段
        out.extend(_extract_after_codeblock_header(t))
        # 3) 若 source_type=code 或 tags 有 code/examples，抓四縮排塊
        tags = [ (p.get("doc_type") or "").lower() ] + [ (x or "").lower() for x in (p.get("tags") or []) ]
        if "code" in (p.get("source_type") or "").lower() or any(k in tags for k in ["code_snippet","code","examples","example"]):
            if not FENCE_ANY.search(t):  # 沒有 fenced，才抓縮排塊避免重複
                blocks = INDENT_CODE.findall(t)
                if blocks:
                    # 合併相鄰縮排行
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

# --- OAS parsing (robust) ---

# 寬鬆列出所有 paths 的 key（兩格以上縮排開頭、後接 /something:）
PATH_KEY_RE = re.compile(r'(?m)^\s{2,}(/[^:\s]+)\s*:\s*$')

# 注意：量詞用 {{ }} 轉義，只有 {path} 會被 format 替換
PATH_BLOCK_FOR_TEMPLATE = (
    r'(?ms)^\s{{2,}}{path}\s*:\s*\n'   # 行首2+空白  +  路徑  + 冒號換行
    r'(\s{{4,}}.*?)(?=^\s{{2,}}/|\Z)' # 從下一行的4+空白開始，到下一個同層級 path 或文末
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
    # 支援 'sst','level' / "sst", "level" / sst, level / sst and level
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

    return {
        "params": sorted(params),
        "append_allowed": sorted(allowed),
        "paths": sorted(paths),
    }

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

# ----------------------
# Prompt & LLM
# ----------------------
def pick_need_code(question: str) -> bool:
    return detect_code_intent(question)

def build_prompt(question: str, ctx: List[Dict[str, Any]], need_code: bool,
                 oas_info: Optional[Dict[str, Any]], strict_api: bool) -> str:
    sys_rule = (
        "你是 ODB（海洋學門資料庫）助理。請只根據『依據』內容回答；沒有依據就回答「無法在已知資料中找到答案。」"
        "不得虛構 API 參數或值。不要在正文貼『內部檢索片段』或自行輸出『引用清單/=== 引用 ===』。"
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

    return f"{sys_rule}\n\n問題：{question}\n\n依據：{ctx_text}\n\n請作答："

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

# ----------------------
# Post-processing
# ----------------------
CLEAN_PATTERNS = [
    (re.compile(r'(?im)^\s*<\s*/?(system|user|assistant)\s*>\s*$'), ""),
    (re.compile(r'(?im)^\s*引用清單\s*[:：]\s*$'), ""),
    (re.compile(r'(?im)^\s*===\s*引用\s*===\s*$'), ""),
    (re.compile(r'(?im)^===\s*內部檢索片段.*$', re.S), ""),
]

def sanitize_answer(text: str) -> str:
    if not text: return text
    for pat, rep in CLEAN_PATTERNS:
        text = pat.sub(rep, text).strip()
    text = re.sub(r'(?is)\n+===\s*引用\s*===\s*\n+.*$', "", text).strip()
    text = re.sub(r'\n{3,}', "\n\n", text)

    # 粗暴去重：避免同一段重覆
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    seen, out = set(), []
    for p in paras:
        key = p[:200]
        if key in seen: continue
        seen.add(key); out.append(p)
    return "\n\n".join(out)

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

        # 粗略重疊度
        overlap = len(q_tokens & bag_tokens)
        score = overlap

        # 問題不含 enso → 降權
        if not detect_enso(question):
            if any(t in tags for t in ["enso","el niño","la niña"]):
                score -= 3

        # 問題不含 cli/指令 → 降權
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
        if s < 0:  # 全負分就別列了
            continue
        uniq.append((t,u))
        if len(uniq) >= 5: break

    if not uniq:
        return "（無）"
    return "\n".join([f"[{i}] {t} — {u}" for i, (t,u) in enumerate(uniq, 1)])

# ----------------------
# Core QA
# ----------------------
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

    # 1) 檢索 + booster 重排
    hits = query_qdrant(question, topk=topk, debug=debug)

    # Debug: code blocks
    debug_scan_code_blocks(hits, debug=debug)

    # 2) 只有在 strict_api 或 問題帶 API/程式意圖時，才去掃 OAS
    need_code = detect_code_intent(question)
    need_api  = detect_api_intent(question)
    oas_info: Optional[Dict[str, Any]] = None

    if strict_api or need_api:
        api_specs = collect_api_specs(debug=debug)
        oas_info  = harvest_oas_whitelist(api_specs, hits, debug=debug)

    # 3) 控制上下文長度
    reserve = max_tokens + 800
    hits_ctx = clamp_context(hits, max_ctx_tokens=max_ctx_tokens, reserve=reserve)[:max_chunks]

    # 4) Prompt
    prompt = build_prompt(question, hits_ctx, need_code, oas_info if (strict_api or need_api) else None, strict_api)

    # 5) Call LLM（只在 400/context 超限時 fallback 一次）
    try:
        if llm == "ollama":
            raw = await call_ollama(prompt, temperature=temp, max_tokens=max_tokens)
        else:
            raw = await call_llama(prompt, temperature=temp, max_tokens=max_tokens)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400 and "context" in (e.response.text or "").lower():
            if debug: print("[DEBUG] llama 400; rebuild with smaller context and retry once …")
            hits_ctx = hits_ctx[: max(1, len(hits_ctx)//2) ]
            prompt = build_prompt(question, hits_ctx, need_code, oas_info if (strict_api or need_api) else None, strict_api)
            if llm == "ollama":
                raw = await call_ollama(prompt, temperature=temp, max_tokens=max_tokens)
            else:
                raw = await call_llama(prompt, temperature=temp, max_tokens=max_tokens)
        else:
            raise

    # 6) 後處理
    answer = sanitize_answer(raw)
    if strict_api and oas_info:
        answer = post_fix_strict_api(answer, oas_info)

    # 7) 引用
    cites = format_citations(hits_ctx, question)
    return f"{answer}\n\n=== 引用 ===\n\n{cites}"

# ----------------------
# Chat
# ----------------------
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

# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser(description="ODB MHW RAG CLI")
    ap.add_argument("question", nargs="?", default="", help="你的問題（--chat 時可省略）")
    ap.add_argument("--llm", choices=["ollama","llama"], default="llama")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-chunks", type=int, default=5, help="最多合併多少個 RAG 片段進 prompt")
    ap.add_argument("--ctx", type=int, default=3072, help="模型 context window 預算（用於裁切 RAG 片段）")
    ap.add_argument("--strict-api", action="store_true", help="嚴格依 OAS 的 params/append 白名單修正 params 區塊")
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



