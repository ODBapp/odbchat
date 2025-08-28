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
from typing import Any, Dict, List, Tuple, Optional

# ---- 抑制舊 NVIDIA Driver 的 PyTorch CUDA 初始化警告 ----
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
)
from sentence_transformers import SentenceTransformer

# ----------------------
# Config (env-overridable)
# ----------------------
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION  = os.environ.get("QDRANT_COL", "odb_mhw_knowledge_v1")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "thenlper/gte-small")
EMBED_DIM   = 384

OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL= os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "120"))

LLAMA_URL   = os.environ.get("LLAMA_URL", "http://localhost:8001/completion")

# ----------------------
# Init
# ----------------------
qdrant  = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder= SentenceTransformer(EMBED_MODEL, device="cpu")

# ----------------------
# Helpers
# ----------------------
CODE_FENCE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.I)

def approx_tokens(s: str) -> int:
    return max(1, math.ceil(len(s) / 3))

def clamp_context(chunks: List[Dict[str, Any]], max_ctx_tokens: int, reserve: int = 512) -> List[Dict[str, Any]]:
    out, used = [], 0
    budget = max(256, max_ctx_tokens - reserve)
    for h in chunks:
        t = h.get("text","")
        tk = approx_tokens(t)
        if used + tk > budget:
            break
        out.append(h)
        used += tk
    return out

def pick_need_code(question: str) -> bool:
    qs = question.lower()
    return any(k in qs for k in ["python","程式","code","範例","example","sample code"])

def uniq_sources(hits: List[Dict[str, Any]]) -> List[Tuple[str,str]]:
    seen = set()
    outs = []
    for h in hits:
        m = h.get("payload", h)
        title = m.get("title") or m.get("source_file") or "Unknown"
        url   = m.get("canonical_url") or m.get("source_file") or "Unknown"
        key = (title, url)
        if key not in seen:
            seen.add(key)
            outs.append(key)
    return outs[:5]

def extract_code_blocks(texts: List[str]) -> List[str]:
    out = []
    for t in texts:
        for m in CODE_FENCE.finditer(t):
            block = m.group(1).strip()
            if block:
                out.append(block)
    return out

def build_query_filter(force_api: bool=False) -> Optional[Filter]:
    if not force_api:
        return None
    return Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="api_spec"))])

def encode_query(q: str) -> List[float]:
    return embedder.encode([q], normalize_embeddings=True)[0].tolist()

def query_qdrant(question: str, topk: int = 5, force_api: bool=False) -> List[Dict[str,Any]]:
    """
    qdrant-client 1.15.x:
      - 使用 client.query_points(query=<vector>, query_filter=<Filter>, limit=.., with_payload=..)
      - 回傳物件有 .points 屬性
    """
    vec = encode_query(question)
    flt = build_query_filter(force_api)
    resp = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=topk,
        with_payload=True,
        with_vectors=False,
        query_filter=flt,   # 注意：是 query_filter 不是 filter
    )
    points = getattr(resp, "points", resp)  # 某些版本直接回 list
    out = []
    for sp in points:
        payload = getattr(sp, "payload", {}) or {}
        out.append({"text": payload.get("text",""), "payload": payload})
    return out

def collect_api_specs() -> List[Dict[str,Any]]:
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
        out.append({"text": payload.get("text",""), "payload": payload})
    return out

def parse_oas_params(oas_yaml: str) -> Dict[str, Any]:
    endpoints = {}
    # 抓 /api/mhw 與 /api/mhw/csv 區塊
    path_blocks = re.split(r'(?m)^\s*/api/(mhw(?:/csv)?)\s*:\s*$', oas_yaml)
    if len(path_blocks) >= 2:
        for i in range(1, len(path_blocks), 2):
            name = "/api/" + path_blocks[i]
            body = path_blocks[i+1]
            endpoints[name] = body

    params_whitelist: List[str] = []
    append_allowed: List[str] = []
    for _, body in endpoints.items():
        for m in re.finditer(r'(?m)^\s*-\s*name:\s*([A-Za-z0-9_]+)\s*$', body):
            p = m.group(1)
            if p not in params_whitelist:
                params_whitelist.append(p)
        if "name: append" in body:
            m2 = re.search(
                r"Allowed fields:\s*'([^']+)'(?:,\s*'([^']+)')?(?:,\s*'([^']+)')?(?:,\s*'([^']+)')?",
                body
            )
            if m2:
                for g in m2.groups():
                    if g and g not in append_allowed:
                        append_allowed.append(g)
    return {"params": params_whitelist, "append_allowed": append_allowed, "paths": list(endpoints.keys())}

def harvest_oas_whitelist(api_specs: List[Dict[str,Any]]) -> Dict[str, Any]:
    all_params, all_append, all_paths = set(), set(), set()
    for spec in api_specs:
        txt = spec.get("text","")
        meta = parse_oas_params(txt)
        all_params.update(meta["params"])
        all_append.update(meta["append_allowed"])
        all_paths.update(meta["paths"])
    return {
        "params": sorted(all_params),
        "append_allowed": sorted(all_append),
        "paths": sorted(all_paths),
    }

def extract_code_from_hits(hits: List[Dict[str,Any]]) -> List[str]:
    texts = [h.get("text","") for h in hits]
    return extract_code_blocks(texts)

def build_prompt(question: str, ctx: List[Dict[str, Any]], need_code: bool, oas_info: Optional[Dict[str,Any]], strict_api: bool) -> str:
    sys_rule = (
        "你是海洋學門資料庫(ODB)專業助理，針對海洋熱浪(MHW)與 ODB MHW API 提供準確回覆。"
        "首先以『依據』中的內容為主作答；若沒有依據，請回答：『無法在已知資料中找到答案』。"
        "若問題涉及 API 參數或調用，務必根據 OpenAPI 規格的參數名稱與說明回答，不能虛構參數或值。"
    )
    if need_code:
        sys_rule += "如需提供程式，請以 ```python 換行...``` 包覆，並依 OpenAPI 規格產生正確的 query 參數。"
    if strict_api and oas_info:
        sys_rule += (
            f"【API參數白名單】只能使用：{', '.join(oas_info.get('params', []))}；"
            f"append允許值：{', '.join(oas_info.get('append_allowed', []))}。"
        )

    ctx_text = ""
    for i, h in enumerate(ctx, 1):
        m = h.get("payload", {})
        title = m.get("title") or m.get("source_file") or "Unknown"
        ctx_text += f"\n[來源 {i}] {title}\n{h.get('text','')}\n"

    tmpl = (
        f"{sys_rule}\n\n"
        f"問題：{question}\n\n"
        f"依據：{ctx_text}\n\n"
        f"請作答："
    )
    return tmpl

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
            return data.get("response","").strip()
    except httpx.ReadTimeout:
        return "[ERROR] Ollama ReadTimeout，請確認伺服器狀態或提高 OLLAMA_TIMEOUT。"

async def call_llama(prompt: str, temperature: float=0.2, max_tokens: int=512) -> str:
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        resp = await client.post(
            LLAMA_URL,
            json={
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
            }
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "content" in data:
            return data["content"]
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("text","")
        return str(data)

def post_fix_strict_api(text: str, oas_info: Dict[str,Any], need_code: bool, debug: bool=False) -> str:
    allowed = set(oas_info.get("params", []))
    allowed_append = set(oas_info.get("append_allowed", []))
    m = re.search(r"params\s*=\s*\{([\s\S]*?)\}", text)
    if not m:
        return text

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
                    new_val = ",".join(filtered)
                    line = f'    "append": "{new_val}",'
        new_lines.append(line)

    body_fixed = "\n".join(new_lines)
    return text[:m.start(1)] + body_fixed + text[m.end(1):]

def format_citations(hits: List[Dict[str,Any]]) -> str:
    cites = []
    for i, (title, url) in enumerate(uniq_sources(hits), 1):
        cites.append(f"[{i}] {title} — {url}")
    return "\n".join(cites) if cites else "（無）"

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
    need_code = pick_need_code(question)

    # 1) 檢索
    hits = query_qdrant(question, topk=topk, force_api=False)

    # 2) OAS 白名單
    api_specs = collect_api_specs()
    oas_info  = harvest_oas_whitelist(api_specs)
    if debug:
        code_hits = [h for h in hits if (h.get('payload',{}).get('doc_type') in ('code_snippet','code_example','cli_tool_guide'))]
        print(f"[DEBUG] code docs scanned: {len(code_hits)}, code blocks collected: {len(extract_code_from_hits(hits))}")
        print(f"[DEBUG] Scanned api_spec docs: {len(api_specs)}")
        if oas_info.get("paths"):
            print(f"[DEBUG] OAS paths found: {oas_info['paths']}")
        print(f"[DEBUG] Retrieved hits: {len(hits)}")
        print(f"[DEBUG] strict-api: {strict_api}")
        print(f"[DEBUG] OAS params: {oas_info.get('params', [])}")
        print(f"[DEBUG] OAS append allowed: {oas_info.get('append_allowed', [])}")

    # 3) 裁切 context，避免超 context
    reserve = max_tokens + 800
    hits_ctx = clamp_context(hits, max_ctx_tokens=max_ctx_tokens, reserve=reserve)[:max_chunks]

    # 4) 組 prompt
    prompt = build_prompt(question, hits_ctx, need_code, oas_info if strict_api else None, strict_api)

    # 5) 呼叫 LLM
    if llm == "ollama":
        answer = await call_ollama(prompt, temperature=temp, max_tokens=max_tokens)
    else:
        try:
            answer = await call_llama(prompt, temperature=temp, max_tokens=max_tokens)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400 and "context" in e.response.text.lower():
                if debug: print("[DEBUG] 400 context-too-long; retry with fewer chunks")
                hits_ctx = hits_ctx[: max(1, len(hits_ctx)//2) ]
                prompt = build_prompt(question, hits_ctx, need_code, oas_info if strict_api else None, strict_api)
                answer = await call_llama(prompt, temperature=temp, max_tokens=max_tokens)
            else:
                raise

    # 6) 嚴格 API 後處理
    if strict_api and isinstance(oas_info, dict) and oas_info.get("params"):
        answer = post_fix_strict_api(answer, oas_info, need_code, debug)

    # 7) 引用
    cites = format_citations(hits_ctx)
    final = f"{answer}\n\n=== 引用 ===\n\n{cites}"
    return final

async def chat_loop(llm: str, topk: int, temp: float, max_tokens: int, strict_api: bool, debug: bool, max_ctx_tokens: int, max_chunks: int):
    print("Enter '/exit' to quit. Ask your MHW questions.\n")
    history: List[Tuple[str,str]] = []
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q in ("/exit","/quit"):
            break
        if not q:
            continue
        ans = await answer_once(q, llm, topk, temp, max_tokens, strict_api, debug, history, max_ctx_tokens, max_chunks)
        print("\n=== 答覆 ===\n")
        print(ans)
        history.append((q, ans))
        print()

def main():
    ap = argparse.ArgumentParser(description="ODB MHW RAG CLI")
    ap.add_argument("question", nargs="?", default="", help="your question (omit when --chat)")
    ap.add_argument("--llm", choices=["ollama","llama"], default="llama")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=512, help="max tokens for the answer")
    ap.add_argument("--max-chunks", type=int, default=5, help="max RAG chunks merged into the prompt")
    ap.add_argument("--ctx", type=int, default=3072, help="approx llama context window budget (for truncation)")
    ap.add_argument("--strict-api", action="store_true", help="enforce OpenAPI whitelist in params block")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--chat", action="store_true", help="interactive mode")
    args = ap.parse_args()

    if args.chat:
        if args.debug:
            print("[DEBUG] chat mode on")
        asyncio.run(chat_loop(args.llm, args.topk, args.temp, args.max_tokens, args.strict_api, args.debug, args.ctx, args.max_chunks))
        return

    if not args.question:
        print("請輸入問題或改用 --chat。")
        return

    out = asyncio.run(
        answer_once(
            args.question, args.llm, args.topk, args.temp,
            args.max_tokens, args.strict_api, args.debug,
            history=None, max_ctx_tokens=args.ctx, max_chunks=args.max_chunks
        )
    )
    print("\n=== 答覆 ===\n")
    print(out)

if __name__ == "__main__":
    main()



