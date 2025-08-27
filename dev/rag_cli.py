#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, argparse, textwrap, warnings
from typing import List, Dict, Any
import httpx

# ---- silence CUDA warnings from torch ----
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# ---------- defaults ----------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION   = os.getenv("QDRANT_COLLECTION", "odb_mhw_knowledge_v1")

EMBED_NAME   = os.getenv("EMBED_MODEL", "thenlper/gte-small")
EMBED_DIM    = int(os.getenv("EMBED_DIM", "384"))

# llama.cpp
LLAMA_URL    = os.getenv("LLAMA_URL", "http://localhost:8001/completion")

# ollama
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b-instruct")


def parse_args():
    ap = argparse.ArgumentParser(description="Simple RAG CLI for ODB MHW.")
    ap.add_argument("question", nargs="+", help="Your question text")
    ap.add_argument("--llm", choices=["llama", "ollama"], default="llama")
    ap.add_argument("--topk", type=int, default=12, help="initial qdrant hits")
    ap.add_argument("--mmr_k", type=int, default=4, help="final context chunks")
    ap.add_argument("--temp", type=float, default=0.2, help="temperature (ollama)")
    ap.add_argument("--max_tokens", type=int, default=512, help="max tokens (llama)")
    return ap.parse_args()


# ------------ embedding ------------
_embedder = None
def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_NAME)
    return _embedder

def embed(text: str) -> List[float]:
    return embedder().encode([text], normalize_embeddings=True)[0].tolist()


# ------------ qdrant search (version-compatible) ------------
def _qdrant_query_points(client: QdrantClient, *, query_vec, topk: int, flt: Filter):
    """
    qdrant-client 1.15+ uses `query_filter=`, older builds use `filter=`.
    Try new first, then fallback. Both return an object with `.points`.
    """
    # Preferred for 1.15+
    try:
        return client.query_points(
            collection_name=COLLECTION,
            query=query_vec,
            limit=topk,
            query_filter=flt,
            with_payload=True,
            with_vectors=True,
        )
    except AssertionError:
        # Older kw name
        return client.query_points(
            collection_name=COLLECTION,
            query=query_vec,
            limit=topk,
            filter=flt,
            with_payload=True,
            with_vectors=True,
        )
    except TypeError:
        # Some builds raise TypeError instead
        return client.query_points(
            collection_name=COLLECTION,
            query=query_vec,
            limit=topk,
            filter=flt,
            with_payload=True,
            with_vectors=True,
        )

def search(qdrant: QdrantClient, query: str, topk=12, force_api=False) -> List[Dict[str, Any]]:
    # Light query augmentation to align with your new purpose/faq fields
    aug = ""
    if force_api:
        aug = " API usage OpenAPI Swagger endpoint parameters example tutorial faq purpose "
    qv = embed(query + aug)

    must = [
        FieldCondition(key="dataset",   match=MatchValue(value="marineheatwave")),
        FieldCondition(key="collection",match=MatchValue(value="mhw")),
    ]

    # Soft-boost API-related doc_types via 'should' (doesn't exclude others)
    should = []
    if force_api:
        for dt in ("api_spec", "api_guide", "code_snippet", "cli_tool_guide", "tutorial"):
            should.append(FieldCondition(key="doc_type", match=MatchValue(value=dt)))

    flt = Filter(must=must, should=should or None)

    res = _qdrant_query_points(qdrant, query_vec=qv, topk=topk, flt=flt)
    points = getattr(res, "points", []) or []

    docs = []
    for r in points:
        payload = getattr(r, "payload", {}) or {}
        vec = getattr(r, "vector", None)
        docs.append({
            "id": str(getattr(r, "id", "")),
            "text": payload.get("text", ""),
            "meta": {k: v for k, v in payload.items() if k != "text"},
            "vec":  vec,
        })

    # If nothing after boosting (rare), return as-is
    if not docs:
        return docs

    # Optional: strict post-filter if boost still pulls wrong stuff (disabled by default)
    # if force_api:
    #     prefer = {"api_spec", "api_guide", "code_snippet", "cli_tool_guide", "tutorial"}
    #     api_docs = [d for d in docs if d["meta"].get("doc_type") in prefer]
    #     if api_docs:
    #         docs = api_docs

    return docs


# ------------ MMR re-ranking ------------
def mmr(query_vec, doc_vecs, k=4, lambda_mult=0.6):
    if not doc_vecs or any(v is None for v in doc_vecs):
        return list(range(min(k, len(doc_vecs))))
    selected, idxs = [], list(range(len(doc_vecs)))
    sim_q = [sum(a*b for a,b in zip(query_vec, v)) for v in doc_vecs]
    for _ in range(min(k, len(doc_vecs))):
        if not selected:
            j = max(idxs, key=lambda i: sim_q[i])
            selected.append(j); idxs.remove(j); continue
        def score(i):
            max_sim = max(sum(a*b for a,b in zip(doc_vecs[i], doc_vecs[s])) for s in selected)
            return lambda_mult*sim_q[i] - (1-lambda_mult)*max_sim
        j = max(idxs, key=score)
        selected.append(j); idxs.remove(j)
    return selected


# ------------ prompt ------------
def build_prompt(question: str, ctx: List[Dict[str, Any]]) -> str:
    sys_rule = (
        "你是海洋學門資料庫(Ocean Data Bank, ODB)專業助理，針對海洋資料或相關科學知識，如海洋熱浪(MHW)提供專業回覆。主要依據下方『依據』回答；"
        "若找不到依據，請回覆：『無法在已知資料中找到答案』，並建議查閱 ODB MHW API/文件。"
        "若依據內容不明確，可憑藉你的科學或程式專業，簡潔答覆，但務必在科學上準確，不確定即回回覆：『無法在已知資料中找到答案』，並建議查閱 ODB MHW API/文件。"
        "若問題是要求提供程式，檢索中的程式範例以'''python ... '''包覆，提供相關範例。ODB MHW API在檢索中可查到OpenAPI specification（OAS)，必須嚴格根據此規格上面的參數定義，以你的程式專業，提供回覆。程式回覆也必須以'''python ... '''包覆，並加上分行符號，讓輸出呈現可閱讀的code block"
        "如同時檢索到不同定義 (monthly vs daily)，需明確指出差異並標註來源，不可混淆。"
        "回答可中英雙語，務必簡潔，並在文末標示引用如 [1][2]。"
    )
    blocks = []
    for i, s in enumerate(ctx, 1):
        m = s["meta"]
        title = m.get("title") or m.get("doc_type", "snippet")
        src = m.get("canonical_url") or m.get("source_file","")
        blocks.append(f"[{i}] {title} | {src}\n{s['text']}")
    context = "\n\n".join(blocks)
    return f"{sys_rule}\n\n=== 依據 ===\n{context}\n\n=== 問題 ===\n{question}\n\n=== 回答 ==="


# ------------ LLM backends ------------
async def call_llama(prompt: str, max_tokens=512) -> str:
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(LLAMA_URL, json={"prompt": prompt, "n_predict": max_tokens})
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "content" in data:
            return data["content"].strip()
        if isinstance(data, str):
            return data.strip()
        return json.dumps(data)

async def call_ollama(prompt: str, temperature=0.2) -> str:
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "top_p": 0.9}
        })
        resp.raise_for_status()
        data = resp.json()
        return data.get("response","").strip()


# ------------ CLI main ------------
def render_citations(hits: List[Dict[str,Any]]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        title  = m.get("title") or m.get("doc_type","snippet")
        issuer = m.get("issuer","")
        url    = m.get("canonical_url") or m.get("source_file","")
        lic    = m.get("license","")
        tag    = f"[{i}] {title}"
        detail = " — " + issuer if issuer else ""
        detail += f", {url}" if url else ""
        detail += f" ({lic})" if lic else ""
        lines.append(tag + detail)
    return "\n".join(lines)

async def main():
    args = parse_args()
    question = " ".join(args.question).strip()
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    q_lower = question.lower()
    force_api = any(k in q_lower for k in ["api","openapi","swagger","使用","example","程式","endpoint","參數"])

    hits = search(qdrant, question, topk=args.topk, force_api=force_api)
    if not hits:
        print("無法在已知資料中找到答案（檢索為空）。")
        sys.exit(0)

    qv = embed(question)
    order = mmr(qv, [h["vec"] for h in hits], k=args.mmr_k)
    ctx = [hits[i] for i in order]

    prompt = build_prompt(question, ctx)

    if args.llm == "llama":
        answer = await call_llama(prompt, max_tokens=args.max_tokens)
    else:
        answer = await call_ollama(prompt, temperature=args.temp)

    print("\n=== 答覆 ===\n")
    print(textwrap.fill(answer, width=100))
    print("\n=== 引用 ===\n")
    print(render_citations(ctx))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
