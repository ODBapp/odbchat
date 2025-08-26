#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, math, argparse, textwrap
from typing import List, Dict, Any
import httpx

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# ---------- defaults ----------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION   = os.getenv("QDRANT_COLLECTION", "odb_mhw_knowledge_v1")

EMBED_NAME   = os.getenv("EMBED_MODEL", "thenlper/gte-small")  # must match ingest
EMBED_DIM    = int(os.getenv("EMBED_DIM", "384"))

# llama.cpp server
LLAMA_URL    = os.getenv("LLAMA_URL", "http://localhost:8001/completion")
LLAMA_MODEL_HINT = os.getenv("LLAMA_MODEL", "")  # optional, for prompt header only

# ollama
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")


def parse_args():
    ap = argparse.ArgumentParser(description="Simple RAG CLI for ODB MHW.")
    ap.add_argument("question", nargs="+", help="Your question text")
    ap.add_argument("--llm", choices=["llama", "ollama"], default="llama",
                    help="Generator backend (default: llama)")
    ap.add_argument("--topk", type=int, default=12, help="initial qdrant hits")
    ap.add_argument("--mmr_k", type=int, default=4, help="final context chunks")
    ap.add_argument("--temp", type=float, default=0.2, help="generation temperature (ollama)")
    ap.add_argument("--max_tokens", type=int, default=512, help="max tokens (llama n_predict)")
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


# ------------ qdrant search + mmr ------------
def mmr(query_vec, doc_vecs, k=4, lambda_mult=0.6):
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

def search(qdrant: QdrantClient, query: str, topk=12) -> List[Dict[str, Any]]:
    qv = embed(query)
    flt = Filter(must=[
        FieldCondition(key="dataset",   match=MatchValue(value="marineheatwave")),
        FieldCondition(key="collection",match=MatchValue(value="mhw")),
    ])
    res = qdrant.search(
        collection_name=COLLECTION,
        query_vector=qv,
        limit=topk,
        query_filter=flt,
        with_payload=True,
        with_vectors=True
    )
    docs = [{
        "id": str(r.id),
        "text": r.payload.get("text",""),
        "meta": {k:v for k,v in r.payload.items() if k!="text"},
        "vec":  r.vector,
    } for r in res]
    if not docs:
        return []

    order = mmr(qv, [d["vec"] for d in docs], k=4, lambda_mult=0.6)
    return [docs[i] for i in order]


# ------------ prompt building ------------
def build_prompt(question: str, ctx: List[Dict[str, Any]]) -> str:
    # keep bilingual-friendly instructions, strictly grounded
    sys_rule = (
        "你是海洋熱浪(MHW)助理。你只能根據下方『依據』回答；"
        "若依據不足，請回覆：『無法在已知資料中找到答案』，並建議查閱 ODB MHW API/文件。"
        "回答可中英雙語，務必簡潔，且在文末標示引用如 [1][2]。"
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
        # llama.cpp completion endpoint
        resp = await client.post(LLAMA_URL, json={
            "prompt": prompt,
            "n_predict": max_tokens,
            # you can add 'temperature', 'top_p' if your server supports
        })
        resp.raise_for_status()
        data = resp.json()
        # recent llama.cpp returns {"content": "..."} or streaming chunks
        if isinstance(data, dict) and "content" in data:
            return data["content"].strip()
        # fallback if server returns plain string
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


# ------------ run once ------------
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
    if not question:
        print("Empty question"); sys.exit(1)

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    hits = search(qdrant, question, topk=args.topk)

    if not hits:
        print("無法在已知資料中找到答案（檢索為空）。請確認已 ingest MHW 知識。")
        sys.exit(0)

    prompt = build_prompt(question, hits[:args.mmr_k])

    if args.llm == "llama":
        answer = await call_llama(prompt, max_tokens=args.max_tokens)
    else:
        answer = await call_ollama(prompt, temperature=args.temp)

    print("\n=== 答覆 ===\n")
    print(textwrap.fill(answer, width=100))
    print("\n=== 引用 ===\n")
    print(render_citations(hits[:args.mmr_k]))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
