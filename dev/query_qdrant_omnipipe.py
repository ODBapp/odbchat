#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
query_qdrant_omnipipe.py
- 針對 Omnipipe collection 做檢索測試
- 支援 doc_type filter（text_chunk / table / api_endpoint / section）
"""

import os
import argparse
from typing import List, Dict, Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COL = os.environ.get("QDRANT_COL", "AI")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "thenlper/gte-small")

embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def encode_query(q: str) -> List[float]:
    return embedder.encode([q], normalize_embeddings=True)[0].tolist()


def query_by_doc_type(question: str, doc_type: str, topk: int = 5, collection: str = QDRANT_COL) -> List[Dict[str, Any]]:
    vec = encode_query(question)
    resp = qdrant.query_points(
        collection_name=collection,
        query=vec,
        limit=topk,
        with_payload=True,
        with_vectors=False,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchValue(value=doc_type),
                )
            ]
        ),
    )
    return [
        {
            "score": p.score,
            "artifact_id": p.payload.get("artifact_id", ""),
            "doc_type": p.payload.get("doc_type", ""),
            "source_file": p.payload.get("source_file", ""),
            "table_label": p.payload.get("table_label", ""),
            "caption": p.payload.get("caption", ""),
            "markdown_content": p.payload.get("markdown_content", ""),
            "text": (p.payload.get("text", "")[:240] + "...") if len(p.payload.get("text", "")) > 240 else p.payload.get("text", ""),
        }
        for p in resp.points
    ]


def query_all(question: str, topk: int = 8, collection: str = QDRANT_COL) -> List[Dict[str, Any]]:
    vec = encode_query(question)
    resp = qdrant.query_points(
        collection_name=collection,
        query=vec,
        limit=topk,
        with_payload=True,
        with_vectors=False,
        query_filter=None,
    )
    return [
        {
            "score": p.score,
            "artifact_id": p.payload.get("artifact_id", ""),
            "doc_type": p.payload.get("doc_type", ""),
            "source_file": p.payload.get("source_file", ""),
            "table_label": p.payload.get("table_label", ""),
            "caption": p.payload.get("caption", ""),
            "markdown_content": p.payload.get("markdown_content", ""),
            "text": (p.payload.get("text", "")[:240] + "...") if len(p.payload.get("text", "")) > 240 else p.payload.get("text", ""),
        }
        for p in resp.points
    ]


def main():
    parser = argparse.ArgumentParser(description="Query Omnipipe Qdrant collection.")
    parser.add_argument("--collection", default=QDRANT_COL)
    parser.add_argument("--query", default="WOA23 Table 4 variables")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--doc-type", default="", help="Filter by doc_type (text_chunk/table/api_endpoint/section)")
    args = parser.parse_args()

    collection = args.collection

    if args.doc_type:
        hits = query_by_doc_type(args.query, args.doc_type, topk=args.topk, collection=collection)
        print(f"[doc_type={args.doc_type}]")
    else:
        hits = query_all(args.query, topk=args.topk, collection=collection)
        print("[all doc_types]")

    for idx, hit in enumerate(hits, 1):
        score = hit["score"]
        score_str = f"{score:.4f}" if isinstance(score, (float, int)) else "-"
        print(f"{idx:02d}. score={score_str} [{hit['doc_type']}] {hit['artifact_id']} ({hit['source_file']})")
        text = hit.get("text") or ""
        if text:
            print(f"    {text}")
        if hit.get("caption"):
            print(f"    caption: {hit['caption']}")
        if hit.get("table_label"):
            print(f"    table_label: {hit['table_label']}")
        if hit.get("markdown_content"):
            md = hit["markdown_content"]
            preview = md[:240] + ("..." if len(md) > 240 else "")
            print(f"    markdown: {preview}")


if __name__ == "__main__":
    main()
