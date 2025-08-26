#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import glob
import uuid
import time
import yaml
import hashlib
import argparse
from typing import List, Dict, Any, Tuple

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from sentence_transformers import SentenceTransformer

# ----------------------------
# Defaults (override via CLI)
# ----------------------------
DEFAULT_ROOT = "rag"
DEFAULT_COLLECTION = "odb_mhw_knowledge_v1"
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_EMBED_MODEL = "thenlper/gte-small"   # 384-dim, multilingual
DEFAULT_EMBED_DIM = 384

def parse_args():
    ap = argparse.ArgumentParser(
        description="Ingest MHW-only YAML docs (front-matter + content: |) into Qdrant."
    )
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Root folder (e.g., rag_doc)")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name")
    ap.add_argument("--qdrant-host", default=DEFAULT_QDRANT_HOST)
    ap.add_argument("--qdrant-port", type=int, default=DEFAULT_QDRANT_PORT)
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    ap.add_argument("--dry-run", action="store_true", help="Parse only; no upsert")
    return ap.parse_args()


# ----------------------------
# YAML loader (front-matter + content: |)
# ----------------------------
def load_frontmatter_and_content(yaml_path: str) -> Tuple[Dict[str, Any], str]:
    """
    Expect a single YAML document whose top-level keys include our metadata and
    a 'content' scalar block. We don't require the '---' markers since the file
    itself is YAML.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{yaml_path} is not a valid YAML mapping")

    content = data.get("content", "")
    if content is None:
        content = ""
    meta = dict(data)
    meta.pop("content", None)

    return meta, str(content)


def is_top_level_manifest(meta: Dict[str, Any]) -> bool:
    # Treat as "manifest (top-level)" if doc_type explicitly says so,
    # or if it has 'components' and no content.
    if meta.get("doc_type", "").lower().startswith("manifest"):
        return True
    if "components" in meta:
        return True
    return False


# ----------------------------
# Utilities
# ----------------------------
def stable_doc_id(meta: Dict[str, Any]) -> str:
    """
    Ensure a stable doc_id. If provided keep it; otherwise derive from canonical_url
    or source_file path as a fallback.
    """
    if "doc_id" in meta and meta["doc_id"]:
        return str(meta["doc_id"])

    basis = meta.get("canonical_url") or meta.get("source_file") or repr(sorted(meta.items()))
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def normalize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provide sensible defaults and ensure required attrs exist.
    """
    out = dict(meta)

    # Required “namespace” keys for filtering
    out.setdefault("dataset", meta.get("dataset_name", "marineheatwave"))
    out.setdefault("collection", "mhw")

    # Type defaults
    out.setdefault("doc_type", "note")
    out.setdefault("lang", "zh")

    # Citation-related defaults
    out.setdefault("issuer", "Unknown")
    out.setdefault("canonical_url", "")
    out.setdefault("license", "Unknown")
    out.setdefault("retrieved_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # Source bookkeeping
    out.setdefault("source_type", "internal")
    out.setdefault("source_file", "")

    # Title fallback
    if not out.get("title"):
        # try from path
        fn = os.path.basename(out.get("source_file") or "untitled")
        out["title"] = os.path.splitext(fn)[0]

    # doc_id
    out["doc_id"] = stable_doc_id(out)

    # clean arrays
    for k in ("tags", "related_doc", "depends_on"):
        if k in out and isinstance(out[k], list):
            out[k] = [x for x in out[k] if str(x).strip()]

    return out


def chunk_text(text: str, doc_type: str) -> List[str]:
    """
    Simple char-based chunking policy by doc_type.
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    if doc_type in {"api_spec", "api_guide", "code_snippet", "cli_tool_guide", "tutorial"}:
        size, step = 450, 350
    else:
        size, step = 900, 750

    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + size].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def ensure_collection(client: QdrantClient, collection: str, dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def embed_many(embedder: SentenceTransformer, chunks: List[str]) -> List[List[float]]:
    if not chunks:
        return []
    # normalize_embeddings=True gives cosine-friendly unit vectors
    return embedder.encode(chunks, normalize_embeddings=True).tolist()


def ingest_file(
    path: str,
    client: QdrantClient,
    collection: str,
    embedder: SentenceTransformer,
    dry_run: bool = False,
) -> int:
    meta, content = load_frontmatter_and_content(path)

    # Skip top-level manifests and files without content
    if is_top_level_manifest(meta) or not content.strip():
        return 0

    # Fill defaults, ensure doc_id, etc.
    meta.setdefault("source_file", os.path.relpath(path, start=os.getcwd()))
    meta = normalize_meta(meta)

    # MHW-only guard (optional but recommended)
    if meta.get("dataset") != "marineheatwave" or meta.get("collection") != "mhw":
        # keep strict for this phase
        return 0

    chunks = chunk_text(content, meta.get("doc_type", "note"))
    if not chunks:
        return 0

    vectors = embed_many(embedder, chunks)
    points = []
    for i, (ch, vec) in enumerate(zip(chunks, vectors)):
        payload = dict(meta)
        payload.update(
            {
                "text": ch,
                "chunk_id": i,
            }
        )
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))

    if dry_run:
        print(f"[DRY] {os.path.basename(path)} -> {len(points)} chunks")
        return len(points)

    client.upsert(collection_name=collection, points=points)
    return len(points)


def main():
    args = parse_args()
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    ensure_collection(client, args.collection, args.embed_dim)
    embedder = SentenceTransformer(args.embed_model)

    # gather YAMLs
    pattern = os.path.join(args.root, "**", "*.yml")
    files = glob.glob(pattern, recursive=True)
    if not files:
        print(f"[WARN] No .yml files under {args.root}")
        return

    total = 0
    for p in tqdm(sorted(files), desc="Ingest MHW"):
        try:
            total += ingest_file(p, client, args.collection, embedder, args.dry_run)
        except Exception as e:
            print(f"[ERROR] {p}: {e}")

    print(f"[DONE] Upserted chunks: {total}")


if __name__ == "__main__":
    main()
