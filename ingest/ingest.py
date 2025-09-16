#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "sentence-transformers is required for ingestion. Install it with 'pip install sentence-transformers'."
    ) from exc

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "qdrant-client is required for ingestion. Install it with 'pip install qdrant-client'."
    ) from exc

LOG_LEVEL = os.environ.get("INGEST_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("odbchat.ingest")

DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "thenlper/gte-small")
DEFAULT_EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

try:
    _EMBED_DIM: Optional[int] = int(os.environ.get("EMBED_DIM", ""))
    if _EMBED_DIM is not None and _EMBED_DIM <= 0:
        _EMBED_DIM = None
except ValueError:
    _EMBED_DIM = None

_embedder: Optional[SentenceTransformer] = None
_qdrant: Optional[QdrantClient] = None
_COLLECTION_STATE: Dict[str, Dict[str, bool]] = {}


@dataclass
class ChunkItem:
    collection: str
    doc_id: str
    chunk_id: int
    text: str
    payload: Dict[str, Any]


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.debug("Loading embedding model %s", DEFAULT_EMBED_MODEL)
        _embedder = SentenceTransformer(DEFAULT_EMBED_MODEL, device=DEFAULT_EMBED_DEVICE)
    return _embedder


def embedding_dim() -> int:
    global _EMBED_DIM
    if _EMBED_DIM is not None:
        return _EMBED_DIM
    model = get_embedder()
    try:
        _EMBED_DIM = int(model.get_sentence_embedding_dimension())
    except Exception:  # pragma: no cover - fallback for older libraries
        vec = model.encode(["dimension_probe"], normalize_embeddings=True)
        _EMBED_DIM = len(vec[0])
    return _EMBED_DIM


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    return _qdrant


def iter_yaml_docs(path: pathlib.Path) -> List[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return list(yaml.safe_load_all(handle))


def is_oas(doc: Any) -> bool:
    return isinstance(doc, dict) and ("openapi" in doc or "swagger" in doc)


def chunk_text(text: str, doc_type: Optional[str] = None) -> List[str]:
    if not isinstance(text, str):
        text = str(text)
    stripped = text.strip()
    if not stripped:
        return []
    doc_type_norm = (doc_type or "").lower()
    if doc_type_norm == "api_spec":
        return [stripped]
    if doc_type_norm in {
        "api_guide",
        "api_spec_guide",
        "tutorial",
        "cli_tool_guide",
        "code_snippet",
        "code_example",
    }:
        size, step = 900, 750
    else:
        size, step = 1200, 1000
    if len(stripped) <= size:
        return [stripped]
    out: List[str] = []
    start = 0
    while start < len(stripped):
        chunk = stripped[start : start + size].strip()
        if chunk:
            out.append(chunk)
        start += step
    if not out:
        out.append(stripped)
    return out


def normalize_front_matter(front: Dict[str, Any], path: pathlib.Path) -> Dict[str, Any]:
    meta = {k: v for k, v in (front or {}).items() if k != "content"}
    dataset = front.get("dataset") or front.get("dataset_name")
    if dataset:
        meta["dataset"] = dataset
    collection = meta.get("collection")
    if not collection:
        raise ValueError(f"{path} missing required front-matter key 'collection'")
    meta["collection"] = str(collection)
    meta.setdefault("doc_type", "note")
    meta.setdefault("title", path.stem)
    meta.setdefault("source_file", str(path))
    tags = meta.get("tags")
    if isinstance(tags, str):
        meta["tags"] = [tags]
    return meta


def build_payload(base_meta: Dict[str, Any], doc_meta: Dict[str, Any], path: pathlib.Path, doc_id: str) -> Dict[str, Any]:
    payload = dict(base_meta)
    if isinstance(doc_meta, dict):
        for key, value in doc_meta.items():
            if key == "content":
                continue
            if key == "tags" and isinstance(value, str):
                payload[key] = [value]
            else:
                payload[key] = value
    payload.setdefault("collection", base_meta.get("collection"))
    payload.setdefault("doc_type", base_meta.get("doc_type", "note"))
    payload.setdefault("title", base_meta.get("title") or path.stem)
    payload["source_file"] = str(path)
    payload["doc_id"] = doc_id
    tags = payload.get("tags")
    if isinstance(tags, str):
        payload["tags"] = [tags]
    return payload


def extract_content_text(doc: Any, doc_meta: Dict[str, Any]) -> str:
    if isinstance(doc_meta, dict):
        content = doc_meta.get("content")
        if isinstance(content, str) and content.strip():
            return content
    if isinstance(doc, str):
        return doc
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def should_treat_as_oas(base_doc_type: Optional[str], doc_meta: Dict[str, Any], doc: Any) -> bool:
    if isinstance(doc_meta, dict) and doc_meta.get("doc_type") == "api_spec":
        return True
    if (base_doc_type or "").lower() == "api_spec":
        return True
    return is_oas(doc)


def next_chunk_id(counter: Dict[str, int], doc_id: str) -> int:
    current = counter.get(doc_id, 0)
    counter[doc_id] = current + 1
    return current


def ensure_collection(name: str, mode: str, dry: bool) -> None:
    state = _COLLECTION_STATE.setdefault(name, {"ensured": False, "cleared": False})
    if mode == "overwrite" and not state["cleared"]:
        if dry:
            logger.info("[DRY] recreate collection %s", name)
            state["cleared"] = True
            state["ensured"] = True
            return
        client = get_qdrant()
        try:
            client.delete_collection(name)
        except Exception as exc:  # pragma: no cover
            logger.debug("delete_collection(%s) skipped: %s", name, exc)
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=embedding_dim(), distance=Distance.COSINE),
        )
        state["cleared"] = True
        state["ensured"] = True
        return
    if state["ensured"]:
        return
    if dry:
        logger.info("[DRY] ensure collection %s", name)
        state["ensured"] = True
        return
    client = get_qdrant()
    exists = False
    try:
        resp = client.get_collections()
        collections = getattr(resp, "collections", [])
        exists = any(getattr(col, "name", None) == name for col in collections)
    except Exception as exc:  # pragma: no cover
        logger.debug("get_collections failed: %s", exc)
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=embedding_dim(), distance=Distance.COSINE),
        )
    state["ensured"] = True


def has_existing_doc(collection: str, doc_id: str) -> bool:
    client = get_qdrant()
    try:
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
            limit=1,
            with_payload=False,
        )
        return bool(points)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to check for existing doc_id=%s in %s: %s", doc_id, collection, exc)
        return False


def upsert_collection_chunks(collection: str, items: List[ChunkItem], mode: str, dry: bool) -> None:
    if not items:
        return
    ensure_collection(collection, mode, dry)
    if dry:
        for item in items:
            print(
                json.dumps(
                    {
                        "action": "plan",
                        "mode": mode,
                        "collection": collection,
                        "doc_id": item.doc_id,
                        "chunk_id": item.chunk_id,
                        "doc_type": item.payload.get("doc_type"),
                        "title": item.payload.get("title"),
                        "text_length": len(item.text),
                    },
                    ensure_ascii=False,
                )
            )
        return
    if mode == "insert":
        for doc_id in {item.doc_id for item in items}:
            if has_existing_doc(collection, doc_id):
                raise RuntimeError(
                    f"Collection '{collection}' already contains doc_id '{doc_id}'. Use --mode upsert to replace."
                )
    model = get_embedder()
    vectors = model.encode([item.text for item in items], normalize_embeddings=True, show_progress_bar=False)
    client = get_qdrant()
    points = []
    for item, vector in zip(items, vectors):
        payload = dict(item.payload)
        payload["doc_id"] = item.doc_id
        payload["chunk_id"] = item.chunk_id
        payload["text"] = item.text
        vec_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec_list, payload=payload))
    client.upsert(collection_name=collection, points=points)
    logger.info("Upserted %d chunk(s) into %s", len(points), collection)


def process_file(path: pathlib.Path, mode: str, dry: bool) -> None:
    docs = iter_yaml_docs(path)
    if not docs:
        logger.warning("Skipping %s (no documents)", path)
        return
    front = docs[0] if isinstance(docs[0], dict) else {}
    base_meta = normalize_front_matter(front, path)
    base_doc_id = str(front.get("doc_id") or front.get("id") or path.stem)
    chunk_counts: Dict[str, int] = {}
    scheduled: Dict[str, List[ChunkItem]] = {}

    def enqueue(item: ChunkItem) -> None:
        scheduled.setdefault(item.collection, []).append(item)

    logger.info("Processing %s", path)

    if isinstance(front, dict):
        content = front.get("content")
        if isinstance(content, str) and content.strip():
            doc_id = str(front.get("doc_id") or base_doc_id)
            payload = build_payload(base_meta, {}, path, doc_id)
            for text in chunk_text(content, payload.get("doc_type")):
                chunk_id = next_chunk_id(chunk_counts, doc_id)
                enqueue(ChunkItem(payload["collection"], doc_id, chunk_id, text, dict(payload)))

    for idx, raw_doc in enumerate(docs[1:], start=1):
        doc_meta = raw_doc if isinstance(raw_doc, dict) else {}
        doc_id = str(doc_meta.get("doc_id") or doc_meta.get("id") or base_doc_id)
        payload = build_payload(base_meta, doc_meta, path, doc_id)
        target_collection = payload.get("collection")
        if not target_collection:
            raise ValueError(f"{path} doc[{idx}] missing 'collection'")
        is_api_spec = should_treat_as_oas(base_meta.get("doc_type"), doc_meta, raw_doc)
        if is_api_spec:
            text = yaml.safe_dump(raw_doc, sort_keys=False, allow_unicode=True)
            text = text.strip()
            if not text:
                continue
            payload_spec = dict(payload)
            payload_spec["doc_type"] = "api_spec"
            chunk_id = next_chunk_id(chunk_counts, doc_id)
            enqueue(ChunkItem(target_collection, doc_id, chunk_id, text, payload_spec))
            continue
        text = extract_content_text(raw_doc, doc_meta).strip()
        if not text:
            continue
        for piece in chunk_text(text, payload.get("doc_type")):
            chunk_id = next_chunk_id(chunk_counts, doc_id)
            enqueue(ChunkItem(target_collection, doc_id, chunk_id, piece, dict(payload)))

    if not scheduled:
        logger.warning("No chunks generated from %s", path)
        return

    for collection, items in scheduled.items():
        upsert_collection_chunks(collection, items, mode, dry)


def gather_files(root: Optional[pathlib.Path], single: Optional[pathlib.Path]) -> List[pathlib.Path]:
    if single:
        if not single.exists():
            raise FileNotFoundError(single)
        if single.is_dir():
            raise ValueError(f"--file expects a file, got directory: {single}")
        return [single]
    if not root:
        raise ValueError("Provide --root or --file")
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(path for path in root.rglob("*.yml") if path.is_file())


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest multi-doc YAML files into Qdrant")
    parser.add_argument("--root", type=pathlib.Path, help="Directory to scan for .yml files")
    parser.add_argument("--file", type=pathlib.Path, help="Ingest a single .yml file")
    parser.add_argument(
        "--mode",
        choices=["overwrite", "upsert", "insert", "dry-run"],
        default="upsert",
        help="Ingestion mode (overwrite clears the target collection before ingesting)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --mode dry-run to print the planned operations without writing",
    )
    args = parser.parse_args(argv)

    if args.file and args.root:
        parser.error("Use --root or --file, not both")

    mode = "dry-run" if args.dry_run else args.mode
    dry = mode == "dry-run"

    files = gather_files(args.root, args.file)
    if not files:
        logger.warning("No YAML files found for ingestion")
        return

    for path in files:
        try:
            process_file(path, mode, dry)
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", path, exc)
            if not dry:
                raise


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
