#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
    raise RuntimeError("qdrant-client is required for ingestion. Install it with 'pip install qdrant-client'.") from exc

LOG_LEVEL = os.environ.get("INGEST_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("odbchat.ingest_json")

DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "thenlper/gte-small")
DEFAULT_EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")
DEFAULT_COLLECTION_PREFIX = os.environ.get("OMNIPIPE_COLLECTION_PREFIX", "")

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

_embedder: Optional[SentenceTransformer] = None
_embed_dim: Optional[int] = None
_qdrant: Optional[QdrantClient] = None
_COLLECTION_STATE: Dict[str, Dict[str, bool]] = {}


@dataclass
class ArtifactItem:
    collection: str
    artifact_id: str
    chunk_id: int
    text: str
    payload: Dict[str, Any]


def get_embedder(model_name: str, device: str) -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.debug("Loading embedding model %s", model_name)
        _embedder = SentenceTransformer(model_name, device=device)
    return _embedder


def embedding_dim(model_name: str, device: str) -> int:
    global _embed_dim
    if _embed_dim is not None:
        return _embed_dim
    model = get_embedder(model_name, device)
    try:
        _embed_dim = int(model.get_sentence_embedding_dimension())
    except Exception:  # pragma: no cover
        vec = model.encode(["dimension_probe"], normalize_embeddings=True)
        _embed_dim = len(vec[0])
    return _embed_dim


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    return _qdrant


def _coalesce_text(values: Sequence[Optional[str]], sep: str = " ") -> str:
    parts = [str(v).strip() for v in values if isinstance(v, str) and v.strip()]
    return sep.join(parts).strip()

def _extract_first_url(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"https?://\S+", str(text))
    if not match:
        return ""
    return match.group(0).strip().rstrip(").,")


def _metadata_text_from_image(metadata: Dict[str, Any]) -> str:
    if not metadata:
        return ""
    parts: List[str] = []
    extracted = metadata.get("extracted_items")
    if isinstance(extracted, list):
        for item in extracted:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity") or "").strip()
            details = str(item.get("details") or "").strip()
            if entity and details:
                parts.append(f"{entity}: {details}")
            elif entity:
                parts.append(entity)
            elif details:
                parts.append(details)
    raw_text = metadata.get("raw_text")
    if isinstance(raw_text, str) and raw_text.strip():
        parts.append(raw_text.strip())
    return "\n".join(parts).strip()


def text_for_artifact(artifact: Dict[str, Any]) -> str:
    artifact_type = str(artifact.get("artifact_type") or "")
    if artifact_type == "text_chunk":
        base = str(artifact.get("text") or "").strip()
        metadata = artifact.get("metadata")
        if isinstance(metadata, dict):
            image_text = _metadata_text_from_image(metadata)
            if image_text:
                return _coalesce_text([base, image_text], sep="\n").strip()
        return base
    if artifact_type == "table":
        caption = str(artifact.get("caption") or "").strip()
        markdown = str(artifact.get("markdown_content") or "").strip()
        return _coalesce_text([caption, markdown], sep="\n").strip()
    if artifact_type == "api_endpoint":
        method = artifact.get("method")
        path = artifact.get("path")
        summary = artifact.get("summary")
        description = artifact.get("description")
        head = _coalesce_text([str(method).upper() if method else None, path], sep=" ")
        tail = _coalesce_text([summary], sep=" ")
        body = _coalesce_text([description], sep=" ")
        line1 = _coalesce_text([head, f"-- {tail}" if tail else None], sep=" ")
        return _coalesce_text([line1, body], sep="\n").strip()
    if artifact_type == "section":
        return _coalesce_text([artifact.get("section_number"), artifact.get("title")], sep=" ").strip()
    return ""


def _normalize_tags(tags: Any) -> List[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tags]
    if isinstance(tags, Iterable):
        return [str(tag) for tag in tags if tag is not None]
    return [str(tags)]

def _infer_dataset_name(source_file: str) -> Optional[str]:
    if not source_file:
        return None
    stem = pathlib.Path(source_file).stem
    return stem or None

def _merge_tags(*tag_lists: Iterable[Any]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for tags in tag_lists:
        for tag in _normalize_tags(tags):
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(tag)
    return merged


def _payload_from_artifact(artifact: Dict[str, Any], collection: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    artifact_type = str(artifact.get("artifact_type") or "")
    payload["artifact_id"] = str(artifact.get("id") or "")
    payload["artifact_type"] = artifact_type
    payload["doc_type"] = artifact_type
    payload["source_file"] = str(artifact.get("source_file") or "")
    payload["links"] = artifact.get("links") or []
    payload["tags"] = _normalize_tags(artifact.get("tags"))
    payload["collection"] = collection
    payload["doc_id"] = payload["artifact_id"]

    metadata = artifact.get("metadata")
    if isinstance(metadata, dict):
        meta_doc_type = metadata.get("doc_type") or metadata.get("type")
        if meta_doc_type and artifact_type == "text_chunk":
            payload["doc_type"] = str(meta_doc_type)
        for key in ("dataset_name", "dataset", "title", "lang", "issuer", "source_type"):
            if key in metadata and metadata.get(key) is not None:
                payload[key] = metadata.get(key)
        meta_tags = metadata.get("tags")
        if meta_tags:
            payload["tags"] = _merge_tags(payload.get("tags"), meta_tags)
        if not payload.get("canonical_url"):
            for key in ("canonical_url", "source_url", "url", "source_file"):
                value = metadata.get(key)
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    payload["canonical_url"] = value
                    break
        if not payload.get("canonical_url"):
            raw_text = metadata.get("raw_text")
            if isinstance(raw_text, str):
                url = _extract_first_url(raw_text)
                if url:
                    payload["canonical_url"] = url

    for key in (
        "page_number",
        "chunk_index",
        "section_number",
        "title",
        "method",
        "path",
        "summary",
        "description",
        "caption",
        "table_label",
        "markdown_content",
        "parameters",
        "usage_example",
        "metadata",
    ):
        if key in artifact and artifact.get(key) is not None:
            payload[key] = artifact.get(key)

    if not payload.get("dataset_name"):
        inferred = _infer_dataset_name(payload.get("source_file") or "")
        if inferred:
            payload["dataset_name"] = inferred

    core_fields = (
        "doc_type",
        "dataset_name",
        "title",
        "tags",
        "table_label",
        "caption",
        "path",
        "method",
    )
    payload_core: Dict[str, Any] = {}
    for key in core_fields:
        value = payload.get(key)
        if value is None or value == "":
            continue
        payload_core[key] = value
    payload["payload_core"] = payload_core

    return payload


def _artifact_items_for_file(data: Dict[str, Any], collection: str) -> List[ArtifactItem]:
    artifacts = data.get("artifacts") if isinstance(data, dict) else None
    if artifacts is None:
        raise ValueError("JSON must contain an 'artifacts' list")
    if not isinstance(artifacts, list):
        raise ValueError("'artifacts' must be a list")

    items: List[ArtifactItem] = []
    missing = Counter()
    table_missing = Counter()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        artifact_id = str(artifact.get("id") or "").strip()
        if not artifact_id:
            continue
        text = text_for_artifact(artifact)
        if not text:
            continue
        payload = _payload_from_artifact(artifact, collection)
        chunk_id = artifact.get("chunk_index")
        try:
            chunk_id = int(chunk_id) if chunk_id is not None else 0
        except Exception:
            chunk_id = 0
        payload["chunk_id"] = chunk_id
        payload["doc_id"] = artifact_id
        if not payload.get("dataset_name"):
            missing["dataset_name"] += 1
        if not payload.get("doc_type"):
            missing["doc_type"] += 1
        if not payload.get("title"):
            missing["title"] += 1
        if not payload.get("tags"):
            missing["tags"] += 1
        if payload.get("doc_type") == "table":
            if not payload.get("caption"):
                table_missing["caption"] += 1
            if not payload.get("markdown_content"):
                table_missing["markdown_content"] += 1
            if not payload.get("table_label"):
                table_missing["table_label"] += 1
        items.append(ArtifactItem(collection, artifact_id, chunk_id, text, payload))
    if missing:
        logger.warning("Missing metadata in %s: %s", data.get("filename", "JSON"), dict(missing))
    if table_missing:
        logger.warning("Missing table fields in %s: %s", data.get("filename", "JSON"), dict(table_missing))
    return items


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"artifacts": data}
    raise ValueError("JSON must be an object or a list of artifacts")


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
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def ensure_collection(name: str, mode: str, model_name: str, device: str, dry: bool) -> None:
    if mode == "overwrite":
        state = _COLLECTION_STATE.setdefault(name, {"cleared": False, "ensured": False})
        if state["cleared"]:
            return
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
            vectors_config=VectorParams(size=embedding_dim(model_name, device), distance=Distance.COSINE),
        )
        state["cleared"] = True
        state["ensured"] = True
        return

    state = _COLLECTION_STATE.setdefault(name, {"cleared": False, "ensured": False})
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
            vectors_config=VectorParams(size=embedding_dim(model_name, device), distance=Distance.COSINE),
        )
    state["ensured"] = True


def _artifact_exists(collection: str, artifact_id: str) -> bool:
    client = get_qdrant()
    try:
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[FieldCondition(key="artifact_id", match=MatchValue(value=artifact_id))]),
            limit=1,
            with_payload=False,
        )
        return bool(points)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to check for existing artifact_id=%s in %s: %s", artifact_id, collection, exc)
        return False


def _delete_artifact(collection: str, artifact_id: str) -> None:
    client = get_qdrant()
    try:
        client.delete(
            collection_name=collection,
            points_selector=Filter(must=[FieldCondition(key="artifact_id", match=MatchValue(value=artifact_id))]),
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to delete artifact_id=%s in %s: %s", artifact_id, collection, exc)


def upsert_items(items: List[ArtifactItem], mode: str, model_name: str, device: str, dry: bool) -> None:
    if not items:
        return
    collection = items[0].collection
    ensure_collection(collection, mode, model_name, device, dry)

    if dry:
        for item in items:
            print(
                json.dumps(
                    {
                        "action": "plan",
                        "mode": mode,
                        "collection": item.collection,
                        "artifact_id": item.artifact_id,
                        "doc_type": item.payload.get("doc_type"),
                        "text_length": len(item.text),
                        "links": len(item.payload.get("links") or []),
                    },
                    ensure_ascii=False,
                )
            )
        return

    if mode == "insert":
        for artifact_id in {item.artifact_id for item in items}:
            if _artifact_exists(collection, artifact_id):
                raise RuntimeError(
                    f"Collection '{collection}' already contains artifact_id '{artifact_id}'. Use --mode upsert."
                )
    if mode == "upsert":
        for artifact_id in {item.artifact_id for item in items}:
            if _artifact_exists(collection, artifact_id):
                _delete_artifact(collection, artifact_id)

    model = get_embedder(model_name, device)
    vectors = model.encode([item.text for item in items], normalize_embeddings=True, show_progress_bar=False)
    points = []
    for item, vector in zip(items, vectors):
        payload = dict(item.payload)
        payload["text"] = item.text
        vec_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec_list, payload=payload))
    client = get_qdrant()
    client.upsert(collection_name=collection, points=points)
    logger.info("Upserted %d artifact(s) into %s", len(points), collection)


def process_file(
    path: pathlib.Path, collection: Optional[str], mode: str, model_name: str, device: str, dry: bool
) -> None:
    logger.info("Processing %s", path)
    data = load_json(path)
    if not collection:
        stem = path.stem
        collection = f"{DEFAULT_COLLECTION_PREFIX}{stem}" if DEFAULT_COLLECTION_PREFIX else stem
    items = _artifact_items_for_file(data, collection)
    if not items:
        logger.warning("No artifacts to ingest from %s", path)
        return
    upsert_items(items, mode, model_name, device, dry)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest Omnipipe JSON artifacts into Qdrant")
    parser.add_argument("--root", type=pathlib.Path, help="Directory to scan for .json files")
    parser.add_argument("--file", type=pathlib.Path, help="Ingest a single .json file")
    parser.add_argument(
        "--collection",
        help="Qdrant collection name (defaults to JSON filename or OMNIPIPE_COLLECTION_PREFIX + filename)",
    )
    parser.add_argument(
        "--mode",
        choices=["overwrite", "upsert", "insert", "dry-run"],
        default="upsert",
        help="Ingestion mode (overwrite clears the target collection before ingesting)",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBED_MODEL,
        help="SentenceTransformer model name (default: EMBED_MODEL env)",
    )
    parser.add_argument(
        "--embedding-device",
        default=DEFAULT_EMBED_DEVICE,
        help="Device for embeddings (default: EMBED_DEVICE env)",
    )
    args = parser.parse_args(argv)

    if args.file and args.root:
        parser.error("Use --root or --file, not both")
    if not args.file and not args.root:
        parser.error("Provide --root or --file")

    mode = args.mode
    dry = mode == "dry-run"

    files = gather_files(args.root, args.file)
    if not files:
        logger.warning("No JSON files found for ingestion")
        return

    for path in files:
        try:
            process_file(path, args.collection, mode, args.embedding_model, args.embedding_device, dry)
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", path, exc)
            if not dry:
                raise


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
