#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import yaml
import uuid
import hashlib
import argparse
import warnings
from typing import List, Dict, Any, Tuple, Optional

# silence CUDA warnings from torch used by sentence-transformers
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer


# ----------------------------
# Defaults (override via CLI)
# ----------------------------
DEFAULT_ROOT = "rag_doc"
DEFAULT_COLLECTION = "odb_mhw_knowledge_v1"
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_EMBED_MODEL = "thenlper/gte-small"   # 384-dim, multilingual
DEFAULT_EMBED_DIM = 384
DEFAULT_MODE = "add"                         # add | overwrite
DEFAULT_DUP = "skip"                         # when mode=add: skip | replace | upsert


def parse_args():
    ap = argparse.ArgumentParser(
        description="Ingest MHW YAML docs (front-matter + content: |) and multi-doc OAS into Qdrant."
    )
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Root folder (e.g., rag_doc)")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name")
    ap.add_argument("--qdrant-host", default=DEFAULT_QDRANT_HOST)
    ap.add_argument("--qdrant-port", type=int, default=DEFAULT_QDRANT_PORT)
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)

    ap.add_argument("--mode", choices=["overwrite", "add"], default=DEFAULT_MODE,
                    help="overwrite: drop & recreate collection; add: append without duplicates")
    ap.add_argument("--duplicate", choices=["skip", "replace", "upsert"], default=DEFAULT_DUP,
                    help="Behavior when doc_id already exists (only applies to --mode add)")
    ap.add_argument("--dry-run", action="store_true", help="Parse only; no Qdrant writes")
    return ap.parse_args()


# ----------------------------
# YAML loaders
# ----------------------------
def load_single_doc_yaml(yaml_path: str) -> Tuple[Dict[str, Any], str]:
    """
    Single YAML mapping with 'content' key (string).
    """

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{yaml_path} is not a valid YAML mapping")

    content = data.get("content", "") or ""
    meta = dict(data)
    meta.pop("content", None)

    return meta, str(content)


def load_multi_doc_yaml(yaml_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Multi-doc YAML support. Returns (meta_doc, oas_doc) where:
      - meta_doc: first YAML document (front-matter), dict or None
      - oas_doc : second YAML document if contains 'openapi' or 'swagger', else None
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))

    if not docs:
        return None, None

    meta_doc = docs[0] if isinstance(docs[0], dict) else None

    oas_doc = None
    for d in docs[1:]:
        if isinstance(d, dict) and (("openapi" in d) or ("swagger" in d)):
            oas_doc = d
            break

    return meta_doc, oas_doc


def load_frontmatter_content_or_oas(yaml_path: str) -> Tuple[Dict[str, Any], str, Optional[Dict[str, Any]]]:
    """
    Try multi-doc first; if not, fallback to single-doc.
    Returns: (meta, content_text, oas_dict_or_None)
    """
    try:
        meta_doc, oas_doc = load_multi_doc_yaml(yaml_path)
        if meta_doc is not None:
            meta = dict(meta_doc)
            content = meta.pop("content", "") or ""
            return meta, str(content), oas_doc
    except yaml.YAMLError:
        pass

    meta, content = load_single_doc_yaml(yaml_path)
    return meta, content, None


# ----------------------------
# OpenAPI summarizer
# ----------------------------
def _fmt(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, (int, float, bool)):
        return str(s)
    return str(s).strip()

def _schema_to_brief(schema: Dict[str, Any]) -> str:
    if not isinstance(schema, dict):
        return ""
    t = schema.get("type")
    fmt = schema.get("format")
    title = schema.get("title")
    anyOf = schema.get("anyOf")
    oneOf = schema.get("oneOf")
    items = schema.get("items")
    if t:
        out = t
        if fmt:
            out += f"({fmt})"
        if title:
            out += f" «{title}»"
        if t == "array" and isinstance(items, dict):
            out += f"[{_schema_to_brief(items)}]"
        return out
    if anyOf and isinstance(anyOf, list):
        return " | ".join(_schema_to_brief(x) for x in anyOf)
    if oneOf and isinstance(oneOf, list):
        return " || ".join(_schema_to_brief(x) for x in oneOf)
    if title:
        return f"«{title}»"
    return ""

def textwrap_indent(s: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line if line.strip() else line for line in s.splitlines())

def render_oas_summary(oas: Dict[str, Any]) -> str:
    """
    OpenAPI/Swagger dict -> compact text summary:
      - servers, info(title/version/desc)
      - paths: method, summary/operationId
      - parameters: name/in/required/type
      - responses: status + schema hints
    """
    if not isinstance(oas, dict):
        return ""

    lines: List[str] = []
    if "openapi" in oas:
        lines.append(f"# OpenAPI {oas.get('openapi')}")
    elif "swagger" in oas:
        lines.append(f"# Swagger {oas.get('swagger')}")

    info = oas.get("info", {}) or {}
    title = info.get("title") or ""
    version = info.get("version") or ""
    desc = info.get("description") or ""
    if title or version:
        lines.append(f"Info: {title} (v{version})")
    if desc:
        lines.append("Description:")
        lines.append(textwrap_indent(_fmt(desc), 2))

    servers = oas.get("servers") or []
    if isinstance(servers, list) and servers:
        lines.append("Servers:")
        for s in servers:
            url = s.get("url") if isinstance(s, dict) else None
            if url:
                lines.append(f"  - {url}")

    paths = oas.get("paths", {}) or {}
    if isinstance(paths, dict) and paths:
        lines.append("Paths:")
        for pth, obj in paths.items():
            if not isinstance(obj, dict):
                continue
            for method, meta in obj.items():
                if method.lower() not in ("get", "post", "put", "delete", "patch", "head", "options"):
                    continue
                if not isinstance(meta, dict):
                    continue
                summary = meta.get("summary") or meta.get("operationId") or ""
                lines.append(f"- {method.upper()} {pth} :: {summary}")

                params = meta.get("parameters") or []
                if isinstance(params, list) and params:
                    lines.append("  params:")
                    for pr in params:
                        if not isinstance(pr, dict):
                            continue
                        nm = pr.get("name")
                        loc = pr.get("in")
                        req = pr.get("required", False)
                        sch = pr.get("schema") or {}
                        lines.append(f"    - {nm} ({loc}) req={req} type={_schema_to_brief(sch)}")

                if "requestBody" in meta and isinstance(meta["requestBody"], dict):
                    rb = meta["requestBody"]
                    req = rb.get("required", False)
                    lines.append(f"  requestBody: required={req}")
                    content = rb.get("content") or {}
                    for ctype, spec in (content.items() if isinstance(content, dict) else []):
                        schema = (spec.get("schema") if isinstance(spec, dict) else {}) or {}
                        lines.append(f"    - {ctype}: {_schema_to_brief(schema)}")

                responses = meta.get("responses") or {}
                if isinstance(responses, dict) and responses:
                    lines.append("  responses:")
                    for code, rmeta in responses.items():
                        if not isinstance(rmeta, dict):
                            continue
                        dsc = rmeta.get("description", "")
                        dsc_txt = f" - {dsc}" if dsc else ""
                        lines.append(f"    - {code}{dsc_txt}")
                        content = rmeta.get("content") or {}
                        if isinstance(content, dict):
                            for ctype, spec in content.items():
                                schema = (spec.get("schema") if isinstance(spec, dict) else {}) or {}
                                lines.append(f"      * {ctype}: {_schema_to_brief(schema)}")

    return "\n".join(lines).strip()


# ----------------------------
# Utilities
# ----------------------------
def stable_doc_id(meta: Dict[str, Any]) -> str:
    """
    If 'doc_id' provided use it; otherwise derive from canonical_url or source_file (deterministic).
    """
    if "doc_id" in meta and meta["doc_id"]:
        return str(meta["doc_id"])
    basis = meta.get("canonical_url") or meta.get("source_file") or repr(sorted(meta.items()))
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()

def normalize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(meta)

    # MHW-only phase
    out.setdefault("dataset", meta.get("dataset_name", "marineheatwave"))
    out.setdefault("collection", "mhw")

    out.setdefault("doc_type", "note")
    out.setdefault("lang", "zh")
    out.setdefault("issuer", "Unknown")
    out.setdefault("canonical_url", "")
    out.setdefault("license", "Unknown")
    out.setdefault("retrieved_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out.setdefault("source_type", "internal")
    out.setdefault("source_file", "")

    if not out.get("title"):
        fn = os.path.basename(out.get("source_file") or "untitled")
        out["title"] = os.path.splitext(fn)[0]

    out["doc_id"] = stable_doc_id(out)

    for k in ("tags", "related_doc", "depends_on", "faq"):
        if k in out and isinstance(out[k], list):
            out[k] = [x for x in out[k] if str(x).strip()]

    return out

def chunk_text(text: str, doc_type: str) -> List[str]:
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

def ensure_collection(client: QdrantClient, collection: str, dim: int, mode: str, dry: bool):
    exists = client.collection_exists(collection)
    if mode == "overwrite":
        if exists:
            if dry:
                print(f"[DRY] delete_collection({collection})")
            else:
                client.delete_collection(collection)
        if dry:
            print(f"[DRY] create_collection({collection})")
        else:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
    else:  # add
        if not exists:
            if dry:
                print(f"[DRY] create_collection({collection})")
            else:
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )

def doc_exists(client: QdrantClient, collection: str, doc_id: str) -> bool:
    flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    sc, _ = client.scroll(collection_name=collection, scroll_filter=flt, limit=1, with_payload=False)
    return len(sc) > 0

def delete_doc(client: QdrantClient, collection: str, doc_id: str, dry: bool):
    flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    if dry:
        print(f"[DRY] delete where doc_id == {doc_id}")
    else:
        client.delete(collection_name=collection, points_selector=flt)

def embed_many(embedder: SentenceTransformer, chunks: List[str]) -> List[List[float]]:
    if not chunks:
        return []
    return embedder.encode(chunks, normalize_embeddings=True).tolist()

def build_point_id(doc_id: str, chunk_idx: int) -> str:
    """
    Generate deterministic UUIDv5 from doc_id + chunk_idx (valid for Qdrant >= 1.15).
    """
    name = f"{doc_id}:{chunk_idx}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))

def upsert_doc(
    client: QdrantClient,
    collection: str,
    embedder: SentenceTransformer,
    meta: Dict[str, Any],
    content: str,
    dry: bool,
    duplicate: str,
) -> int:
    """
    Ingest one logical document (meta + prepared content).
    Returns number of chunks written (or would write in dry-run).
    """
    if meta.get("dataset") != "marineheatwave" or meta.get("collection") != "mhw":
        return 0

    doc_id = meta["doc_id"]
    exists = doc_exists(client, collection, doc_id)

    if exists:
        if duplicate == "skip":
            msg = f"[SKIP] doc_id exists: {doc_id}"
            print(f"[DRY] {msg}" if dry else msg)
            return 0
        elif duplicate == "replace":
            delete_doc(client, collection, doc_id, dry)

    chunks = chunk_text(content, meta.get("doc_type", "note"))
    if not chunks:
        return 0

    vectors = embed_many(embedder, chunks)
    points = []
    for i, (ch, vec) in enumerate(zip(chunks, vectors)):
        payload = dict(meta)
        payload.update({"text": ch, "chunk_id": i})
        pid = build_point_id(doc_id, i)
        points.append(PointStruct(id=pid, vector=vec, payload=payload))

    if dry:
        print(f"[DRY] upsert {len(points)} chunks for doc_id={doc_id}")
        return len(points)

    client.upsert(collection_name=collection, points=points)
    return len(points)

def ingest_file(
    path: str,
    client: QdrantClient,
    collection: str,
    embedder: SentenceTransformer,
    dry_run: bool,
    mode: str,
    duplicate: str,
) -> int:
    meta, content, oas = load_frontmatter_content_or_oas(path)

    # Skip top-level manifests (no content, has 'components') or explicit manifest types
    if (not content.strip()) and (oas is None) and (
        meta.get("doc_type", "").lower().startswith("manifest") or "components" in meta
    ):
        return 0

    meta.setdefault("source_file", os.path.relpath(path, start=os.getcwd()))
    meta = normalize_meta(meta)

    prepared_content = content.strip()

    # If OAS present (openapi or swagger), render a compact summary and append
    if isinstance(oas, dict) and (("openapi" in oas) or ("swagger" in oas)):
        summary = render_oas_summary(oas)
        if summary:
            prepared_content = (prepared_content + "\n\n## OpenAPI Summary\n" + summary).strip()

    if not prepared_content:
        return 0

    return upsert_doc(client, collection, embedder, meta, prepared_content, dry_run, duplicate)

def main():
    args = parse_args()
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    ensure_collection(client, args.collection, args.embed_dim, args.mode, args.dry_run)
    embedder = SentenceTransformer(args.embed_model)

    # gather YAMLs
    files = glob.glob(os.path.join(args.root, "**", "*.yml"), recursive=True)
    files += glob.glob(os.path.join(args.root, "**", "*.yaml"), recursive=True)

    if not files:
        print(f"[WARN] No .yml/.yaml files under {args.root}")
        return

    total = 0
    for p in tqdm(sorted(files), desc=f"Ingest MHW ({args.mode}/{args.duplicate})"):
        try:
            total += ingest_file(
                path=p,
                client=client,
                collection=args.collection,
                embedder=embedder,
                dry_run=args.dry_run,
                mode=args.mode,
                duplicate=args.duplicate,
            )
        except Exception as e:
            print(f"[ERROR] {p}: {e}")

    print(f"[DONE] {'[DRY] ' if args.dry_run else ''}Upserted chunks: {total}")

if __name__ == "__main__":
    main()
