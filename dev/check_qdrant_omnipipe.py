#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
check_qdrant_omnipipe.py
- 掃描 Qdrant collection（分頁 scroll）
- 輸出 Omnipipe payload 欄位分佈（doc_type / source_file / collection / tags）
- 統計 links / link_type 分佈、是否含 api_endpoint / table / section
- 以 SentenceTransformer (thenlper/gte-small) 做測試檢索並列出 Top-K
"""

import os
import re
import json
import argparse
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient

try:
    import qdrant_client as _qdc
    _QDRANT_CLIENT_VER = getattr(_qdc, "__version__", "unknown")
except Exception:
    _QDRANT_CLIENT_VER = "unknown"

_EMBED_MODEL_DEFAULT = os.environ.get("EMBED_MODEL", "thenlper/gte-small")
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inspect and probe an Omnipipe Qdrant collection.")
    ap.add_argument("--host", default=os.environ.get("QDRANT_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("QDRANT_PORT", "6333")))
    ap.add_argument("--collection", default=os.environ.get("QDRANT_COL", "AI"))
    ap.add_argument("--max-pages", type=int, default=0, help="最大分頁數（0 代表盡量全量）")
    ap.add_argument("--page-size", type=int, default=256, help="每頁點數（scroll limit）")
    ap.add_argument("--sample", type=int, default=8, help="抽樣顯示幾條 payload 摘要")
    ap.add_argument("--topk", type=int, default=12, help="測試檢索 Top-K")
    ap.add_argument("--no-search", action="store_true", help="不執行測試檢索")
    ap.add_argument("--queries-file", default="", help="自訂查詢 JSON 檔（[\"q1\", \"q2\", ...]）")
    return ap.parse_args()


class QueryEncoder:
    def __init__(self, model_name: str = _EMBED_MODEL_DEFAULT, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.encoder = None
        if _HAS_ST:
            try:
                self.encoder = SentenceTransformer(model_name, device=device)
            except Exception as e:
                print(f"[WARN] SentenceTransformer init failed: {e}")
        else:
            print("[WARN] sentence-transformers not installed; search() will be skipped.")

    def encode(self, text: str) -> Optional[List[float]]:
        if not self.encoder:
            return None
        v = self.encoder.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return v.tolist() if hasattr(v, "tolist") else list(v)


def get_payload_text(payload: Dict[str, Any]) -> str:
    txt = payload.get("text")
    if isinstance(txt, str):
        return txt
    for k in ("content", "body", "raw", "markdown_content"):
        v = payload.get(k)
        if isinstance(v, str):
            return v
    return ""


def as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _normalize_scroll_response(resp: Any) -> Tuple[List[Any], Any]:
    points = []
    next_off = None

    if isinstance(resp, tuple) and len(resp) == 2:
        records, next_off = resp
        points = records if isinstance(records, list) else []
        return points, next_off

    if isinstance(resp, dict):
        points = resp.get("points", [])
        next_off = (resp.get("next_page_offset") or
                    resp.get("next_offset") or
                    resp.get("offset") or
                    resp.get("next_page_token"))
        if isinstance(resp, list):
            points = resp
            next_off = None
        return points, next_off

    pts = getattr(resp, "points", None)
    if isinstance(pts, list):
        points = pts
    else:
        if isinstance(resp, list):
            points = resp

    for attr in ("next_page_offset", "next_offset", "offset", "next_page_token"):
        if hasattr(resp, attr):
            val = getattr(resp, attr)
            if val is not None:
                next_off = val
                break

    return points, next_off


class QdrantScanner:
    def __init__(self, host: str, port: int, collection: str):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection

    def count(self) -> int:
        try:
            resp = self.client.count(self.collection, exact=True)
            cnt = getattr(resp, "count", None)
            if cnt is None and isinstance(resp, dict):
                cnt = resp.get("count")
            return int(cnt or 0)
        except Exception:
            return -1

    def scroll_iter(self, page_size: int = 256, max_pages: int = 0) -> Iterable[Dict[str, Any]]:
        page = 0
        offset_token = None
        while True:
            try:
                resp = self.client.scroll(
                    collection_name=self.collection,
                    limit=page_size,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset_token
                )
            except TypeError:
                resp = self.client.scroll(
                    collection_name=self.collection,
                    limit=page_size,
                    with_payload=True,
                    with_vectors=False,
                    page_offset=offset_token
                )

            points, next_off = _normalize_scroll_response(resp)

            for p in points:
                if isinstance(p, dict):
                    payload = p.get("payload")
                else:
                    payload = getattr(p, "payload", None)
                if isinstance(payload, dict):
                    yield payload

            page += 1
            if max_pages and page >= max_pages:
                break
            if not next_off:
                break
            offset_token = next_off

    def query(self, vector: List[float], topk: int = 12):
        resp = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=topk,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(resp, "points", None) or resp
        out = []
        for p in points:
            if isinstance(p, dict):
                pay = p.get("payload") or {}
                score = p.get("score")
                if score is None:
                    score = p.get("similarity") or p.get("distance")
            else:
                pay = getattr(p, "payload", None) or {}
                score = getattr(p, "score", None)
                if score is None:
                    score = getattr(p, "similarity", None) or getattr(p, "distance", None)
            out.append((score, pay))
        return out


def summarize_payloads(payload_iter: Iterable[Dict[str, Any]], sample_n: int = 8) -> Dict[str, Any]:
    total = 0
    c_doc_type = Counter()
    c_collection = Counter()
    c_source_file = Counter()
    c_tags = Counter()
    c_has_links = Counter()
    c_link_types = Counter()
    c_has_path = Counter()
    c_missing = Counter()
    c_table_missing = Counter()

    samples = []

    for payload in payload_iter:
        total += 1
        dt = str(payload.get("doc_type", "")).lower()
        coll = str(payload.get("collection", ""))
        src = str(payload.get("source_file", ""))
        tags = as_list(payload.get("tags"))
        links = payload.get("links") or []

        c_doc_type[dt] += 1
        c_collection[coll] += 1
        c_source_file[src] += 1
        for t in tags:
            c_tags[str(t).strip().lower()] += 1

        if isinstance(links, list) and links:
            c_has_links["has_links"] += 1
            for link in links:
                if isinstance(link, dict):
                    lt = str(link.get("link_type") or "").lower()
                    if lt:
                        c_link_types[lt] += 1
        else:
            c_has_links["no_links"] += 1

        if payload.get("path"):
            c_has_path["has_path"] += 1

        if not payload.get("doc_type"):
            c_missing["doc_type"] += 1
        if not payload.get("dataset_name"):
            c_missing["dataset_name"] += 1
        if not payload.get("title"):
            c_missing["title"] += 1
        if not payload.get("tags"):
            c_missing["tags"] += 1

        if dt == "table":
            if not payload.get("caption"):
                c_table_missing["caption"] += 1
            if not payload.get("markdown_content"):
                c_table_missing["markdown_content"] += 1
            if not payload.get("table_label"):
                c_table_missing["table_label"] += 1

        if len(samples) < sample_n:
            samples.append({
                "artifact_id": payload.get("artifact_id"),
                "doc_type": dt,
                "source_file": src,
                "tags": tags[:6],
                "links": len(links) if isinstance(links, list) else 0,
            })

    return {
        "total": total,
        "doc_type": c_doc_type,
        "collection": c_collection,
        "source_file": c_source_file,
        "tags": c_tags,
        "links": c_has_links,
        "link_types": c_link_types,
        "has_path": c_has_path,
        "missing_fields": c_missing,
        "table_missing": c_table_missing,
        "samples": samples,
    }


def print_top(counter: Counter, k: int = 12, label: str = ""):
    items = counter.most_common(k)
    if label:
        print(f"\n[Top {k}] {label}")
    for name, cnt in items:
        print(f"  {name or '(blank)'} : {cnt}")


def pretty_print_summary(summary: Dict[str, Any]):
    total = summary["total"]
    print("=" * 80)
    print(f"[COLLECTION SUMMARY] total payloads scanned: {total}")
    print("=" * 80)

    print_top(summary["doc_type"], 24, "doc_type")
    print_top(summary["collection"], 12, "collection")
    print_top(summary["tags"], 24, "tags (normalized)")

    with_links = summary["links"].get("has_links", 0)
    print(f"\n[Links] {with_links} / {total}  ({(100.0*with_links/max(1,total)):.1f}%)")
    print_top(summary["link_types"], 12, "link_type")

    has_path = summary["has_path"].get("has_path", 0)
    print(f"\n[API endpoints] {has_path} / {total}  ({(100.0*has_path/max(1,total)):.1f}%)")

    missing = summary.get("missing_fields", Counter())
    if missing:
        print("\n[Missing fields]")
        for name, cnt in missing.items():
            print(f"  {name}: {cnt}")

    table_missing = summary.get("table_missing", Counter())
    if table_missing:
        print("\n[Table missing fields]")
        for name, cnt in table_missing.items():
            print(f"  {name}: {cnt}")

    print("\n[Samples]")
    for i, s in enumerate(summary["samples"], 1):
        print(f"  {i:02d}. {s['artifact_id']} | {s['doc_type']} | links={s['links']} | tags={s['tags']}")


def run_search_probes(scanner: QdrantScanner, encoder: QueryEncoder, topk: int, user_queries: Optional[List[str]] = None):
    if not encoder.encoder:
        print("\n[Search] sentence-transformers not available; skip probing.")
        return

    default_queries = [
        "WOA23 data variables depth ranges",
        "WOA23 Table 4 variables",
        "MHW data variables",
        "Niño 3.4 long-term trend plot",
        "ODB MHW API endpoint parameters",
    ]

    queries = user_queries if user_queries else default_queries

    print("\n" + "=" * 80)
    print("[SEARCH PROBES]")
    print("=" * 80)

    for q in queries:
        vec = encoder.encode(q)
        if vec is None:
            continue
        print(f"\nQ: {q}")
        hits = scanner.query(vec, topk=topk)

        by_doc_type = Counter()
        show = []
        for i, (score, payload) in enumerate(hits, 1):
            dt = (payload.get("doc_type") or "").lower()
            by_doc_type[dt] += 1
            title = payload.get("title") or payload.get("artifact_id") or "(untitled)"
            show.append((i, score, dt, title, as_list(payload.get("tags"))[:5]))

        print_top(by_doc_type, 12, "Top-K doc_type")

        for i, score, dt, title, tags in show[: min(8, len(show))]:
            score_str = f"{score:.4f}" if isinstance(score, (float, int)) else "-"
            print(f"  #{i:02d}  score={score_str}  [{dt}]  {title!r}  tags={tags}")


def main():
    args = parse_args()
    print(f"[INFO] qdrant_client version: {_QDRANT_CLIENT_VER}")
    print(f"[INFO] Connecting to Qdrant: {args.host}:{args.port}, collection={args.collection}")

    scanner = QdrantScanner(host=args.host, port=args.port, collection=args.collection)
    total = scanner.count()
    if total >= 0:
        print(f"[INFO] Collection count (exact): {total}")
    else:
        print("[WARN] Could not get exact count; proceeding with scroll scan.")

    payloads = scanner.scroll_iter(page_size=args.page_size, max_pages=args.max_pages)
    summary = summarize_payloads(payloads, sample_n=args.sample)
    pretty_print_summary(summary)

    if summary["total"] == 0:
        print("\n[WARN] No payloads were scanned. This usually means the client/server 'scroll' "
              "API returned in a shape we didn't parse before, or the points have no payloads.")

    if not args.no_search:
        user_queries = None
        if args.queries_file and os.path.isfile(args.queries_file):
            try:
                with open(args.queries_file, "r", encoding="utf-8") as f:
                    arr = json.load(f)
                    if isinstance(arr, list):
                        user_queries = [str(x) for x in arr]
            except Exception as e:
                print(f"[WARN] Failed to load queries file: {e}")

        encoder = QueryEncoder(model_name=_EMBED_MODEL_DEFAULT, device="cpu")
        run_search_probes(scanner, encoder, topk=args.topk, user_queries=user_queries)


if __name__ == "__main__":
    main()
