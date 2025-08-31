#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
check_qdrant.py
- 掃描 Qdrant collection（分頁 scroll，全量或指定最大頁數）
- 輸出欄位分佈（doc_type / source_type / collection / lang / issuer / tags）
- 偵測 code-like 文件（doc_type=code_snippet/code_example、source_type=code、含 fenced code）
- 偵測 OAS（doc_type=api_spec 或文字內含 openapi/swagger 標記）
- 以 SentenceTransformer (thenlper/gte-small) 做幾組測試檢索並列出 Top-K 分佈
- 參數化 CLI；對不同版本 qdrant_client 的 scroll/page_offset/next_offset 做容錯
"""

import os
import sys
import re
import json
import argparse
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client import models  # for filter/conditions & offset types

# 版本資訊（方便除錯）
try:
    import qdrant_client as _qdc
    _QDRANT_CLIENT_VER = getattr(_qdc, "__version__", "unknown")
except Exception:
    _QDRANT_CLIENT_VER = "unknown"

# Embedding（預設與 ingest_mhw.py 同：thenlper/gte-small）
_EMBED_MODEL_DEFAULT = os.environ.get("EMBED_MODEL", "thenlper/gte-small")
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


# ========= CLI =========

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inspect and probe a Qdrant collection for ODB MHW RAG.")
    ap.add_argument("--host", default=os.environ.get("QDRANT_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("QDRANT_PORT", "6333")))
    ap.add_argument("--collection", default=os.environ.get("QDRANT_COL", "odb_mhw_knowledge_v1"))
    ap.add_argument("--max-pages", type=int, default=0, help="最大分頁數（0 代表盡量全量）")
    ap.add_argument("--page-size", type=int, default=256, help="每頁點數（scroll limit）")
    ap.add_argument("--sample", type=int, default=8, help="抽樣顯示幾條 payload 摘要")
    ap.add_argument("--topk", type=int, default=12, help="測試檢索 Top-K")
    ap.add_argument("--no-search", action="store_true", help="不執行測試檢索")
    ap.add_argument("--queries-file", default="", help="自訂查詢 JSON 檔（[\"q1\", \"q2\", ...]）")
    return ap.parse_args()


# ========= Embedding =========

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


# ========= Utilities =========

def get_payload_text(payload: Dict[str, Any]) -> str:
    """取出文字欄位（ingest 時通常放在 payload['text']）"""
    txt = payload.get("text")
    if isinstance(txt, str):
        return txt
    # 兼容某些格式
    for k in ("content", "body", "raw"):
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

def has_code_fence(text: str) -> bool:
    """偵測是否含 ```xxx fenced code"""
    if not text:
        return False
    return bool(re.search(r"```[a-zA-Z0-9_-]*\s+[\s\S]+?```", text))

def count_code_fences(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"```[a-zA-Z0-9_-]*\s+[\s\S]+?```", text))

def is_code_doc(payload: Dict[str, Any]) -> bool:
    dt = (payload.get("doc_type") or "").lower()
    st = (payload.get("source_type") or "").lower()
    tags = ",".join(map(lambda x: str(x).lower(), as_list(payload.get("tags"))))
    if dt in ("code_snippet", "code_example"):
        return True
    if "code" in st:
        return True
    if "python" in tags or "程式" in tags or "example" in tags or "範例" in tags:
        return True
    # 再看 text 是否有 fenced code
    txt = get_payload_text(payload)
    return has_code_fence(txt)

def is_oas_doc(payload: Dict[str, Any]) -> bool:
    dt = (payload.get("doc_type") or "").lower()
    if dt == "api_spec":
        return True
    # 容錯：有些人會把 OAS 文放到 text，但 doc_type 沒設好
    txt = get_payload_text(payload)
    if re.search(r"^\s*(openapi|swagger)\s*:\s*.*$", txt, re.I | re.M):
        return True
    return False


# ========= Qdrant Wrappers =========

def _normalize_scroll_response(resp: Any) -> Tuple[List[Any], Any]:
    """
    兼容 qdrant_client 不同版本：
      - 可能回傳 tuple: (records: List[Record], next_offset)
      - 也可能是 dict: {'points': [...], 'next_page_offset': ...}
      - 或具屬性的物件: resp.points, resp.next_page_offset
    回傳 (points, next_offset)
    """
    points = []
    next_off = None

    # tuple 形式
    if isinstance(resp, tuple) and len(resp) == 2:
        records, next_off = resp
        points = records if isinstance(records, list) else []
        return points, next_off

    # dict 形式
    if isinstance(resp, dict):
        points = resp.get("points", [])
        next_off = (resp.get("next_page_offset") or
                    resp.get("next_offset") or
                    resp.get("offset") or
                    resp.get("next_page_token"))
        # 某些舊版可能直接回 List
        if isinstance(resp, list):
            points = resp
            next_off = None
        return points, next_off

    # 物件形式
    pts = getattr(resp, "points", None)
    if isinstance(pts, list):
        points = pts
    else:
        # 某些版本直接回 List
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
        """
        遍歷整個 collection 的 payload（with_payload=True, with_vectors=False）
        容錯處理不同客戶端版本的 offset / page_offset / next_offset 欄位。
        產生器返回 payload（dict），若點沒有 payload 就略過。
        """
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
                # 有些版本參數稱作 page_offset
                resp = self.client.scroll(
                    collection_name=self.collection,
                    limit=page_size,
                    with_payload=True,
                    with_vectors=False,
                    page_offset=offset_token
                )

            points, next_off = _normalize_scroll_response(resp)

            for p in points:
                # p 可能是 dict 或具 payload 屬性的物件
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

    def query(self, vector: List[float], topk: int = 12, query_filter: Optional[models.Filter] = None):
        resp = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=topk,
            with_payload=True,
            with_vectors=False,
            query_filter=query_filter
        )
        # 兼容各版本回傳
        points = getattr(resp, "points", None) or resp
        out = []
        for p in points:
            if isinstance(p, dict):
                pay = p.get("payload") or {}
                # 安全擷取 score
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


# ========= Analysis =========

def summarize_payloads(payload_iter: Iterable[Dict[str, Any]], sample_n: int = 8) -> Dict[str, Any]:
    N = 0
    c_doc_type = Counter()
    c_source_type = Counter()
    c_collection = Counter()
    c_lang = Counter()
    c_issuer = Counter()
    c_has_code = Counter()
    c_oas = Counter()
    tag_counter = Counter()

    samples = []

    for payload in payload_iter:
        N += 1
        dt = str(payload.get("doc_type", "")).lower()
        st = str(payload.get("source_type", "")).lower()
        coll = str(payload.get("collection", ""))
        lang = str(payload.get("lang", ""))
        issuer = str(payload.get("issuer", ""))
        tags = as_list(payload.get("tags"))

        c_doc_type[dt] += 1
        c_source_type[st] += 1
        c_collection[coll] += 1
        c_lang[lang] += 1
        c_issuer[issuer] += 1
        for t in tags:
            tag_counter[str(t).strip().lower()] += 1

        if is_code_doc(payload):
            c_has_code["code_like"] += 1
        else:
            c_has_code["non_code"] += 1

        if is_oas_doc(payload):
            c_oas["api_spec_like"] += 1
        else:
            c_oas["non_oas"] += 1

        # 抽樣
        if len(samples) < sample_n:
            samples.append({
                "title": payload.get("title"),
                "doc_type": dt,
                "source_type": st,
                "collection": coll,
                "lang": lang,
                "issuer": issuer,
                "tags": tags[:8],
            })

    return {
        "total": N,
        "doc_type": c_doc_type,
        "source_type": c_source_type,
        "collection": c_collection,
        "lang": c_lang,
        "issuer": c_issuer,
        "tags": tag_counter,
        "code_flag": c_has_code,
        "oas_flag": c_oas,
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
    print_top(summary["source_type"], 24, "source_type")
    print_top(summary["collection"], 12, "collection")
    print_top(summary["lang"], 12, "lang")
    print_top(summary["issuer"], 12, "issuer")

    print_top(summary["tags"], 24, "tags (normalized)")

    code_like = summary["code_flag"].get("code_like", 0)
    print(f"\n[Code-like docs] {code_like} / {total}  ({(100.0*code_like/max(1,total)):.1f}%)")

    oas_like = summary["oas_flag"].get("api_spec_like", 0)
    print(f"[OAS-like docs ] {oas_like} / {total}  ({(100.0*oas_like/max(1,total)):.1f}%)")

    print("\n[Samples]")
    for i, s in enumerate(summary["samples"], 1):
        print(f"  {i:02d}. {s['title']!r} | {s['doc_type']} | tags={s['tags']} | coll={s['collection']}")


# ========= Test Search =========

_CODE_HINT = [
    "code", "程式", "python", "範例", "example", "sample", "examples",
    "畫圖", "plot", "地圖", "時序", "API", "OpenAPI", "swagger", "請提供程式",
]

def detect_code_intent(q: str) -> bool:
    ql = q.lower()
    return any(w.lower() in ql for w in _CODE_HINT)

def run_search_probes(scanner: QdrantScanner, encoder: QueryEncoder, topk: int, user_queries: Optional[List[str]] = None):
    if not encoder.encoder:
        print("\n[Search] sentence-transformers not available; skip probing.")
        return

    default_general = [
        "請分析海洋熱浪對海洋生態的影響",
        "什麼是 ENSO？",
        "Marine heatwaves impact overview",
    ]
    default_code = [
        "請用程式畫出台灣附近 2024 年的 MHW 等級時間序列",
        "How to use ODB MHW API to fetch monthly SST anomalies near Taiwan? Provide Python example.",
        "用 Python 下載 ODB MHW API 的 CSV 並畫圖",
    ]

    queries = user_queries if user_queries else (default_general + default_code)

    print("\n" + "=" * 80)
    print("[SEARCH PROBES]")
    print("=" * 80)

    for q in queries:
        vec = encoder.encode(q)
        if vec is None:
            continue
        is_code = detect_code_intent(q)
        print(f"\nQ: {q}   [{'CODE' if is_code else 'GEN'}]")
        hits = scanner.query(vec, topk=topk, query_filter=None)

        by_doc_type = Counter()
        by_source = Counter()
        show = []
        for i, (score, payload) in enumerate(hits, 1):
            dt = (payload.get("doc_type") or "").lower()
            st = (payload.get("source_type") or "").lower()
            by_doc_type[dt] += 1
            by_source[st] += 1
            title = payload.get("title") or "(untitled)"
            show.append((i, score, dt, title, as_list(payload.get("tags"))[:5]))

        print_top(by_doc_type, 12, "Top-K doc_type")
        print_top(by_source, 12, "Top-K source_type")

        for i, score, dt, title, tags in show[: min(8, len(show))]:
            score_str = f"{score:.4f}" if isinstance(score, (float, int)) else "-"
            print(f"  #{i:02d}  score={score_str}  [{dt}]  {title!r}  tags={tags}")


# ========= Main =========

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

    # 全量或限制頁數掃描
    payloads = scanner.scroll_iter(page_size=args.page_size, max_pages=args.max_pages)
    summary = summarize_payloads(payloads, sample_n=args.sample)
    pretty_print_summary(summary)

    if summary["total"] == 0:
        print("\n[WARN] No payloads were scanned. This usually means the client/server 'scroll' "
              "API returned in a shape we didn't parse before, or the points have no payloads. "
              "If you still see 0 here while count > 0, please share the qdrant_client version shown above.")

    if not args.no_search:
        # 自訂 queries（JSON 檔）或預設 probes
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
