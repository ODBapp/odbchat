#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ingest_mhw.py
- 支援 single-doc 與 multi-doc YAML（同檔上半段 usage note，下半段 OAS）
- 將 OAS 以「完整 YAML 純文字」寫入 Qdrant（doc_type=api_spec），避免切片破壞結構
- 其他 doc 依類型採不同 chunk 策略
- --mode overwrite|add 與 --dry-run
"""

import os
import re
import sys
import uuid
import yaml
import argparse
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

from sentence_transformers import SentenceTransformer

# ========== 基本設定 ==========
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION  = os.environ.get("QDRANT_COL", "odb_mhw_knowledge_v1")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "thenlper/gte-small")  # 多語好用
EMBED_DIM   = 384

# 掃描根目錄（預設 rag/）
DEFAULT_ROOT = "rag"

# OAS 偵測
OAS_HEAD_RE = re.compile(r"^\s*(openapi|swagger)\s*:\s*.*$", re.I)

# ========== 初始化 ==========
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
qdrant   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ========== 小工具 ==========
def ensure_collection(mode: str, dry: bool=False):
    exists = False
    try:
        lst = qdrant.get_collections().collections
        exists = any(c.name == COLLECTION for c in lst)
    except Exception:
        pass

    if mode == "overwrite":
        if dry:
            print(f"[DRY] recreate_collection({COLLECTION})")
        else:
            # 新版 API 建議：若存在先刪除再 create
            if exists:
                qdrant.delete_collection(COLLECTION)
            qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
    else:
        # add：若不存在才建
        if not exists:
            if dry:
                print(f"[DRY] create_collection({COLLECTION})")
            else:
                qdrant.create_collection(
                    collection_name=COLLECTION,
                    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                )

def encode_chunks(chunks: List[str]) -> List[List[float]]:
    return embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False).tolist()

def upsert_chunks(chunks: List[str], base_meta: Dict[str, Any], dry: bool=False):
    if not chunks:
        return
    vecs = encode_chunks(chunks)
    if dry:
        print(f"[DRY] upsert {len(chunks)} chunks for doc_id={base_meta.get('doc_id','(none)')}")
        return
    points = []
    for i, (txt, vec) in enumerate(zip(chunks, vecs)):
        pl = dict(base_meta)
        pl["text"] = txt
        pl["chunk_id"] = i
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=pl))
    qdrant.upsert(collection_name=COLLECTION, points=points)

def chunk_text(txt: str, doc_type: str) -> List[str]:
    if not txt:
        return []
    # OAS 一律單塊，避免分割造成 YAML 解析失敗
    if doc_type == "api_spec":
        return [txt.strip()]
    # 文件/教學/規格導覽：短片段以利檢索準確
    if doc_type in {"api_guide","api_spec_guide","tutorial","cli_tool_guide","code_snippet","code_example"}:
        size, step = 900, 800
    else:
        size, step = 1200, 1000
    out, i = [], 0
    while i < len(txt):
        out.append(txt[i:i+size].strip())
        i += step
    return [x for x in out if x]

def is_oas_doc(obj: Any, raw_text: Optional[str]=None) -> bool:
    # 若傳進來的是 dict（yaml 解析後），看有無 openapi/swagger 鍵
    if isinstance(obj, dict) and any(k in obj for k in ("openapi", "swagger")):
        return True
    # 若是純文字，直接檢查首行/任一行
    if isinstance(raw_text, str):
        for ln in raw_text.splitlines():
            if OAS_HEAD_RE.match(ln):
                return True
    return False

def load_multidoc_yaml(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    回傳：
      - docs_meta: 所有以 dict 形式載入成功的 YAML 物件（第一段通常是 front-matter+content）
      - raw_docs:  每一段 YAML 的「原始文字」（之後可決定要不要原封不動放進向量庫）
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # 用 yaml.safe_load_all 逐段解析
    docs_meta: List[Dict[str,Any]] = []
    raw_docs: List[str] = []
    # 先切段：--- 作為分隔（確保是行首）
    parts = re.split(r"(?m)^\s*---\s*$", raw)
    # 如果檔案最開頭就有 '---'，parts[0] 會是空字串
    # 我們用原始切段，回頭各段分別解析（容忍某段不能解析成 dict）
    for seg in parts:
        seg = seg.strip()
        if not seg:
            continue
        raw_docs.append(seg)
        try:
            parsed = yaml.safe_load(seg)
            if isinstance(parsed, dict):
                docs_meta.append(parsed)
            else:
                docs_meta.append({})
        except Exception:
            docs_meta.append({})
    return docs_meta, raw_docs

def build_meta_for_oas(base_meta: Dict[str,Any]) -> Dict[str,Any]:
    meta = dict(base_meta or {})
    meta["doc_type"] = "api_spec"
    # 避免和上半段重名
    t = meta.get("title") or "OpenAPI"
    if "(OAS)" not in t:
        meta["title"] = f"{t} (OAS)"
    return meta

def ingest_one_yaml(path: str, dry: bool=False):
    docs_meta, raw_docs = load_multidoc_yaml(path)
    if not raw_docs:
        print("[SKIP] empty:", path)
        return

    # 第一段通常是 front-matter（含 content: | 的 usage note），用它做為 base_meta
    base_meta = docs_meta[0] if docs_meta else {}
    if not isinstance(base_meta, dict):
        base_meta = {}
    title = base_meta.get("title") or os.path.basename(path)
    coll  = base_meta.get("collection") or "mhw"
    doc_type = base_meta.get("doc_type") or "note"

    # 嘗試從第一段拿出 content
    content_text = ""
    if "content" in base_meta and isinstance(base_meta["content"], str):
        content_text = base_meta["content"]

    # 先 upsert 第一段（如果有內容）
    if content_text.strip():
        m1 = dict(base_meta)
        m1.setdefault("title", title)
        m1.setdefault("collection", coll)
        m1.setdefault("doc_type", doc_type)
        m1.setdefault("source_file", path)
        chunks = chunk_text(content_text, m1["doc_type"])
        upsert_chunks(chunks, m1, dry=dry)

    # 後續段：尋找 OAS 或其他 YAML 段
    for i in range(1, len(raw_docs)):
        raw_seg = raw_docs[i]
        seg_meta = docs_meta[i] if i < len(docs_meta) else {}
        seg_meta = seg_meta if isinstance(seg_meta, dict) else {}

        # 判定是不是 OAS
        if is_oas_doc(seg_meta, raw_text=raw_seg):
            # 把整段 OAS YAML 文字原封上傳（不切）
            m2 = build_meta_for_oas(base_meta)
            m2["source_file"] = path
            # 若 seg_meta 有語言/版本之類的，合併進來
            for k in ("lang","version","tags"):
                if k in seg_meta:
                    m2[k] = seg_meta[k]
            chunks = [raw_seg.strip()]
            upsert_chunks(chunks, m2, dry=dry)
        else:
            # 不是 OAS 的第二段：當作一般文件（若該段含 content/或 seg 本身是純文本需處理）
            text2 = ""
            if "content" in seg_meta and isinstance(seg_meta["content"], str):
                text2 = seg_meta["content"]
            else:
                # 若 seg_meta 是 dict 但沒有 content，就把整段 YAML dump 成文字存一份（方便檢索）
                if isinstance(seg_meta, dict) and seg_meta:
                    text2 = yaml.safe_dump(seg_meta, sort_keys=False)
                else:
                    # 生文本（rare）
                    text2 = raw_seg
            if text2.strip():
                m3 = dict(base_meta)
                # 有自己的 doc_type 就用，否則沿用第一段
                if "doc_type" in seg_meta:
                    m3["doc_type"] = seg_meta["doc_type"]
                m3.setdefault("title", f"{title} (part-{i+1})")
                m3["source_file"] = path
                chunks = chunk_text(text2, m3.get("doc_type","note"))
                upsert_chunks(chunks, m3, dry=dry)

def walk_files(root: str) -> List[str]:
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.endswith(".yml") or f.endswith(".yaml"):
                paths.append(os.path.join(r, f))
    return sorted(paths)

# ========== 主程式 ==========
def main():
    ap = argparse.ArgumentParser(description="Ingest ODB MHW multi-doc YAML into Qdrant")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="root folder (default: rag)")
    ap.add_argument("--mode", choices=["overwrite","add"], default="add")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_collection(args.mode, dry=args.dry_run)

    files = walk_files(args.root)
    pbar = tqdm(files, desc=f"Ingest MHW ({args.mode})")
    total_chunks = 0

    for path in pbar:
        try:
            # 只處理你們 repo 的 YAML（其餘可自行加白名單規則）
            ingest_one_yaml(path, dry=args.dry_run)
        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    print("[DONE] Ingestion finished.")

if __name__ == "__main__":
    main()
