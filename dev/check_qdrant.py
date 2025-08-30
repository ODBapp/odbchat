#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
debug_qdrant_query.py
調試 Qdrant 查詢為什麼總是返回相同的文檔
"""

import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 設定（與你的 ingest_mhw.py 保持一致）
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COL = os.environ.get("QDRANT_COL", "odb_mhw_knowledge_v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "thenlper/gte-small")

# 初始化
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def debug_collection_stats():
    """檢查 collection 的基本統計"""
    print("=== Collection Statistics ===")
    try:
        info = qdrant.get_collection(QDRANT_COL)
        print(f"Collection: {QDRANT_COL}")
        print(f"Vector count: {info.points_count}")
        print(f"Vector size: {info.config.params.vectors.size}")
        print(f"Distance: {info.config.params.vectors.distance}")
        print()
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return

def debug_payload_distribution():
    """檢查不同 payload 欄位的分佈"""
    print("=== Payload Distribution ===")
    try:
        # 獲取一批點來分析 payload 分佈
        resp = qdrant.scroll(
            collection_name=QDRANT_COL,
            limit=100,  # 取樣 100 個點
            with_payload=True,
            with_vectors=False
        )
        
        doc_types = {}
        collections = {}
        titles = {}
        source_files = {}
        
        for point in resp[0]:  # resp 是 tuple (points, next_page_offset)
            payload = point.payload or {}
            
            # 統計 doc_type
            dt = payload.get("doc_type", "unknown")
            doc_types[dt] = doc_types.get(dt, 0) + 1
            
            # 統計 collection
            coll = payload.get("collection", "unknown")
            collections[coll] = collections.get(coll, 0) + 1
            
            # 統計 title
            title = payload.get("title", "unknown")
            titles[title] = titles.get(title, 0) + 1
            
            # 統計 source_file 路徑
            sf = payload.get("source_file", "unknown")
            if sf != "unknown":
                # 取得路徑的第一層目錄
                parts = sf.split("/")
                if len(parts) > 1:
                    folder = parts[1] if parts[0] == "rag" else parts[0]
                else:
                    folder = sf
                source_files[folder] = source_files.get(folder, 0) + 1
        
        print("Doc Types:")
        for dt, count in sorted(doc_types.items()):
            print(f"  {dt}: {count}")
        print()
        
        print("Collections:")
        for coll, count in sorted(collections.items()):
            print(f"  {coll}: {count}")
        print()
        
        print("Source Folders:")
        for folder, count in sorted(source_files.items()):
            print(f"  {folder}: {count}")
        print()
        
        print("Titles (top 10):")
        sorted_titles = sorted(titles.items(), key=lambda x: x[1], reverse=True)[:10]
        for title, count in sorted_titles:
            print(f"  {title}: {count}")
        print()
        
    except Exception as e:
        print(f"Error analyzing payloads: {e}")

def encode_query(q: str) -> List[float]:
    return embedder.encode([q], normalize_embeddings=True)[0].tolist()

def debug_query_results(question: str, topk: int=10):
    """詳細分析查詢結果"""
    print(f"=== Query Results for: '{question}' ===")
    
    try:
        vec = encode_query(question)
        print(f"Query vector dimension: {len(vec)}")
        print(f"Query vector sample: {vec[:5]}...")
        print()
        
        resp = qdrant.query_points(
            collection_name=QDRANT_COL,
            query=vec,
            limit=topk,
            with_payload=True,
            with_vectors=False,
            query_filter=None,
        )
        
        print(f"Found {len(resp.points)} results:")
        print()
        
        source_folders = {}
        for i, point in enumerate(resp.points):
            payload = point.payload or {}
            score = point.score
            title = payload.get("title", "No title")
            doc_type = payload.get("doc_type", "unknown")
            source_file = payload.get("source_file", "unknown")
            chunk_id = payload.get("chunk_id", "unknown")
            
            # 統計來源資料夾
            if source_file != "unknown":
                parts = source_file.split("/")
                folder = parts[1] if len(parts) > 1 and parts[0] == "rag" else parts[0]
                source_folders[folder] = source_folders.get(folder, 0) + 1
            
            print(f"#{i+1} [Score: {score:.4f}]")
            print(f"  Title: {title}")
            print(f"  Doc Type: {doc_type}")
            print(f"  Source: {source_file}")
            print(f"  Chunk ID: {chunk_id}")
            
            # 顯示文字內容的前 100 字
            text = payload.get("text", "")
            if text:
                preview = text[:100].replace("\n", " ")
                print(f"  Text Preview: {preview}...")
            print()
        
        print("Source folder distribution in results:")
        for folder, count in sorted(source_folders.items()):
            print(f"  {folder}: {count}")
        print()
        
    except Exception as e:
        print(f"Error querying: {e}")

def debug_specific_folders():
    """檢查特定資料夾的文檔是否被正確索引"""
    print("=== Checking specific folders ===")
    
    folders_to_check = ["code_snippets", "data_summaries", "manifests", "manuals", "papers"]
    
    for folder in folders_to_check:
        print(f"\nChecking folder: {folder}")
        try:
            # 使用 scroll 來查找特定資料夾的文檔
            resp = qdrant.scroll(
                collection_name=QDRANT_COL,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchText(text=folder)
                        )
                    ]
                ),
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            
            points = resp[0]
            print(f"  Found {len(points)} documents")
            
            for point in points[:2]:  # 只顯示前 2 個
                payload = point.payload or {}
                title = payload.get("title", "No title")
                source = payload.get("source_file", "unknown")
                print(f"    - {title} ({source})")
                
        except Exception as e:
            print(f"  Error: {e}")

def test_different_queries():
    """測試不同類型的查詢"""
    test_queries = [
        "海洋生物是否受海洋熱浪的影響",  # 你的原始查詢
        "marine heatwave biological impact",  # 英文版
        "code example",  # 測試是否能找到程式碼
        "data summary",  # 測試是否能找到資料摘要
        "manual guide",  # 測試是否能找到手冊
        "paper research",  # 測試是否能找到論文
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        debug_query_results(query, topk=5)

if __name__ == "__main__":
    print("Qdrant 查詢調試工具")
    print("="*50)
    
    # 1. 檢查 collection 基本統計
    debug_collection_stats()
    
    # 2. 檢查 payload 分佈
    debug_payload_distribution()
    
    # 3. 檢查特定資料夾是否被索引
    debug_specific_folders()
    
    # 4. 測試不同查詢
    test_different_queries()