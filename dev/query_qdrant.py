#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
改進的查詢策略：結合多種文檔類型的搜尋
"""

import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# 設定
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COL = os.environ.get("QDRANT_COL", "odb_mhw_knowledge_v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "thenlper/gte-small")

# 初始化
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def encode_query(q: str) -> List[float]:
    return embedder.encode([q], normalize_embeddings=True)[0].tolist()

def query_by_doc_type(question: str, doc_type: str, topk: int = 3) -> List[Dict[str, Any]]:
    """根據文檔類型查詢"""
    vec = encode_query(question)
    
    resp = qdrant.query_points(
        collection_name=QDRANT_COL,
        query=vec,
        limit=topk,
        with_payload=True,
        with_vectors=False,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchValue(value=doc_type)
                )
            ]
        )
    )
    
    return [
        {
            "score": p.score,
            "title": p.payload.get("title", ""),
            "doc_type": p.payload.get("doc_type", ""),
            "source_file": p.payload.get("source_file", ""),
            "text": p.payload.get("text", "")[:200] + "..." if len(p.payload.get("text", "")) > 200 else p.payload.get("text", "")
        }
        for p in resp.points
    ]

def query_qdrant_improved(question: str, topk: int = 5, include_all_types: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """
    改進的查詢策略：
    1. 先做全域查詢
    2. 如果需要，針對不同文檔類型分別查詢
    """
    
    results = {}
    
    # 1. 全域查詢（你原來的方法）
    vec = encode_query(question)
    resp = qdrant.query_points(
        collection_name=QDRANT_COL,
        query=vec,
        limit=max(8, topk*2),
        with_payload=True,
        with_vectors=False,
        query_filter=None,
    )
    
    results["all"] = [
        {
            "score": p.score,
            "title": p.payload.get("title", ""),
            "doc_type": p.payload.get("doc_type", ""),
            "source_file": p.payload.get("source_file", ""),
            "text": p.payload.get("text", "")[:200] + "..." if len(p.payload.get("text", "")) > 200 else p.payload.get("text", "")
        }
        for p in resp.points[:topk]
    ]
    
    # 2. 如果需要多樣性，分類別查詢
    if include_all_types:
        doc_types = ["web_article", "paper_note", "code_snippet", "cli_tool_guide", "api_spec"]
        
        for doc_type in doc_types:
            type_results = query_by_doc_type(question, doc_type, topk=2)
            if type_results:
                results[doc_type] = type_results
    
    return results

def smart_query(question: str, strategy: str = "adaptive") -> Dict[str, Any]:
    """
    智能查詢策略：根據問題類型選擇不同的查詢方法
    """
    
    # 分析問題類型
    question_lower = question.lower()
    question_keywords = {
        "code": ["code", "程式", "代碼", "範例", "example", "script", "python"],
        "manual": ["manual", "guide", "使用", "操作", "指南", "教學"],
        "paper": ["paper", "research", "study", "論文", "研究", "學術"],
        "api": ["api", "接口", "規格", "spec", "endpoint"]
    }
    
    detected_type = None
    for qtype, keywords in question_keywords.items():
        if any(kw in question_lower for kw in keywords):
            detected_type = qtype
            break
    
    results = {"question_type": detected_type, "results": {}}
    
    if strategy == "adaptive" and detected_type:
        # 根據檢測到的問題類型，優先查詢對應的文檔類型
        if detected_type == "code":
            results["results"]["code_snippet"] = query_by_doc_type(question, "code_snippet", 3)
            results["results"]["all"] = query_qdrant_improved(question, topk=3)["all"]
        elif detected_type == "manual":
            results["results"]["cli_tool_guide"] = query_by_doc_type(question, "cli_tool_guide", 3)
            results["results"]["all"] = query_qdrant_improved(question, topk=3)["all"]
        elif detected_type == "paper":
            results["results"]["paper_note"] = query_by_doc_type(question, "paper_note", 3)
            results["results"]["all"] = query_qdrant_improved(question, topk=3)["all"]
        elif detected_type == "api":
            results["results"]["api_spec"] = query_by_doc_type(question, "api_spec", 3)
            results["results"]["all"] = query_qdrant_improved(question, topk=3)["all"]
    else:
        # 多樣化查詢：從每種類型都取一些
        results["results"] = query_qdrant_improved(question, topk=5, include_all_types=True)
    
    return results

def test_improved_queries():
    """測試改進的查詢方法"""
    
    test_questions = [
        "海洋生物是否受海洋熱浪的影響",  # 原始問題
        "如何使用 ODB API 獲取海溫數據",    # API/程式碼相關
        "有沒有海洋熱浪的研究論文",         # 論文相關
        "CLI 工具怎麼使用",               # 手冊相關
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        
        # 使用智能查詢
        results = smart_query(question, strategy="adaptive")
        print(f"Detected question type: {results['question_type']}")
        
        for category, docs in results["results"].items():
            print(f"\n--- {category.upper()} ---")
            for i, doc in enumerate(docs[:3], 1):
                print(f"{i}. [{doc['score']:.4f}] {doc['title']}")
                print(f"   Type: {doc['doc_type']}, Source: {doc.get('source_file', 'unknown')}")

def fix_source_file_paths():
    """
    修復 source_file 路徑問題的腳本
    注意：這會修改現有的數據，建議先備份
    """
    print("=== Fixing Source File Paths ===")
    
    # 先獲取所有點
    all_points = []
    offset = None
    
    while True:
        resp = qdrant.scroll(
            collection_name=QDRANT_COL,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )
        
        points, next_offset = resp
        all_points.extend(points)
        
        if next_offset is None:
            break
        offset = next_offset
    
    print(f"Total points to check: {len(all_points)}")
    
    # 找出需要修復的點
    points_to_update = []
    
    for point in all_points:
        payload = point.payload or {}
        source_file = payload.get("source_file", "")
        
        # 檢查是否需要修復
        needs_fix = False
        new_source_file = source_file
        
        if source_file.startswith("https://"):
            # 這些應該來自 papers 或 code_snippets
            if "github.com" in source_file:
                new_source_file = f"code_snippets/{source_file.split('/')[-1]}"
            elif "doi.org" in source_file:
                new_source_file = f"papers/{source_file.split('/')[-1]}"
            needs_fix = True
        elif source_file.startswith("rag\\"):
            # Windows 路徑問題
            new_source_file = source_file.replace("\\", "/")
            needs_fix = True
        
        if needs_fix:
            new_payload = dict(payload)
            new_payload["source_file"] = new_source_file
            
            points_to_update.append({
                "id": point.id,
                "vector": point.vector,
                "payload": new_payload
            })
    
    print(f"Found {len(points_to_update)} points that need fixing")
    
    # 這裡只是打印，不實際更新（避免意外）
    if points_to_update:
        print("Sample fixes:")
        for fix in points_to_update[:5]:
            old_source = fix["payload"].get("source_file", "")
            # 從原始點找舊路徑
            original = next(p for p in all_points if str(p.id) == str(fix["id"]))
            old_source = original.payload.get("source_file", "")
            print(f"  {old_source} -> {fix['payload']['source_file']}")
        
        print("\n如果要實際執行修復，請取消下面的註解：")
        print("# qdrant.upsert(collection_name=QDRANT_COL, points=[")
        print("#     PointStruct(id=fix['id'], vector=fix['vector'], payload=fix['payload'])")
        print("#     for fix in points_to_update")
        print("# ])")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-queries", action="store_true", help="測試改進的查詢方法")
    parser.add_argument("--fix-paths", action="store_true", help="修復 source_file 路徑問題")
    
    args = parser.parse_args()
    
    if args.test_queries:
        test_improved_queries()
    elif args.fix_paths:
        fix_source_file_paths()
    else:
        print("請使用 --test-queries 或 --fix-paths 參數")