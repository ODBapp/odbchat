from server.rag.onepass_core import Hit, _looks_like_table_request, _pick_best_table_hit, run_onepass


def test_looks_like_table_request_handles_depth_variable():
    assert _looks_like_table_request("請問海洋學變數深度範圍")
    assert _looks_like_table_request("WOA23 Table 4 variables")


def test_pick_best_table_hit_prefers_table4_caption():
    query = "WOA23 表格4 海洋學變數 深度範圍"
    table1 = Hit(
        id="t1",
        score=0.8,
        title="table1",
        doc_type="table",
        source_file="woa23documentation.pdf",
        chunk_id=0,
        text="",
        payload={
            "caption": "Table 1. Time Spans for World Ocean Atlas 2023",
            "markdown_content": "| Time Span | Abbreviation |",
            "doc_type": "table",
        },
    )
    table4 = Hit(
        id="t4",
        score=0.7,
        title="table4",
        doc_type="table",
        source_file="woa23documentation.pdf",
        chunk_id=0,
        text="",
        payload={
            "caption": "Table 4. Depth ranges and standard depth levels numbers for annual, seasonal, and monthly statistics",
            "markdown_content": "| Oceanographic Variable | Depths for Annual |",
            "doc_type": "table",
        },
    )

    best = _pick_best_table_hit(query, [table1, table4])
    assert best is table4


def test_onepass_table_query_returns_table4_citation(monkeypatch):
    query = "WOA23 Table 4 variables"
    table1 = Hit(
        id="t1",
        score=0.8,
        title="Table 1",
        doc_type="table",
        source_file="woa23documentation.pdf",
        chunk_id=0,
        text="",
        payload={
            "title": "Table 1. Time Spans for World Ocean Atlas 2023",
            "caption": "Table 1. Time Spans for World Ocean Atlas 2023",
            "markdown_content": "| Time Span | Abbreviation |",
            "doc_type": "table",
        },
    )
    table4 = Hit(
        id="t4",
        score=0.7,
        title="Table 4",
        doc_type="table",
        source_file="woa23documentation.pdf",
        chunk_id=0,
        text="",
        payload={
            "title": "Table 4. Depth ranges and standard depth levels numbers for annual, seasonal, and monthly statistics",
            "caption": "Table 4. Depth ranges and standard depth levels numbers for annual, seasonal, and monthly statistics",
            "markdown_content": "| Oceanographic Variable | Depths for Annual |",
            "doc_type": "table",
        },
    )

    hits = [table1, table4]
    monkeypatch.setattr("server.rag.onepass_core.search_qdrant", lambda *args, **kwargs: hits)
    monkeypatch.setattr("server.rag.onepass_core._fetch_table_candidates", lambda *args, **kwargs: hits)
    monkeypatch.setattr(
        "server.rag.onepass_core.decide_and_generate",
        lambda **kwargs: {"mode": "explain", "answer": "See Table 4 for variables."},
    )

    result = run_onepass(query, k=2, debug=False)
    assert result.citations
    assert any("Table 4" in citation.title for citation in result.citations)


def test_expand_hits_via_links_falls_back_to_doc_id():
    from server.rag.onepass_core import expand_hits_via_links
    class _FakeClient:
        def __init__(self):
            self.calls = []
        def scroll(self, collection_name, scroll_filter, limit, with_payload, with_vectors):
            self.calls.append((collection_name, scroll_filter))
            # Simulate artifact_id miss, doc_id hit
            key = scroll_filter.must[0].key
            val = scroll_filter.must[0].match.value
            if key == "artifact_id":
                return ([], None)
            if key == "doc_id" and val == "target-1":
                return ([type("P", (), {"payload": {"doc_id": "target-1", "doc_type": "table", "source_file": "f", "title": "T", "links": []}})()], None)
            return ([], None)

    hit = Hit(
        id="h1",
        score=1.0,
        title="t",
        doc_type="table_card",
        source_file="f",
        chunk_id=0,
        text="",
        payload={
            "collection": "AI",
            "artifact_id": "source-1",
            "links": [{"target_id": "target-1", "link_type": "references"}],
        },
    )
    expanded = expand_hits_via_links([hit], _FakeClient(), limit=1)
    assert expanded


def test_rerank_diversify_allows_multiple_tables_same_file():
    from server.rag.onepass_core import rerank_diversify_hits

    table1 = Hit(
        id="t1",
        score=1.0,
        title="Table 1",
        doc_type="table",
        source_file="woa23documentation.pdf",
        chunk_id=0,
        text="",
        payload={"table_label": "Table 1", "caption": "Table 1"},
    )
    table4 = Hit(
        id="t4",
        score=0.9,
        title="Table 4",
        doc_type="table",
        source_file="woa23documentation.pdf",
        chunk_id=0,
        text="",
        payload={"table_label": "Table 4", "caption": "Table 4"},
    )
    out = rerank_diversify_hits([table1, table4], k=6)
    labels = {h.payload.get("table_label") for h in out}
    assert "Table 1" in labels
    assert "Table 4" in labels
