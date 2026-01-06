from server.rag.onepass_core import Hit, _looks_like_table_request, _pick_best_table_hit


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
