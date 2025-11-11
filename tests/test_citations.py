import pytest

from server.rag.onepass_core import (
    Hit,
    format_citations,
    _query_is_meaningful,
    _wants_code,
    _looks_like_followup,
)


def _make_hit(idx: int, text: str, title: str = "Doc") -> Hit:
    return Hit(
        id=str(idx),
        score=1.0 / (idx + 1),
        title=title,
        doc_type="web_article",
        source_file=f"https://example.com/{idx}",
        chunk_id=idx,
        text=text,
        payload={"title": title, "canonical_url": f"https://example.com/{idx}"},
    )


def test_format_citations_prefers_relevant_hits():
    hits = [
        _make_hit(0, "Hidy Viewer 提供 GUI 互動界面與時間選擇", title="Marine Heatwaves (ODB)"),
        _make_hit(1, "ENSO 定義與氣壓變化", title="ENSO"),
    ]
    cites = format_citations(hits, answer_text="GUI 互動界面 Hidy Viewer", limit=2)
    assert len(cites) == 1
    assert "Marine" in cites[0].title


def test_format_citations_limits_when_no_overlap():
    hits = [_make_hit(i, f"Snippet {i}", title=f"Doc{i}") for i in range(4)]
    cites = format_citations(hits, answer_text="", limit=2)
    assert len(cites) == 2


def test_query_meaningfulness_checks():
    assert not _query_is_meaningful("....")
    assert _query_is_meaningful("何謂 ENSO")


def test_wants_code_keywords():
    assert _wants_code("請給我Python程式碼")
    assert not _wants_code("是否有GUI界面？")


def test_followup_detection_heuristic():
    prev = {"endpoint": "/api/mhw", "params": {"start": "2024-01-01"}}
    assert _looks_like_followup("改畫2024-01 海洋熱浪地圖", prev)
    assert not _looks_like_followup("請解釋 ENSO", prev)
