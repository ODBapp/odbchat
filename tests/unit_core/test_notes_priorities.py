from server.rag.onepass_core import Hit, _compress_notes


def test_compress_notes_non_code_keeps_top_score_order():
    hits = [
        Hit(
            id="doc1",
            score=1.2,
            title="Top Doc",
            doc_type="image_description",
            source_file="odb_open_apis01.png",
            chunk_id=0,
            text="ODB APIs: MHW, GHRSST, Tide",
            payload={"doc_type": "image_description", "title": "Top Doc"},
        ),
        Hit(
            id="doc2",
            score=1.1,
            title="API Spec",
            doc_type="api_spec",
            source_file="odb_mhw_openapi.yml",
            chunk_id=0,
            text="openapi: 3.0",
            payload={"doc_type": "api_spec", "title": "API Spec"},
        ),
    ]

    notes = _compress_notes(hits, max_docs=2, query="ODB 有哪些 API")
    assert notes.splitlines()[0].startswith("[image_description]")
