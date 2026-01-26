from ingest import ingest_json


def test_payload_promotes_metadata_and_infers_dataset():
    artifact = {
        "id": "a1",
        "source_file": "bak/temp_uploads/woa23documentation.pdf",
        "artifact_type": "text_chunk",
        "links": [],
        "tags": ["woa"],
        "text": "example text",
        "metadata": {
            "doc_type": "web_article",
            "dataset_name": "WOA23",
            "title": "WOA23 Documentation",
            "tags": ["temperature"],
            "lang": "en",
        },
    }
    payload = ingest_json._payload_from_artifact(artifact, "AI")

    assert payload["artifact_type"] == "text_chunk"
    assert payload["doc_type"] == "web_article"
    assert payload["dataset_name"] == "WOA23"
    assert payload["title"] == "WOA23 Documentation"
    assert payload["lang"] == "en"
    assert "woa" in payload["tags"]
    assert "temperature" in payload["tags"]
    assert payload["payload_core"]["doc_type"] == "web_article"
    assert payload["payload_core"]["dataset_name"] == "WOA23"


def test_payload_infers_dataset_when_missing():
    artifact = {
        "id": "a2",
        "source_file": "bak/temp_uploads/mhw_odb_severity_zh.yml",
        "artifact_type": "text_chunk",
        "links": [],
        "tags": [],
        "text": "example text",
        "metadata": {},
    }
    payload = ingest_json._payload_from_artifact(artifact, "AI")

    assert payload["artifact_type"] == "text_chunk"
    assert payload["doc_type"] == "text_chunk"
    assert payload.get("dataset_name") == "mhw_odb_severity_zh"


def test_payload_includes_table_label():
    artifact = {
        "id": "t1",
        "source_file": "bak/temp_uploads/woa23documentation.pdf",
        "artifact_type": "table",
        "links": [],
        "tags": [],
        "caption": "Table 4. Depth ranges",
        "table_label": "Table 4",
        "markdown_content": "| A | B |",
        "metadata": {"dataset_name": "WOA23"},
    }
    payload = ingest_json._payload_from_artifact(artifact, "AI")

    assert payload["doc_type"] == "table"
    assert payload["table_label"] == "Table 4"
    assert payload["payload_core"]["table_label"] == "Table 4"


def test_text_for_artifact_includes_image_metadata():
    artifact = {
        "id": "img1",
        "source_file": "bak/temp_uploads/odb_open_apis01.png",
        "artifact_type": "text_chunk",
        "text": "Ocean Data Bank API List",
        "metadata": {
            "raw_text": "ODB Open APIs",
            "extracted_items": [
                {"entity": "ODB CTD API", "details": "CTD data"},
                {"entity": "Tide API", "details": "TPXO9 model"},
            ],
        },
    }
    text = ingest_json.text_for_artifact(artifact)
    assert "Ocean Data Bank API List" in text
    assert "ODB CTD API: CTD data" in text
    assert "Tide API: TPXO9 model" in text
    assert "ODB Open APIs" in text


def test_payload_canonical_url_from_metadata_source_file():
    artifact = {
        "id": "a1",
        "source_file": "bak/temp_uploads/odb_mhw_openapi.yml",
        "artifact_type": "text_chunk",
        "text": "OpenAPI spec",
        "metadata": {
            "source_file": "https://api.odb.ntu.edu.tw/hub/swagger?node=odb_mhw_v1",
        },
    }
    payload = ingest_json._payload_from_artifact(artifact, "AI")
    assert payload["canonical_url"] == "https://api.odb.ntu.edu.tw/hub/swagger?node=odb_mhw_v1"


def test_payload_canonical_url_from_raw_text():
    artifact = {
        "id": "img1",
        "source_file": "bak/temp_uploads/odb_open_apis01.png",
        "artifact_type": "text_chunk",
        "text": "Ocean Data Bank API List",
        "metadata": {"raw_text": "ODB Open APIs https://api.odb.ntu.edu.tw/hub"},
    }
    payload = ingest_json._payload_from_artifact(artifact, "AI")
    assert payload["canonical_url"] == "https://api.odb.ntu.edu.tw/hub"
