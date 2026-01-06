from server.rag.onepass_core import Hit, _collect_api_endpoints_from_hits


def test_append_allowed_from_param_description():
    payload = {
        "doc_type": "api_endpoint",
        "path": "/api/mhw",
        "parameters": [
            {
                "name": "append",
                "description": "Allowed fields: 'sst', 'sst_anomaly', 'level'",
                "schema": {"type": "string"},
            }
        ],
    }
    hit = Hit(
        id="api1",
        score=0.9,
        title="api",
        doc_type="api_endpoint",
        source_file="odb_mhw_openapi.yml",
        chunk_id=0,
        text="",
        payload=payload,
    )
    whitelist = _collect_api_endpoints_from_hits([hit])
    assert "sst" in whitelist["append_allowed"]
    assert "sst_anomaly" in whitelist["append_allowed"]
