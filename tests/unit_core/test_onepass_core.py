import pytest

from server.rag.onepass_core import Hit, validate_plan, enforce_static_guards, format_citations


def test_validate_plan_accepts_whitelisted_params():
    whitelist = {
        "paths": ["/api/mhw"],
        "params": ["lon0", "lat0", "start", "append", "output"],
        "append_allowed": ["sst", "sst_anomaly"],
        "param_enums": {"output": ["json", "csv"]},
    }
    plan = {
        "endpoint": "/api/mhw",
        "params": {"lon0": 120, "lat0": 23, "start": "2024-07", "append": "sst", "output": "json"},
    }
    validate_plan(plan, whitelist)
    assert plan["params"]["start"] == "2024-07-01"


def test_validate_plan_rejects_unknown_param():
    whitelist = {
        "paths": ["/api/mhw"],
        "params": ["lon0", "lat0"],
        "append_allowed": [],
        "param_enums": {},
    }
    plan = {"endpoint": "/api/mhw", "params": {"foo": 1}}
    with pytest.raises(ValueError):
        validate_plan(plan, whitelist)


def test_enforce_static_guards_blocks_query_concat():
    good = (
        "import requests\n"
        "import pandas as pd\n"
        "r = requests.get('https://eco.odb/api', params={'a': 1})\n"
        "df = pd.DataFrame(r.json())\n"
    )
    enforce_static_guards(good)
    bad = "requests.get('https://eco.odb/api?lon0=1')"
    with pytest.raises(ValueError):
        enforce_static_guards(bad)


def test_format_citations_includes_sources():
    payload = {
        "title": "Doc",
        "source_file": "rag/manuals/doc.yml",
        "canonical_url": "https://eco.odb/doc",
        "doc_type": "web_article",
    }
    hits = [
        Hit(id="1", score=0.5, title="Doc", doc_type="web_article", source_file="rag/manuals/doc.yml", chunk_id=0, text="A", payload=payload),
        Hit(id="2", score=0.4, title="Doc", doc_type="web_article", source_file="rag/manuals/doc.yml", chunk_id=1, text="B", payload=payload),
    ]
    cites = format_citations(hits)
    assert len(cites) == 2
    assert cites[0].source == "https://eco.odb/doc"
