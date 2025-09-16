import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

from shared.schemas import canonicalize_fetch, validate_fetch, canonicalize_plot, validate_plot

def test_fetch_schema_ok():
    req = {
        "lon0": 119.0, "lat0": 20.0, "lon1": 122.0, "lat1": 23.0,
        "start": "2024-07-01", "end": None,
        "append": "sst,sst_anomaly", "output": "json", "include_data": True
    }
    req = canonicalize_fetch(req)
    ok, err = validate_fetch(req)
    assert ok, err
    assert req["end"] == "2024-07-01"

def test_fetch_schema_bad_field():
    req = {
        "lon0": 119.0, "lat0": 20.0, "start": "2024-07-01",
        "append": "sst,bogus", "output": "json", "include_data": True
    }
    ok, err = validate_fetch(req)
    assert not ok
    assert "does not match" in err or "is not valid" in err

def test_plot_cfg_ok_series():
    cfg = {"mode":"series", "fields":["sst","sst_anomaly"]}
    cfg = canonicalize_plot(cfg)
    ok, err = validate_plot(cfg)
    assert ok, err

def test_plot_cfg_bad_mode():
    cfg = {"mode":"foo", "fields":["sst"]}
    ok, err = validate_plot(cfg)
    assert not ok


def _load_tool_schema(name: str):
    schema_path = Path(__file__).resolve().parents[1] / "specs" / "mcp.tools.schema.json"
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    return data["tools"][name]


def test_onepass_input_schema_validates():
    schema = _load_tool_schema("rag.onepass_answer.input")
    validator = Draft7Validator(schema)
    payload = {"query": "海洋熱浪要如何查詢?", "top_k": 5, "temperature": 0.1}
    errors = list(validator.iter_errors(payload))
    assert errors == []


def test_onepass_output_schema_rejects_missing_mode():
    schema = _load_tool_schema("rag.onepass_answer.output")
    validator = Draft7Validator(schema)
    payload = {"text": "ok", "citations": []}
    errors = list(validator.iter_errors(payload))
    assert errors, "mode should be required"
