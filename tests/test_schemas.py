import pytest
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
