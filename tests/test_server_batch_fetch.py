import pytest
from shared.schemas import canonicalize_fetch
from types import SimpleNamespace

# Simulate server-side helper returning rows
def fake_fetch_impl(**r):
    r = canonicalize_fetch(r)
    # return 2 fake rows per request
    rows = [
        {"lon": r["lon0"], "lat": r["lat0"], "date": r["start"], "sst": 25.0, "sst_anomaly": 1.0},
        {"lon": (r["lon1"] or r["lon0"]), "lat": (r["lat1"] or r["lat0"]), "date": r["start"], "sst": 26.0, "sst_anomaly": 0.5},
    ]
    return {"bbox":[r["lon0"],r["lat0"],r["lon1"],r["lat1"]], "period":[r["start"],r["end"]], "data": rows}

def test_batch_merge_contract(monkeypatch):
    # monkeypatch your api.mhw_mcp._mhw_fetch_impl
    import importlib
    m = importlib.import_module("api.mhw_mcp")
    monkeypatch.setattr(m, "_mhw_fetch_impl", fake_fetch_impl, raising=False)

    # call the batch function
    res = m._mhw_fetch_batch_impl(requests=[
        {"lon0":119,"lat0":20,"lon1":122,"lat1":23,"start":"2024-07-01","append":"sst,sst_anomaly","output":"json","include_data":True},
        {"lon0":119,"lat0":20,"lon1":122,"lat1":23,"start":"2024-08-01","append":"sst,sst_anomaly","output":"json","include_data":True},
    ])
    assert "data" in res and "components" in res
    assert len(res["data"]) == 4
    assert len(res["components"]) == 2
