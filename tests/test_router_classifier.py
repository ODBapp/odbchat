from typing import Optional

import pytest
from fastmcp.exceptions import ToolError

from server import router_classifier as router


class DummyResult:
    def __init__(self, mode: str = "explain", text: str = "", mcp: Optional[dict] = None):
        self.mode = mode
        self.text = text
        self.code = None
        self.plan = {"steps": []}
        self.citations = []
        self.debug = None
        self.mcp = mcp


@pytest.mark.asyncio
async def test_router_mcp_point(monkeypatch):
    async def fake_point(lon, lat, date=None, fields=None, method=None):
        assert method == "nearest"
        return {
            "region": [lon, lat],
            "date": date,
            "sst": 26.4,
            "sst_anomaly": 0.12,
        }

    async def fake_bbox(**kwargs):  # pragma: no cover
        raise AssertionError("bbox tool should not be called")

    monkeypatch.setattr(router, "_ghrsst_point_value", fake_point)
    monkeypatch.setattr(router, "_ghrsst_bbox_mean", fake_bbox)
    spec = {
        "tool": "ghrsst.point_value",
        "arguments": {
            "longitude": 123,
            "latitude": 23,
            "date": "2025-01-10",
            "method": "nearest",
        },
        "confidence": 0.9,
    }
    dummy = DummyResult(mode="mcp_tools", mcp=spec)

    async def fake_onepass(query, debug, today):
        return dummy, {"mode": "mcp_tools", "mcp": spec, "citations": []}

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)

    result = await router.classify_and_route("test query", debug=True)

    assert result["mode"] == "mcp_tools"
    assert result["tool"] == "ghrsst.point_value"
    assert result["arguments"]["longitude"] == 123.0
    assert result["arguments"]["latitude"] == 23.0
    assert result["arguments"]["method"] == "nearest"
    assert "result" in result and result["result"]["sst"] == 26.4


@pytest.mark.asyncio
async def test_router_explain(monkeypatch):
    dummy = DummyResult(mode="explain", text="Marine heatwaves explained")

    async def fake_onepass(query, debug, today):
        payload = {
            "mode": "explain",
            "text": "Marine heatwaves explained",
            "citations": [{"title": "Doc", "source": "https://example", "chunk_id": 1}],
        }
        return dummy, payload

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)

    result = await router.classify_and_route("什麼是海洋熱浪?", debug=False)

    assert result["mode"] == "explain"
    assert result["text"] == "Marine heatwaves explained"


@pytest.mark.asyncio
async def test_router_mcp_error(monkeypatch):
    async def fake_point(lon, lat, date=None, fields=None, method=None):
        raise ToolError(
            "Error calling tool 'ghrsst.point_value': Data not exist for requested period"
        )

    monkeypatch.setattr(router, "_ghrsst_point_value", fake_point)
    monkeypatch.setattr(router, "_ghrsst_bbox_mean", lambda **_: None)
    spec = {
        "tool": "ghrsst.point_value",
        "arguments": {
            "longitude": 123,
            "latitude": 23,
            "date": "2025-01-10",
            "method": "nearest",
        },
        "confidence": 0.5,
    }
    dummy = DummyResult(mode="mcp_tools", mcp=spec)

    async def fake_onepass(query, debug, today):
        return dummy, {"mode": "mcp_tools", "mcp": spec, "citations": []}

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)

    result = await router.classify_and_route("今天台灣外海(23N, 123E)海溫多少？", debug=False)

    assert result["mode"] == "mcp_tools"
    assert "無法取得資料" in result["text"]
    assert result["arguments"]["method"] == "nearest"
    assert "error" in result


@pytest.mark.asyncio
async def test_router_mcp_fallback_to_nearest(monkeypatch):
    call_methods: list[str | None] = []

    async def fake_point(lon, lat, date=None, fields=None, method=None):
        call_methods.append(method)
        if method == "exact":
            raise ToolError(
                "Error calling tool 'ghrsst.point_value': Data not exist for requested period"
            )
        return {
            "region": [lon, lat],
            "date": "2025-01-09",
            "sst": 26.0,
            "sst_anomaly": 0.1,
        }

    monkeypatch.setattr(router, "_ghrsst_point_value", fake_point)
    monkeypatch.setattr(router, "_ghrsst_bbox_mean", lambda **_: None)
    spec = {
        "tool": "ghrsst.point_value",
        "arguments": {
            "longitude": 123,
            "latitude": 23,
            "date": "2025-01-10",
            "method": "exact",
        },
        "confidence": 0.6,
    }
    dummy = DummyResult(mode="mcp_tools", mcp=spec)

    async def fake_onepass(query, debug, today):
        return dummy, {"mode": "mcp_tools", "mcp": spec, "citations": []}

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)

    result = await router.classify_and_route("今天台灣外海(23N, 123E)海溫多少？", debug=False)

    assert call_methods == ["exact", "nearest"]
    assert result["arguments"]["method"] == "nearest"
    assert result.get("resolved_date") == "2025-01-09"
    assert "原請求" in result["text"]


@pytest.mark.asyncio
async def test_router_auto_point_fallback(monkeypatch):
    async def fake_onepass(query, debug, today):
        return DummyResult(mode="explain", text="N/A"), {"mode": "explain", "text": "N/A", "citations": []}

    async def fake_point(lon, lat, date=None, fields=None, method=None):
        assert (lon, lat) == (123.0, 25.0)
        return {"region": [lon, lat], "date": date, "sst": 27.1, "sst_anomaly": 0.2}

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_ghrsst_point_value", fake_point)
    monkeypatch.setattr(router, "_ghrsst_bbox_mean", lambda **_: None)
    monkeypatch.setattr(router, "today_tpe", lambda: "2025-01-10")

    result = await router.classify_and_route("現在台灣周遭(123,25)海溫多少？", debug=False)

    assert result["mode"] == "mcp_tools"
    assert result["tool"] == "ghrsst.point_value"
    assert result["arguments"]["longitude"] == 123.0
    assert result["arguments"]["latitude"] == 25.0
    assert result["result"]["sst"] == 27.1


@pytest.mark.asyncio
async def test_router_mcp_bbox_dict_arguments(monkeypatch):
    spec = {
        "tool": "ghrsst.bbox_mean",
        "arguments": {
            "bbox": {"lon0": 118, "lat0": 23, "lon1": 123, "lat1": 30},
            "date": "2025-01-10",
            "method": "nearest",
        },
        "confidence": 0.8,
    }
    dummy = DummyResult(mode="mcp_tools", mcp=spec)

    async def fake_onepass(query, debug, today):
        return dummy, {"mode": "mcp_tools", "mcp": spec, "citations": []}

    async def fake_bbox(bbox, date=None, fields=None, method=None):
        assert bbox == [118.0, 23.0, 123.0, 30.0]
        return {"region": bbox, "date": date, "sst": 26.5, "sst_anomaly": -0.1}

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_ghrsst_bbox_mean", fake_bbox)
    monkeypatch.setattr(router, "_ghrsst_point_value", lambda **_: None)

    result = await router.classify_and_route("bbox dict test", debug=False)

    assert result["mode"] == "mcp_tools"
    assert result["result"]["sst"] == 26.5
    assert result["arguments"]["bbox"] == [118.0, 23.0, 123.0, 30.0]
