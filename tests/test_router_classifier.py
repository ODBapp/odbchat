import pytest
from fastmcp.exceptions import ToolError

from server import router_classifier as router


class DummyResult:
    def __init__(self, mode: str = "explain", text: str = ""):
        self.mode = mode
        self.text = text
        self.code = None
        self.plan = {"steps": []}
        self.citations = []
        self.debug = None


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

    def fake_chat(*args, **kwargs):
        return (
            '{"decision":"mcp_tools","tool":{"name":"ghrsst.point_value",'
            '"arguments":{"longitude":123,"latitude":23,"date":"2025-01-10","method":"nearest"}},'
            '"confidence":0.9}'
        )

    monkeypatch.setattr(router, "_ghrsst_point_value", fake_point)
    monkeypatch.setattr(router, "_ghrsst_bbox_mean", fake_bbox)
    monkeypatch.setattr(router.LLM, "chat", fake_chat)
    monkeypatch.setattr(router, "run_onepass", lambda *a, **k: DummyResult())

    result = await router.classify_and_route("test query", debug=True)

    assert result["mode"] == "mcp_tools"
    assert result["tool"] == "ghrsst.point_value"
    assert result["arguments"]["longitude"] == 123.0
    assert result["arguments"]["latitude"] == 23.0
    assert result["arguments"]["method"] == "nearest"
    assert "result" in result and result["result"]["sst"] == 26.4


@pytest.mark.asyncio
async def test_router_explain(monkeypatch):
    def fake_chat(*args, **kwargs):
        return "{\"decision\":\"explain\",\"confidence\":0.55}"

    def fake_run_onepass(query, k, collections, temperature, debug):
        return DummyResult(text="Marine heatwaves explained")

    monkeypatch.setattr(router.LLM, "chat", fake_chat)
    monkeypatch.setattr(router, "run_onepass", fake_run_onepass)

    result = await router.classify_and_route("什麼是海洋熱浪?", debug=False)

    assert result["mode"] == "explain"
    assert result["text"] == "Marine heatwaves explained"
    assert result.get("confidence") == 0.55


@pytest.mark.asyncio
async def test_router_mcp_error(monkeypatch):
    def fake_chat(*args, **kwargs):
        return (
            '{"decision":"mcp_tools","tool":{"name":"ghrsst.point_value",'
            '"arguments":{"longitude":123,"latitude":23,"date":"2025-01-10","method":"nearest"}},'
            '"confidence":0.5}'
        )

    async def fake_point(lon, lat, date=None, fields=None, method=None):
        raise ToolError(
            "Error calling tool 'ghrsst.point_value': Data not exist for requested period"
        )

    monkeypatch.setattr(router.LLM, "chat", fake_chat)
    monkeypatch.setattr(router, "_ghrsst_point_value", fake_point)
    monkeypatch.setattr(router, "_ghrsst_bbox_mean", lambda **_: None)
    monkeypatch.setattr(router, "run_onepass", lambda *a, **k: DummyResult())

    result = await router.classify_and_route("今天台灣外海(23N, 123E)海溫多少？", debug=False)

    assert result["mode"] == "mcp_tools"
    assert "無法取得資料" in result["text"]
    assert result["arguments"]["method"] == "nearest"
    assert "error" in result


@pytest.mark.asyncio
async def test_router_mcp_fallback_to_nearest(monkeypatch):
    def fake_chat(*args, **kwargs):
        return (
            '{"decision":"mcp_tools","tool":{"name":"ghrsst.point_value",'
            '"arguments":{"longitude":123,"latitude":23,"date":"2025-01-10","method":"exact"}},'
            '"confidence":0.6}'
        )

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

    monkeypatch.setattr(router.LLM, "chat", fake_chat)
    monkeypatch.setattr(router, "_ghrsst_point_value", fake_point)
    monkeypatch.setattr(router, "_ghrsst_bbox_mean", lambda **_: None)
    monkeypatch.setattr(router, "run_onepass", lambda *a, **k: DummyResult())

    result = await router.classify_and_route("今天台灣外海(23N, 123E)海溫多少？", debug=False)

    assert call_methods == ["exact", "nearest"]
    assert result["arguments"]["method"] == "nearest"
    assert result.get("resolved_date") == "2025-01-09"
    assert "原請求" in result["text"]
