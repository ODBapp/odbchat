import pytest
from fastmcp.exceptions import ToolError

from server.api import ghrsst_mcp_proxy as proxy


@pytest.mark.asyncio
async def test_point_value_forwards_arguments(monkeypatch):
    recorded: dict[str, object] = {}

    async def fake_call(tool_name: str, payload: dict):
        recorded["tool_name"] = tool_name
        recorded["payload"] = payload
        return {"result": "ok"}

    monkeypatch.setattr(proxy, "_call_ghrsst_tool", fake_call)

    result = await proxy._ghrsst_point_value(
        longitude=121.5,
        latitude=23.7,
        date="2025-01-01",
        fields=["sst", "sst_anomaly"],
        method="nearest",
    )

    assert result == {"result": "ok"}
    assert recorded["tool_name"] == "ghrsst.point_value"
    assert recorded["payload"] == {
        "longitude": 121.5,
        "latitude": 23.7,
        "date": "2025-01-01",
        "fields": ["sst", "sst_anomaly"],
        "method": "nearest",
    }


@pytest.mark.asyncio
async def test_bbox_mean_requires_four_coordinates():
    with pytest.raises(ToolError):
        await proxy._ghrsst_bbox_mean(bbox=[119, 21, 123], date="2025-01-01")


@pytest.mark.asyncio
async def test_bbox_mean_forwards_arguments(monkeypatch):
    recorded: dict[str, object] = {}

    async def fake_call(tool_name: str, payload: dict):
        recorded["tool_name"] = tool_name
        recorded["payload"] = payload
        return {"mean": 26.5}

    monkeypatch.setattr(proxy, "_call_ghrsst_tool", fake_call)

    result = await proxy._ghrsst_bbox_mean(
        bbox=[119.0, 21.0, 123.0, 26.0],
        date="2025-01-01",
        fields=["sst"],
        method="nearest",
    )

    assert result == {"mean": 26.5}
    assert recorded["tool_name"] == "ghrsst.bbox_mean"
    assert recorded["payload"] == {
        "bbox": [119.0, 21.0, 123.0, 26.0],
        "date": "2025-01-01",
        "fields": ["sst"],
        "method": "nearest",
    }
