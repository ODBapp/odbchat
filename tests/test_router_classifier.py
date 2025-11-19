from typing import Optional

import pytest
from fastmcp.exceptions import ToolError

from server import router_classifier as router

TEST_TZ = "Asia/Taipei"
TEST_QUERY_TIME = "2025-01-10T00:00:00+08:00"


@pytest.fixture(autouse=True)
def _patch_time_utils(monkeypatch):
    monkeypatch.setattr(router, "today_in_tz", lambda tz=None: "2025-01-10")
    monkeypatch.setattr(router, "now_iso_in_tz", lambda tz=None: TEST_QUERY_TIME)


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

    async def fake_onepass(query, debug, today, tz, query_time):
        return dummy, {"mode": "mcp_tools", "mcp": spec, "citations": []}

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)

    result = await router.classify_and_route("test query", tz=TEST_TZ, query_time=TEST_QUERY_TIME, debug=True)

    assert result["mode"] == "mcp_tools"
    assert result["tool"] == "ghrsst.point_value"
    assert result["arguments"]["longitude"] == 123.0
    assert result["arguments"]["latitude"] == 23.0
    assert result["arguments"]["method"] == "nearest"
    assert "result" in result and result["result"]["sst"] == 26.4


@pytest.mark.asyncio
async def test_router_explain(monkeypatch):
    dummy = DummyResult(mode="explain", text="Marine heatwaves explained")

    async def fake_onepass(query, debug, today, tz, query_time):
        payload = {
            "mode": "explain",
            "text": "Marine heatwaves explained",
            "citations": [{"title": "Doc", "source": "https://example", "chunk_id": 1}],
        }
        return dummy, payload

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)

    result = await router.classify_and_route("什麼是海洋熱浪?", tz=TEST_TZ, query_time=TEST_QUERY_TIME, debug=False)

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

    async def fake_onepass(query, debug, today, tz, query_time):
        return dummy, {"mode": "mcp_tools", "mcp": spec, "citations": []}

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)

    result = await router.classify_and_route(
        "今天台灣外海(23N, 123E)海溫多少？",
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

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

    async def fake_onepass(query, debug, today, tz, query_time):
        return dummy, {"mode": "mcp_tools", "mcp": spec, "citations": []}

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)

    result = await router.classify_and_route(
        "今天台灣外海(23N, 123E)海溫多少？",
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert call_methods == ["exact", "nearest"]
    assert result["arguments"]["method"] == "nearest"
    assert result.get("resolved_date") == "2025-01-09"
    assert "原請求" in result["text"]


@pytest.mark.asyncio
async def test_router_auto_point_fallback(monkeypatch):
    async def fake_onepass(query, debug, today, tz, query_time):
        return DummyResult(mode="explain", text="N/A"), {"mode": "explain", "text": "N/A", "citations": []}

    async def fake_point(lon, lat, date=None, fields=None, method=None):
        assert (lon, lat) == (123.0, 25.0)
        return {"region": [lon, lat], "date": date, "sst": 27.1, "sst_anomaly": 0.2}

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_ghrsst_point_value", fake_point)
    monkeypatch.setattr(router, "_ghrsst_bbox_mean", lambda **_: None)

    result = await router.classify_and_route(
        "現在台灣周遭(123,25)海溫多少？",
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

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

    async def fake_onepass(query, debug, today, tz, query_time):
        return dummy, {"mode": "mcp_tools", "mcp": spec, "citations": []}

    async def fake_bbox(bbox, date=None, fields=None, method=None):
        assert bbox == [118.0, 23.0, 123.0, 30.0]
        return {"region": bbox, "date": date, "sst": 26.5, "sst_anomaly": -0.1}

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_ghrsst_bbox_mean", fake_bbox)
    monkeypatch.setattr(router, "_ghrsst_point_value", lambda **_: None)

    result = await router.classify_and_route(
        "bbox dict test",
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert result["mode"] == "mcp_tools"
    assert result["result"]["sst"] == 26.5
    assert result["arguments"]["bbox"] == [118.0, 23.0, 123.0, 30.0]


@pytest.mark.asyncio
async def test_router_tide_forecast(monkeypatch):
    sample_spec = {
        "tool": "tide.forecast",
        "arguments": {
            "longitude": 121.5,
            "latitude": 25.0,
            "tz": TEST_TZ,
            "query_time": TEST_QUERY_TIME,
            "date": "{today}",
        },
        "confidence": 0.77,
    }
    dummy = DummyResult(mode="mcp_tools", mcp=sample_spec)

    async def fake_onepass(query, debug, today, tz, query_time):
        return dummy, {"mode": "mcp_tools", "mcp": sample_spec, "citations": []}

    tide_payload = {
        "date": "2025-01-10",
        "tz": TEST_TZ,
        "state_now": "rising",
        "last_extreme": {"type": "low", "time": "2025-01-10T04:00:00+08:00", "height": 0.1},
        "next_extreme": {"type": "high", "time": "2025-01-10T10:00:00+08:00", "height": 1.1},
        "since_extreme": "PT1H",
        "until_extreme": "PT2H",
        "sun": {
            "begin_civil_twilight": "2025-01-10T05:45:00+08:00",
            "sunrise": "2025-01-10T06:16:00+08:00",
            "sunset": "2025-01-10T17:07:00+08:00",
            "end_civil_twilight": "2025-01-10T17:40:00+08:00",
        },
        "moon": {
            "moonrise": "2025-01-10T15:30:00+08:00",
            "moonset": "2025-01-11T03:00:00+08:00",
        },
        "moonphase": {"current": "Waning Crescent", "illumination": "40%"},
        "messages": [
            "Note: Tide heights are relative to NTDE 1983–2001 MSL and provided for reference only."
        ],
        "meta": {},
    }

    async def fake_tide(**kwargs):
        return tide_payload

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_tide_forecast", fake_tide)

    result = await router.classify_and_route(
        "(121.5,25.0) 現在是漲潮嗎？",
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert result["tool"] == "tide.forecast"
    assert "漲潮" in result["text"]
    assert "曙光" in result["text"]
    assert "日出" in result["text"]
    assert "月盈:40%" in result["text"]
    assert result["arguments"]["date"] == "2025-01-10"


@pytest.mark.asyncio
async def test_router_auto_tide_fallback(monkeypatch):
    async def fake_onepass(query, debug, today, tz, query_time):
        return DummyResult(mode="explain", text="N/A"), {"mode": "explain", "text": "N/A", "citations": []}

    async def fake_tide(**kwargs):
        return {
            "date": "2025-01-10",
            "tz": TEST_TZ,
            "state_now": "falling",
            "last_extreme": {"type": "high", "time": "2025-01-10T02:00:00+08:00"},
            "next_extreme": {"type": "low", "time": "2025-01-10T08:00:00+08:00"},
            "since_extreme": "PT2H",
            "until_extreme": "PT1H",
            "sun": {},
            "moon": {},
            "meta": {},
        }

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_tide_forecast", fake_tide)

    result = await router.classify_and_route(
        "(121.5,25.0) 何時滿潮？乾潮？日出？",
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert result["tool"] == "tide.forecast"
    assert "乾潮" in result["text"]


@pytest.mark.asyncio
async def test_router_tide_forecast_english(monkeypatch):
    sample_spec = {
        "tool": "tide.forecast",
        "arguments": {
            "longitude": 121.5,
            "latitude": 25.0,
            "tz": TEST_TZ,
            "query_time": TEST_QUERY_TIME,
            "date": "{today}",
        },
    }
    dummy = DummyResult(mode="mcp_tools", mcp=sample_spec)

    async def fake_onepass(query, debug, today, tz, query_time):
        return dummy, {"mode": "mcp_tools", "mcp": sample_spec, "citations": []}

    tide_payload = {
        "date": "2025-01-10",
        "tz": TEST_TZ,
        "tide": {
            "highs": [{"time": "2025-01-10T10:00:00+08:00", "height_cm": 45}],
            "lows": [{"time": "2025-01-10T04:00:00+08:00", "height_cm": -61}],
        },
        "sun": {
            "begin_civil_twilight": "2025-01-10T05:50:00+08:00",
            "sunrise": "2025-01-10T06:16:00+08:00",
            "sunset": "2025-01-10T17:07:00+08:00",
            "end_civil_twilight": "2025-01-10T17:35:00+08:00",
        },
        "moon": {
            "moonrise": "2025-01-10T15:30:00+08:00",
            "moonset": "2025-01-11T03:00:00+08:00",
        },
        "moonphase": {"current": "Waning Crescent", "illumination": "40%"},
        "messages": [
            "Note: Tide heights are relative to NTDE 1983–2001 MSL and provided for reference only."
        ],
    }

    async def fake_tide(**kwargs):
        return tide_payload

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_tide_forecast", fake_tide)

    result = await router.classify_and_route(
        "When is the next high tide near (121.5,25.0)?",
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert "Next low tide" in result["text"]
    assert "Civil dawn" in result["text"]
    assert "Sunrise" in result["text"]
    assert "Moon phase" in result["text"]
    assert "潮高" not in result["text"]
    assert "小時" not in result["text"]
    assert "4h" in result["text"]


@pytest.mark.asyncio
async def test_router_sunrise_only_no_notes(monkeypatch):
    sample_spec = {
        "tool": "tide.forecast",
        "arguments": {
            "longitude": 121.0,
            "latitude": 23.0,
            "tz": TEST_TZ,
            "query_time": TEST_QUERY_TIME,
            "date": "{today}",
        },
    }
    dummy = DummyResult(mode="mcp_tools", mcp=sample_spec)

    async def fake_onepass(query, debug, today, tz, query_time):
        return dummy, {"mode": "mcp_tools", "mcp": sample_spec, "citations": []}

    sunrise_payload = {
        "date": "2025-01-10",
        "tz": TEST_TZ,
        "sun": {
            "begin_civil_twilight": "2025-01-10T05:40:00+08:00",
            "sunrise": "2025-01-10T06:08:00+08:00",
            "sunset": "2025-01-10T17:13:00+08:00",
            "end_civil_twilight": "2025-01-10T17:40:00+08:00",
        },
        "moon": {
            "moonrise": "2025-01-10T00:53:00+08:00",
            "moonset": "2025-01-10T13:38:00+08:00",
        },
        "moonphase": {"current": "Waning Crescent", "illumination": "31%"},
        "tide": {},
    }

    async def fake_tide(**kwargs):
        return sunrise_payload

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_tide_forecast", fake_tide)

    result = await router.classify_and_route(
        "日出時間?",
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert "曙光" in result["text"]
    assert "日出" in result["text"]
    assert "潮高" not in result["text"]


@pytest.mark.asyncio
async def test_router_moonrise_english_query_routes_to_tide(monkeypatch):
    async def fake_onepass(query, debug, today, tz, query_time):
        return DummyResult(mode="explain", text="N/A"), {"mode": "explain", "text": "N/A", "citations": []}

    async def fake_tide(**kwargs):
        return {
            "date": "2025-01-10",
            "tz": TEST_TZ,
            "sun": {},
            "moon": {
                "moonrise": "2025-01-10T00:53:00+08:00",
                "moonset": "2025-01-10T13:38:00+08:00",
            },
            "moonphase": {"current": "Waning Crescent", "illumination": "31%"},
            "tide": {},
        }

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_tide_forecast", fake_tide)

    result = await router.classify_and_route(
        "Moon rise at (121.0045, 22.475)?",
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert result["mode"] == "mcp_tools"
    assert result["tool"] == "tide.forecast"
    assert "Moonrise" in result["text"]
    assert "Moon phase" in result["text"]
    assert "Next high tide" not in result["text"]
    assert "Tide heights" not in result["text"]


@pytest.mark.asyncio
async def test_router_tide_specific_date_in_query(monkeypatch):
    async def fake_onepass(query, debug, today, tz, query_time):
        return DummyResult(mode="explain", text="N/A"), {"mode": "explain", "text": "N/A", "citations": []}

    captured: dict = {}

    async def fake_tide(**kwargs):
        captured.update(kwargs)
        return {
            "date": kwargs["date"],
            "tz": kwargs.get("tz"),
            "sun": {
                "sunrise": f"{kwargs['date']}T06:10:00+08:00",
                "sunset": f"{kwargs['date']}T17:05:00+08:00",
            },
            "moon": {},
            "moonphase": {},
            "tide": {},
        }

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_tide_forecast", fake_tide)

    query = "日期2025/11/25 座標約為 (121.5, 25.0) 日落時間？"
    result = await router.classify_and_route(
        query,
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert result["mode"] == "mcp_tools"
    assert result["tool"] == "tide.forecast"
    assert captured.get("date") == "2025-11-25"
    assert ("Sunset" in result["text"]) or ("日落" in result["text"])


@pytest.mark.asyncio
async def test_router_tide_query_without_parentheses_lat_first(monkeypatch):
    async def fake_onepass(query, debug, today, tz, query_time):
        return DummyResult(mode="explain", text="N/A"), {"mode": "explain", "text": "N/A", "citations": []}

    captured: dict = {}

    async def fake_tide(**kwargs):
        captured.update(kwargs)
        return {
            "date": kwargs["date"],
            "tz": kwargs.get("tz"),
            "sun": {
                "sunrise": f"{kwargs['date']}T06:10:00+08:00",
                "sunset": f"{kwargs['date']}T17:05:00+08:00",
            },
            "moon": {},
            "moonphase": {},
            "tide": {
                "highs": [],
                "lows": [],
            },
        }

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_tide_forecast", fake_tide)

    query = "2025/11/25 座標約為 25.2079577, 121.4286115 看日出？"
    result = await router.classify_and_route(
        query,
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert result["mode"] == "mcp_tools"
    assert result["tool"] == "tide.forecast"
    assert pytest.approx(captured.get("longitude"), rel=1e-6) == 121.4286115
    assert pytest.approx(captured.get("latitude"), rel=1e-6) == 25.2079577
    assert captured.get("date") == "2025-11-25"
    assert "日出" in result["text"] or "Sunrise" in result["text"]


@pytest.mark.asyncio
async def test_router_tide_future_date_summary_with_tide(monkeypatch):
    async def fake_onepass(query, debug, today, tz, query_time):
        return DummyResult(mode="explain", text="N/A"), {"mode": "explain", "text": "N/A", "citations": []}

    captured: dict = {}

    async def fake_tide(**kwargs):
        captured.update(kwargs)
        return {
            "date": "2025-11-25",
            "tz": TEST_TZ,
            "tide": {
                "highs": [
                    {"time": "2025-11-25T01:02:00+08:00", "height_cm": 60},
                    {"time": "2025-11-25T14:16:00+08:00", "height_cm": 95},
                ],
                "lows": [
                    {"time": "2025-11-25T07:35:00+08:00", "height_cm": -110},
                    {"time": "2025-11-25T20:29:00+08:00", "height_cm": -40},
                ],
            },
            "sun": {},
            "moon": {},
            "moonphase": {},
        }

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_tide_forecast", fake_tide)

    query = "2025-11-25 (121.5, 25.0) 潮汐？"
    result = await router.classify_and_route(
        query,
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert result["mode"] == "mcp_tools"
    assert "目前" not in result["text"]
    assert "Next" not in result["text"]
    assert "滿潮" in result["text"]
    assert "資料日期 2025-11-25" in result["text"]


@pytest.mark.asyncio
async def test_router_tide_coordinates_with_cardinal_letters(monkeypatch):
    async def fake_onepass(query, debug, today, tz, query_time):
        return DummyResult(mode="explain", text="N/A"), {"mode": "explain", "text": "N/A", "citations": []}

    captured: dict = {}

    async def fake_tide(**kwargs):
        captured.update(kwargs)
        return {
            "date": kwargs["date"],
            "tz": kwargs.get("tz"),
            "tide": {
                "highs": [{"time": f"{kwargs['date']}T01:00:00+08:00", "height_cm": 50}],
                "lows": [{"time": f"{kwargs['date']}T07:00:00+08:00", "height_cm": -60}],
            },
            "sun": {},
            "moon": {},
            "moonphase": {},
        }

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_tide_forecast", fake_tide)

    query = "Tidal condition at (25.2079577N, 121.4286115E) on 2025-11-25?"
    result = await router.classify_and_route(
        query,
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert result["mode"] == "mcp_tools"
    assert pytest.approx(captured.get("longitude"), rel=1e-6) == 121.4286115
    assert pytest.approx(captured.get("latitude"), rel=1e-6) == 25.2079577
    assert captured.get("date") == "2025-11-25"
    assert result["text"].startswith("High tide") or "滿潮" in result["text"]


@pytest.mark.asyncio
async def test_router_prefers_tide_when_query_mentions_tide_and_sst(monkeypatch):
    async def fake_onepass(query, debug, today, tz, query_time):
        return DummyResult(mode="explain", text="N/A"), {"mode": "explain", "text": "N/A", "citations": []}

    async def fake_tide(**kwargs):
        return {
            "date": "2025-01-10",
            "tz": TEST_TZ,
            "tide": {
                "highs": [{"time": "2025-01-10T10:00:00+08:00", "height_cm": 45}],
                "lows": [{"time": "2025-01-10T04:00:00+08:00", "height_cm": -30}],
            },
            "sun": {},
            "moon": {},
            "moonphase": {},
        }

    async def fail_point(*args, **kwargs):  # pragma: no cover
        raise AssertionError("ghrsst.point_value should not be called")

    async def fail_bbox(*args, **kwargs):  # pragma: no cover
        raise AssertionError("ghrsst.bbox_mean should not be called")

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_tide_forecast", fake_tide)
    monkeypatch.setattr(router, "_ghrsst_point_value", fail_point)
    monkeypatch.setattr(router, "_ghrsst_bbox_mean", fail_bbox)

    result = await router.classify_and_route(
        "現在台灣周遭(121.5,25.0)潮高與海溫多少？",
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert result["mode"] == "mcp_tools"
    assert result["tool"] == "tide.forecast"


@pytest.mark.asyncio
async def test_router_forces_tide_even_if_mode_is_code(monkeypatch):
    async def fake_onepass(query, debug, today, tz, query_time):
        payload = {"mode": "code", "citations": []}
        return DummyResult(mode="code", text="", mcp=None), payload

    async def fake_tide(**kwargs):
        return {
            "date": "2025-01-10",
            "tz": TEST_TZ,
            "tide": {
                "highs": [{"time": "2025-01-10T10:00:00+08:00", "height_cm": 42}],
                "lows": [{"time": "2025-01-10T04:00:00+08:00", "height_cm": -20}],
            },
            "sun": {},
            "moon": {},
            "moonphase": {},
        }

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_tide_forecast", fake_tide)

    result = await router.classify_and_route(
        "(121.5,25.0) 何時滿潮？",
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert result["mode"] == "mcp_tools"
    assert result["tool"] == "tide.forecast"
    assert "此點位並無潮汐資料" not in result["text"]


@pytest.mark.asyncio
async def test_router_tide_no_data_appends_notice(monkeypatch):
    async def fake_onepass(query, debug, today, tz, query_time):
        payload = {"mode": "explain", "citations": []}
        return DummyResult(mode="explain", text=""), payload

    async def fake_tide(**kwargs):
        return {
            "date": "2025-01-10",
            "tz": TEST_TZ,
            "tide": {},
            "sun": {},
            "moon": {},
            "moonphase": {},
        }

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_tide_forecast", fake_tide)

    result = await router.classify_and_route(
        "(121.5,25.0) 哪裡潮汐？",
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert "此點位並無潮汐資料" in result["text"]


@pytest.mark.asyncio
async def test_router_forces_ghrsst_even_if_mode_is_code(monkeypatch):
    async def fake_onepass(query, debug, today, tz, query_time):
        payload = {"mode": "code", "citations": []}
        return DummyResult(mode="code", text=""), payload

    async def fake_point(lon, lat, date=None, fields=None, method=None):
        return {"region": [lon, lat], "date": date, "sst": 27.0, "sst_anomaly": 0.1}

    monkeypatch.setattr(router, "_run_onepass_async", fake_onepass)
    monkeypatch.setattr(router, "_ghrsst_point_value", fake_point)
    monkeypatch.setattr(router, "_ghrsst_bbox_mean", lambda **_: None)

    result = await router.classify_and_route(
        "(121.5,25.0) 海溫多少？",
        tz=TEST_TZ,
        query_time=TEST_QUERY_TIME,
        debug=False,
    )

    assert result["tool"] == "ghrsst.point_value"
