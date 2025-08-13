# api/mhw_mcp.py
from __future__ import annotations
# from typing import (
     # Optional, Literal,
     # Dict, Any, List, Tuple
#)
from datetime import date
import requests

MHW_API_JSON = "https://eco.odb.ntu.edu.tw/api/mhw"
MHW_API_CSV  = "https://eco.odb.ntu.edu.tw/api/mhw/csv"
ALLOWED_FIELDS = {"sst", "sst_anomaly", "level", "td"}

REGION_PRESETS = {
    "taiwan": (118.0, 21.0, 123.0, 26.0),
    "台灣":   (118.0, 21.0, 123.0, 26.0),
    "tw":     (118.0, 21.0, 123.0, 26.0),
}

def _default_range(today: date | None = None):
    """12-month window; latest month = (today >=18 ? last month : two months ago)."""
    if today is None:
        today = date.today()
    if today.day >= 18:
        end_y, end_m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    else:
        if today.month > 2:
            end_y, end_m = (today.year, today.month - 2)
        else:
            end_y = today.year - 1
            end_m = 12 if today.month == 1 else 11
    end = date(end_y, end_m, 1)
    # 11 months before end → total 12 months
    y, m = end.year, end.month
    m2 = m - 11
    y2 = y + (m2 - 1) // 12
    m2 = ((m2 - 1) % 12) + 1
    start = date(y2, m2, 1)
    return start.isoformat(), end.isoformat()

def _pick_bbox(
    region_hint: str | None = None, # Optional[str],
    lon0: float | None = None, # Optional[float],
    lat0: float | None = None, # Optional[float],
    lon1: float | None = None, # Optional[float],
    lat1: float | None = None  # Optional[float],
): # -> Tuple[float, float, Optional[float], Optional[float]]:
    if lon0 is not None and lat0 is not None:
        return float(lon0), float(lat0), (None if lon1 is None else float(lon1)), (None if lat1 is None else float(lat1))
    if region_hint:
        key = region_hint.strip().lower()
        if key in REGION_PRESETS:
            return REGION_PRESETS[key]
    return REGION_PRESETS["taiwan"]

def _normalize_fields(append: str | None = None) -> str:
    if not append:
        return "sst,sst_anomaly"
    parts = [p.strip() for p in append.split(",") if p.strip()]
    parts = [p for p in parts if p in ALLOWED_FIELDS]
    if not parts:
        parts = ["sst", "sst_anomaly"]
    # 去重保序
    return ",".join(dict.fromkeys(parts))

def _needs_csv(user_intent: str | None = None, output: str | None = None) -> bool:
    if output and output.lower() == "csv":
        return True
    if not user_intent:
        return False
    hint = user_intent.lower()
    return any(k in hint for k in ["csv", "下載", "檔案"])

def _area_mean(records, fields): # List[Dict[str, Any]], fields: List[str]) -> Dict[str, Any]:
    n = len(records)
    out = {"count": n}
    if n == 0:
        return out
    for f in fields:
        vals = [r.get(f) for r in records if r.get(f) is not None]
        if vals:
            out[f + "_mean"] = sum(vals) / len(vals)
    return out

def register_mhw_tools(mcp) -> None:
    """
    Call from your server to register the mhw_query tool onto the FastMCP instance.
    """

    @mcp.tool()
    async def mhw_query(
          user_intent: str | None = None,
          region_hint: str | None = None,
          output: str = "json",            # validate below
          lon0: float | None = None,
          lat0: float | None = None,
          lon1: float | None = None,
          lat1: float | None = None,
          start: str | None = None,
          end: str | None = None,
          append: str | None = None,
          return_raw: bool | None = False,
        ):  # -> dict[str, Any]:
        """
        Query ODB MHW data with convenient defaults and CSV intent detection.
        """
        bx0, by0, bx1, by1 = _pick_bbox(region_hint, lon0, lat0, lon1, lat1)

        if not start or not end:
            start, end = _default_range()

        if output not in ("json", "csv"):
            output = "json"

        append_norm = _normalize_fields(append)
        fields = [f.strip() for f in append_norm.split(",")]

        use_csv = _needs_csv(user_intent, output)
        url = MHW_API_CSV if use_csv else MHW_API_JSON

        params = {
            "lon0": bx0,
            "lat0": by0,
            "start": start,
            "end": end,
            "append": append_norm,
        }
        if bx1 is not None: params["lon1"] = bx1
        if by1 is not None: params["lat1"] = by1

        if use_csv:
            from urllib.parse import urlencode
            return {
                "endpoint": url,
                "download_url": f"{url}?{urlencode(params)}",
                "bbox": [bx0, by0, bx1, by1],
                "period": [start, end],
                "fields": fields,
            }

        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if return_raw:
            return {
                "endpoint": url,
                "params": params,
                "bbox": [bx0, by0, bx1, by1],
                "period": [start, end],
                "fields": fields,
                "data": data,
            }

        summary = _area_mean(data, fields)
        return {
            "endpoint": url,
            "params": params,
            "bbox": [bx0, by0, bx1, by1],
            "period": [start, end],
            "fields": fields,
            "summary": summary,
            "sample": data[:5],
        }
