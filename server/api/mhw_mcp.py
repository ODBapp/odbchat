# api/mhw_mcp.py
from __future__ import annotations
import requests
from fastmcp import FastMCP
# from typing import Dict, Any, List
from datetime import datetime
from shared.schemas import canonicalize_fetch, validate_fetch

MHW_API_JSON = "https://eco.odb.ntu.edu.tw/api/mhw"
MHW_API_CSV  = "https://eco.odb.ntu.edu.tw/api/mhw/csv"
ALLOWED_FIELDS = {"sst", "sst_anomaly", "level", "td"}

def _normalize_fields(append: str | None) -> str:
    if not append:
        return "sst,sst_anomaly"
    parts = [p.strip() for p in append.split(",") if p.strip()]
    parts = [p for p in parts if p in ALLOWED_FIELDS]
    if not parts:
        parts = ["sst", "sst_anomaly"]
    return ",".join(dict.fromkeys(parts))  # dedupe, keep order

def _to_float(v):
    return None if v is None else float(v)

def _summarize(data, fields): #List[Dict[str, Any]], fields: List[str]) -> Dict[str, Any]:
    out = {"count": len(data)} #out: Dict[str, Any] =
    if not data:
        return out
    try:
        dates = [d.get("date") for d in data if d.get("date")]
        ds = [datetime.fromisoformat(x).date() for x in dates]
        out["effective_period"] = [min(ds).isoformat(), max(ds).isoformat()]
    except Exception:
        pass
    for f in fields:
        vals = [d.get(f) for d in data if isinstance(d.get(f), (int, float))]
        if vals:
            out[f + "_mean"] = sum(vals) / len(vals)
    return out

async def _mhw_fetch_impl(
    lon0: float,
    lat0: float,
    lon1: float | None = None,
    lat1: float | None = None,
    start: str = "",
    end: str = "",
    append: str = "sst,sst_anomaly",
    output: str = "json",          # 'json' | 'csv'
    include_data: bool = True,     # for 'json' only
):  # -> Dict[str, Any]:
    """
        Fetch ODB Marine Heatwaves (MHW) data (thin tool; no intent/defaults).

        Required
        --------
        lon0, lat0 : lower-left (or single point) in degrees
        start, end : 'YYYY-MM-DD' (for JSON); map use-cases may set start=end (month)

        Optional
        --------
        lon1, lat1 : upper-right (omit for point)
        append     : comma-joined variables in {sst, sst_anomaly, level, td}
        output     : 'json' → records; 'csv' → URL only
        include_data: when output='json', include 'data' array (default True)

        Notes
        -----
        API will clamp long periods for large bboxes:
          ≤10×10° → ≤10y,  >10×10° → ≤1y,  >90×90° → ≤1mo.
        Results include a small 'summary' and echoed 'params'.

        TW 範例:
        1) JSON: 台灣近 12 個月（由 client 準備 start/end/bbox）
           mhw_fetch(118,21,123,26,"2024-07-01","2025-06-01","sst,sst_anomaly","json",True)
        2) CSV: 回傳 download_url，不帶 data
           mhw_fetch(118,21,123,26,"2024-07-01","2025-06-01","sst,sst_anomaly","csv",False)
    """
    
    payload = canonicalize_fetch({
        "lon0": lon0, "lat0": lat0, "lon1": lon1, "lat1": lat1,
        "start": start, "end": end, "append": append,
        "output": output, "include_data": include_data,
    })
    ok, err = validate_fetch(payload)
    if not ok:
        return {"error": f"invalid mhw_fetch args: {err}"}
    
    fields_csv = _normalize_fields(append)
    fields = [f.strip() for f in fields_csv.split(",")]

    url = MHW_API_CSV if output.lower() == "csv" else MHW_API_JSON
    params = { #: Dict[str, Any]
        "lon0": _to_float(lon0),
        "lat0": _to_float(lat0),
        "append": fields_csv,
    }
    if lon1 is not None: params["lon1"] = _to_float(lon1)
    if lat1 is not None: params["lat1"] = _to_float(lat1)
    if start: params["start"] = start
    if end:   params["end"] = end

    if output.lower() == "csv":
        from urllib.parse import urlencode
        return {
            "endpoint": url,
            "params": params,
            "bbox": [params.get("lon0"), params.get("lat0"), params.get("lon1"), params.get("lat1")],
            "period": [params.get("start"), params.get("end")],
            "fields": fields,
            "download_url": f"{url}?{urlencode(params)}",
        }

    resp = requests.get(url, params=params, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    result = { #: Dict[str, Any] = {
        "endpoint": url,
        "params": params,
        "bbox": [params.get("lon0"), params.get("lat0"), params.get("lon1"), params.get("lat1")],
        "period": [params.get("start"), params.get("end")],
        "fields": fields,
        "count": len(data),
        "summary": _summarize(data, fields),
    }
    if include_data:
        result["data"] = data
    return result

def _mhw_fetch_batch_impl(requests: list[dict]) -> dict:
    """
        Batch fetch MHW data.
        requests: a list of mhw_fetch argument dicts (same schema as single-call).
        Returns:
            {
                "data": [ ... merged rows ... ],
                "components": [ {meta for each sub-call except 'data'} ... ]
            }
    """
        
    if not isinstance(requests, list) or not requests:
        return {"error": "requests must be a non-empty list"}

    merged = []
    components = []
    for r in requests:
        r = canonicalize_fetch(r)
        ok, err = validate_fetch(r)
        if not ok:
            return {"error": f"invalid request: {err}"}

        # Reuse internal single-call helper _mhw_fetch_impl(**r) that returns {"data": [...], ...}
        res = _mhw_fetch_impl(**r)  # <-- hits the ODB API
        if isinstance(res, dict) and "data" in res:
            merged.extend(res["data"])
            components.append({k: v for k, v in res.items() if k != "data"})
        else:
            return {"error": f"fetch failed for request: {r}"}

    return {"data": merged, "components": components}

def register_mhw_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    async def mhw_fetch(
        lon0: float,
        lat0: float,
        lon1: float | None = None,
        lat1: float | None = None,
        start: str = "",
        end: str = "",
        append: str = "sst,sst_anomaly",
        output: str = "json",          # 'json' | 'csv'
        include_data: bool = True,     # for 'json' only
    ): 
        return await _mhw_fetch_impl(lon0, lat0, lon1, lat1, start, end, append, output, include_data)    

    @mcp.tool()
    def mhw_fetch_batch(requests: list[dict]) -> dict:
        return _mhw_fetch_batch_impl(requests)
        
