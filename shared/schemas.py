# shared/schemas.py
from __future__ import annotations
from typing import Any, Dict, Tuple
from copy import deepcopy

MHWFIELDS = ["sst", "sst_anomaly", "level", "td"]

MHWFETCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "lon0": {"type": "number"},
        "lat0": {"type": "number"},
        "lon1": {"type": ["number", "null"]},
        "lat1": {"type": ["number", "null"]},
        "start": {"type": "string"},            # YYYY-MM-01 (normalized upstream)
        "end":   {"type": ["string", "null"]},
        "append": {
            "type": "string",
            "pattern": r"^(sst|sst_anomaly|level|td)(,(sst|sst_anomaly|level|td))*$"
        },
        "output": {"type": "string", "enum": ["json", "csv"]},
        "include_data": {"type": "boolean"},
    },
    "required": ["lon0", "lat0", "start", "append", "output", "include_data"]
}

PLOT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mode": {"type": "string", "enum": ["series", "month", "map"]},
        "fields": {
            "type": "array",
            "items": {"type": "string", "enum": MHWFIELDS},
            "minItems": 1,
            "uniqueItems": True
        },
        "field": {"type": "string", "enum": MHWFIELDS},
        "map_method": {"type": "string", "enum": ["cartopy", "basemap", "plain"]},
        "cmap": {"type": ["string", "null"]},
        "vmin": {"type": ["number", "null"]},
        "vmax": {"type": ["number", "null"]},
        "outfile": {"type": ["string", "null"]},
    },
    "required": ["mode"]
}

def canonicalize_fetch(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal normalization shared by CLI and MCP:
    - output defaults to 'json'
    - include_data defaults True when output=json
    - end defaults to start
    - append can be list or CSV, normalize to CSV
    """
    x = deepcopy(d)
    x.setdefault("output", "json")
    x.setdefault("include_data", x["output"] == "json")
    if x.get("end") is None:
        x["end"] = x["start"]
    if isinstance(x.get("append"), (list, tuple)):
        x["append"] = ",".join(str(v) for v in x["append"])
    return x

def canonicalize_plot(d: Dict[str, Any]) -> Dict[str, Any]:
    x = deepcopy(d)
    if "fields" not in x:
        if x.get("field"):
            x["fields"] = [x["field"]]
        else:
            x["fields"] = []
    x["fields"] = [f for f in x["fields"] if f in MHWFIELDS]
    return x

def validate_fetch(d: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        from jsonschema import Draft202012Validator
        Draft202012Validator(MHWFETCH_SCHEMA).validate(d)
        return True, ""
    except Exception as e:
        return False, str(e)

def validate_plot(d: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        from jsonschema import Draft202012Validator
        Draft202012Validator(PLOT_SCHEMA).validate(d)
        return True, ""
    except Exception as e:
        return False, str(e)
