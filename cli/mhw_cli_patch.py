# cli/mhw_cli_patch.py
from __future__ import annotations
import json
import sys
import os
import importlib
import importlib.util
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

MHW_HELP = (
    "Query ODB MHW via MCP and (optionally) plot.\n"
    "Usage:\n"
    "  /mhw --bbox lon0,lat0[,lon1,lat1] [--start YYYY[-MM[-DD]]] [--end YYYY[-MM[-DD]]]\n"
    "       [--fields sst,sst_anomaly,level,td] [--csv] [--raw]\n"
    "       [--plot series|month|map] [--plot-field sst|sst_anomaly|level|td]\n"
    "       [--periods \"YYYY,YYYYMM,YYYY-YYYY,YYYYMM-YYYYMM,...\"]\n"
    "       [--map-method cartopy|basemap|plain] [--cmap NAME] [--vmin X] [--vmax Y]\n"
    "       [--outfile path.png]\n"
    "Cmaps: sst→turbo/viridis; sst_anomaly→RdYlBu_r/coolwarm; "
    "level→fixed; td→viridis/BrBG/PuOr/RdYlGn (tip: --vmin 0)\n"
)

ALLOWED_FIELDS = {"sst", "sst_anomaly", "level", "td"}

# ------------------------------ utils ------------------------------

def _strip_q(s: str) -> str:
    return s.strip().strip('"').strip("'")

_PARSE = ("%Y-%m-%d", "%Y%m%d", "%Y-%m", "%Y%m", "%Y")

def _norm_month_first(s: str) -> str:
    s = _strip_q(s)
    for fmt in _PARSE:
        try:
            dt = datetime.strptime(s, fmt)
            return f"{dt.year:04d}-{dt.month:02d}-01"
        except Exception:
            pass
    dt = pd.to_datetime(s)
    return f"{dt.year:04d}-{dt.month:02d}-01"

def _latest_12_complete_months(today: Optional[date] = None) -> Tuple[str, str]:
    if today is None:
        today = date.today()
    if today.day >= 18:
        y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    else:
        if today.month > 2:
            y, m = today.year, today.month - 2
        else:
            y = today.year - 1
            m = 12 if today.month == 1 else 11
    end = date(y, m, 1)
    m2 = m - 11
    y2 = y + (m2 - 1) // 12
    m2 = ((m2 - 1) % 12) + 1
    start = date(y2, m2, 1)
    return start.isoformat(), end.isoformat()

def _records_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    return df

def _delta_lon_shortest(l0: float, l1: float) -> float:
    d = ((l1 - l0 + 180.0) % 360.0) - 180.0
    return abs(d)

# ------------------------------ periods & bbox ------------------------------

def _parse_periods_csv(csv: Optional[str], default_start: Optional[str], default_end: Optional[str]) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    if csv and csv.strip():
        tokens = [t.strip() for t in csv.split(",") if t.strip()]
        for tok in tokens:
            if "-" in tok:
                a, b = tok.split("-", 1)
                sa = _norm_month_first(a)
                sb = _norm_month_first(b)
                out.append((sa, sb, f"{sa[:7]}–{sb[:7]}"))
            else:
                s1 = _norm_month_first(tok)
                out.append((s1, s1, s1[:7]))
    else:
        if default_start:
            s = _norm_month_first(default_start)
            e = _norm_month_first(default_end) if default_end else s
        else:
            s, e = _latest_12_complete_months()
        out.append((s, e, f"{s[:7]}–{e[:7]}"))
    return out

def _split_bbox_by_180(lon0: float, lat0: float, lon1: Optional[float], lat1: Optional[float]) -> List[Tuple[float, float, Optional[float], Optional[float]]]:
    if lon1 is None or lat1 is None:
        return [(lon0, lat0, None, None)]
    if _delta_lon_shortest(lon0, lon1) <= 180.0:
        return [(lon0, lat0, lon1, lat1)]
    # split into two continuous boxes
    eps = 1e-3
    return [(lon0, lat0, 180.0 - eps, lat1), (-180.0 + eps, lat0, lon1, lat1)]

# ------------------------------ plugin loading ------------------------------

def _load_symbol(module_names: List[str], file_candidates: List[str], symbol: str):
    for m in module_names:
        try:
            mod = importlib.import_module(m)
            fn = getattr(mod, symbol, None)
            if callable(fn):
                return fn
        except Exception:
            pass
    here = os.path.dirname(__file__)
    for rel in file_candidates:
        path = os.path.join(here, rel)
        if os.path.exists(path):
            try:
                spec = importlib.util.spec_from_file_location(symbol + "_dyn", path)
                mod = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(mod)  # type: ignore
                fn = getattr(mod, symbol, None)
                if callable(fn):
                    return fn
            except Exception:
                pass
    return None

def _plot_series(df, fields, **kw):
    fn = _load_symbol(
        ["cli.plugins.mhw_plot", "plugins.mhw_plot", "cli.mhw_plot", "mhw_plot"],
        ["plugins/mhw_plot.py", "mhw_plot.py"],
        "plot_series",
    )
    if not fn:
        print("[plot plugin missing] mhw_plot.plot_series", file=sys.stderr)
        return
    fn(df, fields=fields, **kw)

def _plot_month(df, field, periods, **kw):
    fn = _load_symbol(
        ["cli.plugins.mhw_plot", "plugins.mhw_plot", "cli.mhw_plot", "mhw_plot"],
        ["plugins/mhw_plot.py", "mhw_plot.py"],
        "plot_month_climatology",
    )
    if not fn:
        print("[plot plugin missing] mhw_plot.plot_month_climatology", file=sys.stderr)
        return
    fn(df, field=field, periods=periods, **kw)

def _plot_map(df, field, bbox, start, **kw):
    fn = _load_symbol(
        ["cli.plugins.map_plot", "plugins.map_plot", "cli.map_plot", "map_plot"],
        ["plugins/map_plot.py", "map_plot.py"],
        "plot_map",
    )
    if not fn:
        print("[map plugin missing] map_plot.plot_map", file=sys.stderr)
        return
    fn(df, field=field, bbox=bbox, start=start, **kw)

# ------------------------------ MCP call ------------------------------

async def _mcp_call(cli, tool: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not getattr(cli, "client", None):
        print("❌ Not connected. Use '/server connect' first.", file=sys.stderr)
        return None

    async def _do(tool_name: str):
        res = await cli.client.call_tool(tool_name, args)
        if isinstance(res, dict):
            return res
        if hasattr(res, "content"):
            c = res.content
            if isinstance(c, list) and c and hasattr(c[0], "text"):
                txt = c[0].text
                try:
                    return json.loads(txt) if txt and txt.strip().startswith("{") else {"raw": txt}
                except Exception:
                    return {"raw": txt}
        if hasattr(res, "text") and res.text:
            try:
                return json.loads(res.text) if res.text.strip().startswith("{") else {"raw": res.text}
            except Exception:
                return {"raw": res.text}
        return {"raw": str(res)}

    try:
        return await _do("mhw_fetch")
    except Exception:
        try:
            return await _do("mhw_query")
        except Exception as e:
            print(f"❌ MCP call failed: {e}", file=sys.stderr)
            return None

# ------------------------------ flags & canonicalization ------------------------------

def _parse_mhw_flags(tokens: List[str]) -> Dict[str, Any]:
    p: Dict[str, Any] = {
        "lon0": None, "lat0": None, "lon1": None, "lat1": None,
        "start": None, "end": None,
        "_fields": None, "_plot": None, "_plot_field": None,
        "_periods": None, "_outfile": None, "_map_method": None,
        "_cmap": None, "_vmin": None, "_vmax": None, "_csv": False, "_raw": False,
    }
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--bbox" and i + 1 < len(tokens):
            parts = [x.strip() for x in tokens[i + 1].split(",")]
            if len(parts) >= 2:
                p["lon0"], p["lat0"] = float(parts[0]), float(parts[1])
            if len(parts) == 4:
                p["lon1"], p["lat1"] = float(parts[2]), float(parts[3])
            i += 2; continue
        if t == "--start" and i + 1 < len(tokens): p["start"] = _strip_q(tokens[i + 1]); i += 2; continue
        if t == "--end"   and i + 1 < len(tokens): p["end"]   = _strip_q(tokens[i + 1]); i += 2; continue
        if t == "--fields" and i + 1 < len(tokens): p["_fields"] = _strip_q(tokens[i + 1]); i += 2; continue
        if t == "--plot" and i + 1 < len(tokens): p["_plot"] = tokens[i + 1].lower(); i += 2; continue
        if t in ("--plot-field", "--plot_fields") and i + 1 < len(tokens): p["_plot_field"] = tokens[i + 1]; i += 2; continue
        if t == "--periods" and i + 1 < len(tokens): p["_periods"] = _strip_q(tokens[i + 1]); i += 2; continue
        if t == "--outfile" and i + 1 < len(tokens): p["_outfile"] = tokens[i + 1]; i += 2; continue
        if t == "--map-method" and i + 1 < len(tokens): p["_map_method"] = tokens[i + 1]; i += 2; continue
        if t == "--cmap" and i + 1 < len(tokens): p["_cmap"] = tokens[i + 1]; i += 2; continue
        if t == "--vmin" and i + 1 < len(tokens):
            try: p["_vmin"] = float(tokens[i + 1])
            except Exception: p["_vmin"] = None
            i += 2; continue
        if t == "--vmax" and i + 1 < len(tokens):
            try: p["_vmax"] = float(tokens[i + 1])
            except Exception: p["_vmax"] = None
            i += 2; continue
        if t == "--csv": p["_csv"] = True; i += 1; continue
        if t == "--raw": p["_raw"] = True; i += 1; continue
        i += 1
    return p

def _canonical_fetch_arglist(p: Dict[str, Any]) -> List[Dict[str, Any]]:
    lon0, lat0, lon1, lat1 = p.get("lon0"), p.get("lat0"), p.get("lon1"), p.get("lat1")
    if lon0 is None or lat0 is None:
        lon0, lat0, lon1, lat1 = 118.0, 21.0, 123.0, 26.0  # Taiwan default

    periods = _parse_periods_csv(p.get("_periods"), p.get("start"), p.get("end"))
    bbox_parts = _split_bbox_by_180(float(lon0), float(lat0),
                                    None if lon1 is None else float(lon1),
                                    None if lat1 is None else float(lat1))

    fields = p.get("_fields") or "sst,sst_anomaly"
    outmode = "csv" if p.get("_csv") else "json"
    include = not p.get("_csv")

    arglist: List[Dict[str, Any]] = []
    for (s, e, _lab) in periods:
        for (bx0, by0, bx1, by1) in bbox_parts:
            arglist.append({
                "lon0": bx0, "lat0": by0,
                "lon1": bx1, "lat1": by1,
                "start": s, "end": e,
                "append": fields,
                "output": outmode,
                "include_data": include,
            })
    return arglist

def _warn_limits(bbox: Tuple[float, float, Optional[float], Optional[float]], start: str, end: str):
    try:
        lo0, la0, lo1, la1 = bbox
        if lo1 is None or la1 is None:
            return
        lon_span = abs(lo1 - lo0)
        lat_span = abs(la1 - la0)
        days = (pd.to_datetime(end) - pd.to_datetime(start)).days + 1
        msg = None
        if lon_span > 90 and lat_span > 90 and days > 31:
            msg = ">90×90° bbox → ~1 month max (server will clamp)"
        elif lon_span > 10 and lat_span > 10 and days > 366:
            msg = ">10×10° bbox → ~1 year max (server will clamp)"
        elif lon_span <= 10 and lat_span <= 10 and days > 3650:
            msg = "≤10×10° bbox → ~10 years max (server will clamp)"
        if msg:
            print(f"[notice] {msg}. Consider splitting.", file=sys.stderr)
    except Exception:
        pass

# ------------------------------ main handler ------------------------------

async def handle_mhw(cli, line: str):
    tokens = line.strip().split()[1:]  # drop '/mhw'
    p = _parse_mhw_flags(tokens)

    arglist = _canonical_fetch_arglist(p)
    if not arglist:
        print("[mhw] no arguments produced from flags", file=sys.stderr)
        return

    a0 = arglist[0]
    _warn_limits((a0["lon0"], a0["lat0"], a0.get("lon1"), a0.get("lat1")), a0["start"], a0.get("end") or a0["start"])

    if arglist[0]["output"] == "csv":
        outs = []
        for args in arglist:
            res = await _mcp_call(cli, "mhw_fetch", args)
            if not res:
                print("[mcp error] mhw_fetch failed", file=sys.stderr); return
            outs.append(res)
        print(json.dumps(outs if len(outs) > 1 else outs[0], ensure_ascii=False, indent=2))
        return

    frames: List[pd.DataFrame] = []
    metas: List[Dict[str, Any]] = []
    for args in arglist:
        res = await _mcp_call(cli, "mhw_fetch", args)
        if not res or "data" not in res:
            print("[mcp error] mhw_fetch did not return data", file=sys.stderr)
            return
        df = _records_to_df(res["data"])
        s_lab = (args["start"] or "")[:7]
        e_lab = (args.get("end") or args["start"] or "")[:7]
        df["period"] = f"{s_lab}–{e_lab}"
        frames.append(df)
        metas.append({k: v for k, v in res.items() if k != "data"})

    if not frames:
        print("[mhw] empty result", file=sys.stderr)
        return

    big = pd.concat(frames, ignore_index=True, sort=False)

    # concise merged summary
    if len(metas) == 1:
        summary = metas[0]; summary.pop("data", None)
    else:
        summary = {
            "note": f"merged {len(metas)} API responses (periods × bbox parts)",
            "components": [{"bbox": m.get("bbox"), "period": m.get("period")} for m in metas],
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if p.get("_raw"):
        print(big.head().to_string(index=False)); return

    mode = (p.get("_plot") or "").lower()

    if mode == "series":
        fields = [f.strip() for f in (p.get("_plot_field") or arglist[0]["append"]).split(",") if f.strip() in ALLOWED_FIELDS]
        if not fields: fields = ["sst", "sst_anomaly"]
        _plot_series(big, fields=fields, outfile=p.get("_outfile"))
        return

    if mode == "month":
        fld = (p.get("_plot_field") or "sst").strip()
        _plot_month(big, field=fld, periods=p.get("_periods") or "", outfile=p.get("_outfile"))
        return

    if mode == "map":
        fld = (p.get("_plot_field") or "sst_anomaly").strip()
        first = arglist[0]
        bbox = (first["lon0"], first["lat0"], first.get("lon1"), first.get("lat1"))
        start_for_map = first["start"]
        _plot_map(big, field=fld, bbox=bbox, start=start_for_map,
                  outfile=p.get("_outfile"), method=p.get("_map_method"),
                  cmap=p.get("_cmap"), vmin=p.get("_vmin"), vmax=p.get("_vmax"))
        return

    return

def patch_mhw_command(cli) -> None:
    if hasattr(cli, "register_command") and callable(getattr(cli, "register_command")):
        cli.register_command("/mhw", handle_mhw, help_text=MHW_HELP)
        return
    original = getattr(cli, "_handle_command", None)
    async def _patched(line: str):
        if line.strip().lower().startswith("/mhw"):
            await handle_mhw(cli, line); return
        if original:
            await original(line)
        else:
            print("❓ Unknown command. /help for help.")
    setattr(cli, "_handle_command", _patched)

# pluggable register
def install_mhw_command(cli):
    cli.register_command("/mhw", handle_mhw, help_text=MHW_HELP)

