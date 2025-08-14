# cli/mhw_cli_patch.py
from __future__ import annotations
from typing import Optional, Dict, Any, List, Callable, Tuple
import json
import sys
import os
import requests
import pandas as pd
import importlib
import importlib.util
from datetime import datetime, date

MHW_API_JSON = "https://eco.odb.ntu.edu.tw/api/mhw"
MHW_API_CSV  = "https://eco.odb.ntu.edu.tw/api/mhw/csv"

# Will be set by install_mhw_command(cli)
_CLI = None  # type: ignore

def _extract_text_like_cli(res: Any) -> str:
    """
    Mirror odbchat_cli.ODBChatClient._extract_text behavior so we can parse tool results.
    """

    # match the logic in odbchat_cli.py
    if hasattr(res, 'text') and res.text is not None:
        return res.text
    if hasattr(res, 'content'):
        c = res.content
        if isinstance(c, list) and c:
            first = c[0]
            if hasattr(first, 'text') and first.text is not None:
                return first.text
            return str(first)
        return str(c)
    if isinstance(res, str):
        return res
    return json.dumps(res, ensure_ascii=False)

async def _call_mcp_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    global _CLI
    if _CLI is None or _CLI.client is None:
        return None
    try:
        res = await _CLI.client.call_tool(tool_name, args)
        text = _extract_text_like_cli(res)
        return json.loads(text)
    except Exception:
        return None

def _http_json(params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(MHW_API_JSON, params=params, timeout=60)
    r.raise_for_status()
    return {"data": r.json(), "endpoint": MHW_API_JSON, "params": params}

def _http_csv(params: Dict[str, Any]) -> Dict[str, Any]:
    from urllib.parse import urlencode
    return {
        "download_url": f"{MHW_API_CSV}?{urlencode(params)}",
        "endpoint": MHW_API_CSV,
        "params": params
    }

def _parse_bbox(bbox: Optional[str]) -> Dict[str, Any]:
    if not bbox:
        return {}
    parts = [x.strip() for x in bbox.split(",") if x.strip()]
    if len(parts) not in (2,4):
        print("Invalid --bbox. Use lon0,lat0[,lon1,lat1].", file=sys.stderr)
        return {}
    out = {"lon0": float(parts[0]), "lat0": float(parts[1])}
    if len(parts) == 4:
        out["lon1"] = float(parts[2]); out["lat1"] = float(parts[3])
    return out

def _records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

def _build_args_from_tokens(tokens: List[str]) -> Dict[str, Any]:
    """
    Simple inline parser for: /mhw [free-text intent] [flags]
    Flags:
      --bbox lon0,lat0[,lon1,lat1]
      --start YYYY-MM-DD
      --end YYYY-MM-DD
      --fields sst,sst_anomaly,level,td
      --csv
      --raw
      --plot series|month
      --plot-field sst|sst_anomaly|level|td
      --periods "1982-2011,2012-2021,2024"
      --outfile path.png
    """

    args: Dict[str, Any] = {}
    freeform: List[str] = []

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--bbox" and i+1 < len(tokens):
            args.update(_parse_bbox(tokens[i+1])); i += 2; continue
        if t == "--start" and i+1 < len(tokens):
            args["start"] = tokens[i+1]; i += 2; continue
        if t == "--end" and i+1 < len(tokens):
            args["end"] = tokens[i+1]; i += 2; continue
        if t == "--fields" and i+1 < len(tokens):
            args["append"] = tokens[i+1]; i += 2; continue
        if t == "--csv":
            args["output"] = "csv"; i += 1; continue
        if t == "--raw":
            args["return_raw"] = True; i += 1; continue
        if t == "--plot" and i+1 < len(tokens):
            args["_plot_mode"] = tokens[i+1]; i += 2; continue
        if t == "--plot-field" and i+1 < len(tokens):
            args["_plot_field"] = tokens[i+1]; i += 2; continue
        if t == "--periods" and i+1 < len(tokens):
            args["_periods"] = tokens[i+1]; i += 2; continue
        if t == "--outfile" and i+1 < len(tokens):
            args["_outfile"] = tokens[i+1]; i += 2; continue
        # collect free-form
        freeform.append(t); i += 1

    if freeform:
        args["user_intent"] = " ".join(freeform)
    return args

def _load_plot_plugin():
    """
    Try multiple strategies so CLI works whether you run as a package or a plain script.
    Returns a module with plot_series / plot_month_climatology or None.
    """
    # 0) Already importable?
    for modname in ("plugins.mhw_plot", "mhw_plot", "cli.plugins.mhw_plot", "cli.mhw_plot"):
        try:
            return importlib.import_module(modname)
        except Exception:
            pass

    # 1) Load by file path next to this file
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "plugins", "mhw_plot.py"),
        os.path.join(here, "mhw_plot.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                spec = importlib.util.spec_from_file_location("mhw_plot_dyn", path)
                mod = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(mod)
                return mod
            except Exception:
                continue

    return None

def _run_plot(df, args, fields):
    mode = args.get("_plot_mode")
    if not mode:
        return

    plug = _load_plot_plugin()
    if plug is None:
        print("[plot plugin missing] Could not find plugins/mhw_plot.py or mhw_plot.py", file=sys.stderr)
        print("Tip: place the plugin under cli/plugins/mhw_plot.py or cli/mhw_plot.py, or install as a module.", file=sys.stderr)
        return

    # Handle missing matplotlib nicely
    try:
        plot_series = getattr(plug, "plot_series")
        plot_month_climatology = getattr(plug, "plot_month_climatology")
    except Exception as e:
        print(f"[plot plugin error] {e}", file=sys.stderr)
        return

    try:
        if mode == "series":
            title = "MHW Time Series"
            outfile = args.get("_outfile")
            plot_series(df, fields=fields, title=title, outfile=outfile)
            if outfile:
                print(f"🖼  Saved: {outfile}")
        elif mode == "month":
            periods_raw = args.get("_periods")
            periods = [p.strip() for p in (periods_raw or "").split(",") if p.strip()] or None
            field = args.get("_plot_field") or "sst"
            outfile = args.get("_outfile")
            plot_month_climatology(df, field=field, periods=periods,
                                   title=f"Monthly Climatology ({field})", outfile=outfile)
            if outfile:
                print(f"🖼  Saved: {outfile}")
        else:
            print(f"Unknown plot mode: {mode}", file=sys.stderr)

    except ModuleNotFoundError as me:
        # Typical: No module named 'matplotlib'
        missing = getattr(me, "name", "a required plotting dependency")
        print(f"[plot dependency missing] {me}", file=sys.stderr)
        print("Tip: plotting is optional. Install minimal deps for the plugin:", file=sys.stderr)
        print("     pip install matplotlib pandas", file=sys.stderr)
    except Exception as e:
        print(f"[plot error] {e}", file=sys.stderr)

# -----------------------------
# Period parsing for API params
# -----------------------------
def _strip_quotes(s: str) -> str:
    if not s:
        return s
    quotes = {'"', "'", '“', '”', '‘', '’'}
    out = s.strip()
    while len(out) >= 2 and out[0] in quotes and out[-1] in quotes:
        out = out[1:-1].strip()
    return out

def _convert_to_day(date_str: str) -> str:
    """
    Convert various compact date strings to 'YYYY-MM-DD'.
    Accepts: YYYY-MM-DD, YYYYMMDD, YYYY-MM, YYYYMM, YYYY.
    Returns YYYY-MM-DD (first day for YYYY/YYYMM forms).
    """
    s = _strip_quotes((date_str or '').strip())
    # Try precise formats first
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    # Month-only
    for fmt in ("%Y-%m", "%Y%m"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-01")
        except ValueError:
            pass
    # Year-only
    if len(s) == 4 and s.isdigit():
        try:
            return datetime.strptime(s, "%Y").strftime("%Y-01-01")
        except ValueError:
            pass
    raise ValueError(f"Invalid date format: {date_str}")

def _period_to_bounds(period: str, default_start: str = '1982-01-01') -> Tuple[str, str]:
    """
    Convert 'start-end' period string into YYYY-MM-DD start/end.
    start/end may be '', meaning default_start or latest full month start.
    Supports start/end formats accepted by _convert_to_day.
    """
    raw = _strip_quotes((period or '').strip())
    if '-' not in raw:
        # Single token: treat as a single year/month/day; compute a 1-year or 1-month window? Use exact day → same day.
        # For climatology, single token likely means a year. We'll map to full coverage:
        s_norm = _convert_to_day(raw)
        # Determine granularity by length
        if len(raw) == 4:  # year
            s = datetime.strptime(s_norm, "%Y-%m-%d")
            e = datetime(s.year + 1, 1, 1)
            e = (e.replace(day=1) - pd.DateOffset(days=1)).date()
            return s_norm, date(e.year, e.month, e.day).strftime("%Y-%m-%d")
        if len(raw) in (6, 7):  # YYYYMM or YYYY-MM as month
            s = datetime.strptime(s_norm, "%Y-%m-%d")
            # next month first day
            if s.month == 12:
                e = datetime(s.year + 1, 1, 1)
            else:
                e = datetime(s.year, s.month + 1, 1)
            e = (e - pd.DateOffset(days=1)).date()
            return s_norm, date(e.year, e.month, e.day).strftime("%Y-%m-%d")
        # exact day
        return s_norm, s_norm

    start, end = raw.split('-', 1)
    start = _strip_quotes(start.strip())
    end = _strip_quotes(end.strip())
    try:
        if not start:
            start = default_start
        else:
            start = _convert_to_day(start)

        if not end:
            # last full month first day
            today = pd.Timestamp.today().to_pydatetime()
            last_month_first = (today.replace(day=1) - pd.DateOffset(months=1)).date()
            end = date(last_month_first.year, last_month_first.month, last_month_first.day).strftime("%Y-%m-%d")
        else:
            end = _convert_to_day(end)

        return start, end
    except ValueError as e:
        raise ValueError(str(e))

def _overall_bounds_from_periods(periods_csv: str) -> Tuple[str, str]:
    """
    Given a CSV of periods, return the min start and max end.
    """
    csv_clean = _strip_quotes((periods_csv or '').strip())
    periods = [_strip_quotes(p.strip()) for p in csv_clean.split(',') if p.strip()]
    starts: List[str] = []
    ends: List[str] = []
    for p in periods:
        s, e = _period_to_bounds(p)
        starts.append(s)
        ends.append(e)
    if not starts:
        raise ValueError("No valid periods")
    s_min = min(starts)
    e_max = max(ends)
    return s_min, e_max

async def _cmd_mhw_line(cli, line: str) -> None:
    """
    Handle a single '/mhw ...' line using the connected CLI (sync wrapper).
    """

    tokens = line.strip().split()[1:]  # drop '/mhw'
    pargs = _build_args_from_tokens(tokens)

    # Prepare per-period bounds if provided
    bounds_list: List[Tuple[str, str, str]] = []  # (label, start, end)
    if pargs.get("_periods"):
        try:
            csv_clean = _strip_quotes(pargs.get("_periods", "").strip())
            for token in [t for t in csv_clean.split(',') if t.strip()]:
                tok = _strip_quotes(token.strip())
                s, e = _period_to_bounds(tok)
                bounds_list.append((tok, s, e))
        except Exception as e:
            print(f"[period parse warning] {e}", file=sys.stderr)

    # 1) Try MCP mhw_query (excluding plotting-only args)
    base_args = {k: v for k, v in pargs.items() if not k.startswith("_")}
    res = None
    if not bounds_list:
        res = await _call_mcp_tool("mhw_query", base_args)

    # 2) If MCP not available, or we need multi-period fetches, HTTP/MCP per-period fallback
    if res is None or bounds_list:
        params_base = {k: v for k, v in pargs.items() if k in {"lon0","lat0","lon1","lat1","append"}}
        out_meta: List[Dict[str, Any]] = []
        all_records: List[Dict[str, Any]] = []
        # If no bounds_list, derive one from provided start/end or overall
        if not bounds_list:
            s = pargs.get("start")
            e = pargs.get("end")
            if not (s and e) and pargs.get("_periods"):
                s, e = _overall_bounds_from_periods(pargs.get("_periods", ""))
            if not (s and e):
                # leave empty to trigger defaults server-side
                pass
            else:
                bounds_list = [(f"{s}-{e}", s, e)]

        if pargs.get("output") == "csv":
            # Print a CSV URL per period
            for label, s, e in bounds_list or [("", params_base.get("start"), params_base.get("end"))]:
                p = dict(params_base)
                if s: p["start"] = s
                if e: p["end"] = e
                out = _http_csv(p)
                out_meta.append({"label": label, **{k: v for k, v in out.items()}})
            print(json.dumps(out_meta, ensure_ascii=False, indent=2))
            return

        # JSON fetches per period: try MCP per-period first if client available
        if _CLI is not None and _CLI.client is not None:
            for label, s, e in bounds_list:
                args_i = dict(base_args)
                if s: args_i["start"] = s
                if e: args_i["end"] = e
                args_i["return_raw"] = True
                try:
                    r = await _call_mcp_tool("mhw_query", args_i)
                except Exception:
                    r = None
                if r and isinstance(r, dict) and "data" in r:
                    all_records.extend(r.get("data") or [])
                    # capture meta for printing
                    out_meta.append({"endpoint": r.get("endpoint"), "params": r.get("params"), "label": label})
                else:
                    # fallback to HTTP for that period
                    p = dict(params_base)
                    if s: p["start"] = s
                    if e: p["end"] = e
                    http_res = _http_json(p)
                    out_meta.append({k: v for k, v in http_res.items() if k != "data"} | {"label": label})
                    all_records.extend(http_res.get("data") or [])
        else:
            # No MCP client: HTTP per period
            for label, s, e in bounds_list:
                p = dict(params_base)
                if s: p["start"] = s
                if e: p["end"] = e
                http_res = _http_json(p)
                out_meta.append({k: v for k, v in http_res.items() if k != "data"} | {"label": label})
                all_records.extend(http_res.get("data") or [])

        # Print simple meta summary
        print(json.dumps(out_meta, ensure_ascii=False, indent=2))
        # Plot if requested
        fields = (pargs.get("append") or "sst,sst_anomaly").split(",")
        fields = [f.strip() for f in fields if f.strip()]
        _run_plot(_records_to_df(all_records), pargs, fields)
        return

    # 3) MCP JSON result
    # Print non-data parts for readability
    printable = {k: v for k, v in res.items() if k not in {"data"}}
    print(json.dumps(printable, ensure_ascii=False, indent=2))

    # CSV branch ends here
    if "download_url" in res:
        return

    # If plotting requested, ensure we have raw data; if not, re-fetch via HTTP.
    data = res.get("data")
    if data is None:
        params = res.get("params", {})
        http_res = _http_json(params)
        data = http_res["data"]

    fields = res.get("fields") or (pargs.get("append") or "sst,sst_anomaly").split(",")
    fields = [f.strip() for f in fields if f.strip()]
    _run_plot(_records_to_df(data), pargs, fields)

def install_mhw_command(cli) -> None:
    """
    Attach '/mhw' command to an existing ODBChatClient without modifying its code.
    """

    global _CLI
    _CLI = cli  # so _call_mcp_tool can use the connected FastMCP client

    original = cli._handle_command  # keep reference

    async def _patched_handle(line: str):
        if line.strip().lower().startswith("/mhw"):
            # run synchronously in current thread (HTTP + plotting are sync)
            await _cmd_mhw_line(cli, line)
            return
        await original(line)

    # monkey-patch
    cli._handle_command = _patched_handle  # type: ignore
