# cli/mhw_cli_patch.py
from __future__ import annotations
from typing import Optional, Dict, Any, List, Callable
import json
import sys
import os
import requests
import pandas as pd
import importlib
import importlib.util

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

async def _cmd_mhw_line(cli, line: str) -> None:
    """
    Handle a single '/mhw ...' line using the connected CLI (sync wrapper).
    """

    tokens = line.strip().split()[1:]  # drop '/mhw'
    pargs = _build_args_from_tokens(tokens)

    # 1) Try MCP mhw_query
    res = await _call_mcp_tool("mhw_query", {k: v for k, v in pargs.items() if not k.startswith("_")})

    # 2) If MCP not available, HTTP fallback
    if res is None:
        params = {k: v for k, v in pargs.items() if k in {"lon0","lat0","lon1","lat1","start","end","append"}}
        if pargs.get("output") == "csv":
            out = _http_csv(params)
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return
        else:
            out = _http_json(params)
            data = out["data"]
            print(json.dumps({k: v for k, v in out.items() if k != "data"}, ensure_ascii=False, indent=2))
            # Plot if requested (compute fields from append or default)
            fields = (pargs.get("append") or "sst,sst_anomaly").split(",")
            fields = [f.strip() for f in fields if f.strip()]
            _run_plot(_records_to_df(data), pargs, fields)
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
