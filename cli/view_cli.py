from __future__ import annotations
import sys
import json
from typing import Any, Dict, List, Optional
import pandas as pd
from .viewer_client import BaseViewer, PluginUnavailableError, PluginMessageError, display_plot_window, PlotResult


def _parse_view_flags(tokens: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "dataset": None,
        "kind": None,
        "x": None,
        "y": None,
        "agg": None,
        "order": None,
        "engine": None,
        "cmap": None,
        "vmin": None,
        "vmax": None,
        "gridded": False,
    }
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--x" and i + 1 < len(tokens):
            out["x"] = tokens[i + 1]; i += 2; continue
        if t == "--y" and i + 1 < len(tokens):
            out["y"] = tokens[i + 1]; i += 2; continue
        if t == "--agg" and i + 1 < len(tokens):
            out["agg"] = tokens[i + 1]; i += 2; continue
        if t == "--order" and i + 1 < len(tokens):
            out["order"] = tokens[i + 1]; i += 2; continue
        if t == "--engine" and i + 1 < len(tokens):
            out["engine"] = tokens[i + 1]; i += 2; continue
        if t == "--cmap" and i + 1 < len(tokens):
            out["cmap"] = tokens[i + 1]; i += 2; continue
        if t == "--vmin" and i + 1 < len(tokens):
            try: out["vmin"] = float(tokens[i + 1])
            except Exception: out["vmin"] = None
            i += 2; continue
        if t == "--vmax" and i + 1 < len(tokens):
            try: out["vmax"] = float(tokens[i + 1])
            except Exception: out["vmax"] = None
            i += 2; continue
        if t == "--gridded":
            out["gridded"] = True; i += 1; continue
        i += 1
    return out

def _coerce_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df2 = df.copy()
    for col in df2.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df2[col]) or pd.api.types.is_object_dtype(df2[col]):
                converted = pd.to_datetime(df2[col], errors="coerce")
                if pd.api.types.is_datetime64_any_dtype(converted):
                    df2[col] = converted.dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return df2.to_dict(orient="records")

async def handle_view(cli, line: str):
    """
    /view subcommands forwarded to attached viewer.
    - plot: /view plot <dataset> <map|timeseries|profile|climatology> [flags]
    - preview: /view preview <dataset> [--limit N]
    - vars: /view vars <dataset>
    - open: /view open <path-or-url> [--key <name>]
    - export: /view export <dataset> csv [--out path]
    - subset: /view subset <dataset> <columns> [--filter DSL]
    """
    parts = line.strip().split()
    if len(parts) < 2:
        print("Usage: /view plot|preview|vars|open ...", file=sys.stderr)
        return
    sub = parts[1].lower()

    if not hasattr(cli, "viewer") or not isinstance(getattr(cli, "viewer"), BaseViewer):
        print("❌ No viewer attached; please start odbViz.", file=sys.stderr)
        return
    viewer: BaseViewer = cli.viewer  # type: ignore

    if sub == "open":
        if len(parts) < 3:
            print("Usage: /view open <path-or-url> [--key name | as name]", file=sys.stderr)
            return
        path = parts[2]
        key = None
        if "--key" in parts:
            idx = parts.index("--key")
            if idx + 1 < len(parts):
                key = parts[idx + 1]
        if "as" in parts:
            idx = parts.index("as")
            if idx + 1 < len(parts):
                key = parts[idx + 1]
        if key is None:
            import os
            stem = os.path.splitext(os.path.basename(path))[0] or "ds1"
            key = stem
        payload = {"path": path}
        if key:
            payload["datasetKey"] = key
        try:
            resp = await viewer.request("open_dataset", payload)
            ds_key = resp.get("datasetKey") or key or "ds1"
            print(f"[view] dataset opened: {ds_key}")
        except Exception as exc:
            print(f"[view] open failed: {exc}", file=sys.stderr)
        return

    if sub == "preview":
        if len(parts) < 3:
            print("Usage: /view preview <dataset> [--limit N] [--cols a,b] [--start ISO] [--end ISO] [as name]", file=sys.stderr)
            return
        dataset = parts[2]
        limit = None
        cols_param = None
        start_param = None
        end_param = None
        as_key = None
        if "--limit" in parts:
            try:
                limit = int(parts[parts.index("--limit") + 1])
            except Exception:
                limit = None
        if "--cols" in parts:
            idx = parts.index("--cols")
            if idx + 1 < len(parts):
                cols_param = [c.strip() for c in parts[idx + 1].split(",") if c.strip()]
        if "--start" in parts:
            idx = parts.index("--start")
            if idx + 1 < len(parts):
                start_param = parts[idx + 1]
        if "--end" in parts:
            idx = parts.index("--end")
            if idx + 1 < len(parts):
                end_param = parts[idx + 1]
        if "as" in parts:
            idx = parts.index("as")
            if idx + 1 < len(parts):
                as_key = parts[idx + 1]
        target_key = as_key or dataset
        payload = {"datasetKey": dataset}  # preview always runs on source
        if limit:
            payload["limit"] = limit
        try:
            df_new = None
            if hasattr(cli, "datasets") and dataset in cli.datasets:
                df_src = cli.datasets[dataset]
                df_new = df_src.copy()
            else:
                # pull a sample from viewer to materialize locally
                try:
                    resp_src = await viewer.request("preview", {"datasetKey": dataset, "limit": 10000})
                    rows_src = resp_src.get("rows") or []
                    cols_src = resp_src.get("columns") or []
                    if cols_src and rows_src:
                        df_new = pd.DataFrame(rows_src, columns=cols_src)
                except Exception:
                    df_new = None

            if df_new is not None:
                if start_param:
                    try:
                        df_new = df_new[pd.to_datetime(df_new["date"]) >= pd.to_datetime(start_param)]
                    except Exception:
                        pass
                if end_param:
                    try:
                        df_new = df_new[pd.to_datetime(df_new["date"]) <= pd.to_datetime(end_param)]
                    except Exception:
                        pass
                if cols_param:
                    try:
                        df_new = df_new[cols_param]
                    except Exception:
                        pass
                records = _coerce_records(df_new)
                await viewer.open_records({
                    "datasetKey": target_key,
                    "source": "mhw",
                    "schema": {"timeField": "date", "lonField": "lon", "latField": "lat"},
                    "records": records,
                })
                try:
                    cli.datasets[target_key] = df_new
                except Exception:
                    pass

            resp = await viewer.request("preview", payload)
            rows = resp.get("rows") or []
            cols = resp.get("columns") or []
            print(f"[view] preview {target_key} ({len(rows)} rows):")
            df_resp = None
            if cols and rows:
                import pandas as pd
                df_resp = pd.DataFrame(rows, columns=cols)
                print(df_resp)
            if as_key and df_resp is not None:
                # cache and register target dataset in viewer
                try:
                    cli.datasets[target_key] = df_resp
                except Exception:
                    pass
                try:
                    records_new = _coerce_records(df_resp)
                    await viewer.open_records({
                        "datasetKey": target_key,
                        "source": "mhw",
                        "schema": {"timeField": "date", "lonField": "lon", "latField": "lat"},
                        "records": records_new,
                    })
                except Exception:
                    pass
        except Exception as exc:
            print(f"[view] preview failed: {exc}", file=sys.stderr)
        return

    if sub in {"vars", "list_vars"}:
        if len(parts) < 3:
            print("Usage: /view vars <dataset>", file=sys.stderr)
            return
        dataset = parts[2]
        try:
            resp = await viewer.request("list_vars", {"datasetKey": dataset})
            print(json.dumps(resp, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"[view] vars failed: {exc}", file=sys.stderr)
        return

    if sub == "export":
        if len(parts) < 4:
            print("Usage: /view export <dataset> csv [--out path]", file=sys.stderr)
            return
        dataset = parts[2]
        fmt = parts[3].lower()
        if fmt != "csv":
            print("Only csv export is supported in this bridge.", file=sys.stderr)
            return
        out_path = None
        if "--out" in parts:
            idx = parts.index("--out")
            if idx + 1 < len(parts):
                out_path = parts[idx + 1]
        # ensure dataset is loaded to viewer
        if hasattr(cli, "datasets") and dataset in cli.datasets:
            records = _coerce_records(cli.datasets[dataset])
            try:
                await viewer.open_records({
                    "datasetKey": dataset,
                    "source": "mhw",
                    "schema": {"timeField": "date", "lonField": "lon", "latField": "lat"},
                    "records": records,
                })
            except Exception:
                pass
        try:
            resp = await viewer.request("export", {"datasetKey": dataset, "format": "csv"})
            data = b""
            meta = {}
            if isinstance(resp, dict):
                data = resp.get("data", b"") or b""
                meta = resp.get("meta") or {}
            if not data:
                print("[view] export produced no data", file=sys.stderr)
                return
            if out_path is None:
                import tempfile, os
                fd, out_path = tempfile.mkstemp(prefix="odbchat_", suffix=".csv")
                os.close(fd)
            with open(out_path, "wb") as fh:
                fh.write(data)
            print(f"[view] export saved to {out_path} (size={len(data)} bytes)")
        except Exception as exc:
            print(f"[view] export failed: {exc}", file=sys.stderr)
        return

    if sub == "subset":
        if len(parts) < 4:
            print("Usage: /view subset <dataset> <columns> [--filter DSL]", file=sys.stderr)
            return
        dataset = parts[2]
        cols = parts[3]
        filt = None
        if "--filter" in parts:
            idx = parts.index("--filter")
            if idx + 1 < len(parts):
                filt = parts[idx + 1]
        payload = {"datasetKey": dataset, "columns": cols.split(",")}
        if filt:
            payload["filter"] = filt
        try:
            resp = await viewer.request("subset", payload)
            print(json.dumps(resp, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"[view] subset failed: {exc}", file=sys.stderr)
        return

    if sub != "plot" or len(parts) < 4:
        print("Usage: /view plot <dataset> <map|timeseries|profile|climatology> [--x ... --y ...]", file=sys.stderr)
        return

    dataset = parts[2]
    kind = parts[3].lower()
    tokens = parts[4:]
    flags = _parse_view_flags(tokens)

    # push records to viewer
    df = None
    if hasattr(cli, "datasets"):
        df = cli.datasets.get(dataset)
    if df is not None:
        records = _coerce_records(df)
        payload_open = {
            "datasetKey": dataset,
            "source": "mhw",
            "schema": {"timeField": "date", "lonField": "lon", "latField": "lat"},
            "records": records,
        }
        try:
            await viewer.open_records(payload_open)
        except (PluginUnavailableError, PluginMessageError) as exc:
            print(f"[view] open_records failed: {exc}", file=sys.stderr)
            return
        except Exception as exc:
            print(f"[view] open_records failed: {exc}", file=sys.stderr)
            return
    else:
        # rely on viewer-side dataset (opened via /view open)
        payload_open = None

    viewer_payload: Dict[str, Any] = {
        "datasetKey": dataset,
        "kind": kind,
        "style": {},
        "params": {"gridded": bool(flags.get("gridded"))},
    }
    if flags.get("x"): viewer_payload["x"] = flags["x"]
    if flags.get("y"): viewer_payload["y"] = flags["y"]
    if flags.get("agg"): viewer_payload["agg"] = flags["agg"]
    if flags.get("order"): viewer_payload["order"] = flags["order"]
    if flags.get("engine"): viewer_payload["engine"] = flags["engine"]
    style = viewer_payload["style"]
    if flags.get("cmap") is not None: style["cmap"] = flags["cmap"]
    if flags.get("vmin") is not None: style["vmin"] = flags["vmin"]
    if flags.get("vmax") is not None: style["vmax"] = flags["vmax"]

    result = PlotResult(header={}, png=b"")

    async def _on_header(header: Dict[str, Any]) -> None:
        result.header = header

    async def _on_binary(data: bytes) -> None:
        result.png = data

    try:
        await viewer.plot(viewer_payload, on_header=_on_header, on_binary=_on_binary)
    except (PluginUnavailableError, PluginMessageError) as exc:
        print(f"[view] plot failed: {exc}", file=sys.stderr)
        return
    except Exception as exc:
        print(f"[view] plot failed: {exc}", file=sys.stderr)
        return

    if not result.png:
        print("[view] plot produced no image", file=sys.stderr)
        return
    tmp_path = display_plot_window(result.png, filename_hint="plot.png")
    print(f"🖼️  Plot ready → {tmp_path}")


def install_view_command(cli):
    cli.register_command("/view", handle_view, help_text="Plot via attached viewer: /view plot <dataset> <kind> [options]")
