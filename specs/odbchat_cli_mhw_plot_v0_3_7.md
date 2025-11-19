# ODBchat CLI – `/mhw` Plot/Viewer Module (v0.3.7)

**Source:** `cli/mhw_cli_patch.py` (current at v0.3.7)  
**Purpose:** Document the existing `/mhw …` command so we can refactor it into a pluggable CLI module.

---

## 1. High-Level Flow

1. **Registration:** `patch_mhw_command()` (or `install_mhw_command()`) attaches `/mhw` to the CLI dispatcher by calling `cli.register_command("/mhw", handle_mhw, help_text=MHW_HELP)`.
2. **Invocation:** User enters `/mhw …flags…`. The CLI passes the full line to `handle_mhw`.
3. **Flag parsing:** `_parse_mhw_flags()` reads bbox/time/plot flags into a dict of raw values (strings, floats, booleans).
4. **Canonical fetch args:** `_canonical_fetch_arglist()` expands the raw config into one or more MCP requests (splitting periods and BBoxes as needed). Each request is validated with `shared.schemas.canonicalize_fetch/validate_fetch`.
5. **Data fetch:** `_mcp_call()` invokes `mhw_fetch` (or legacy `mhw_query`) via the active MCP client, retrying and normalizing responses.
6. **Output handling:**
   - CSV mode → print JSON blob (raw API output).
   - JSON mode → build pandas DataFrames, print a merged summary, optionally show head (`--raw`), or call a pluggable plotting routine (`series|month|map`).
7. **Plotting:** `_plot_series/_plot_month/_plot_map` dynamically import plotting helpers from `cli.plugins.mhw_plot`, `plugins/mhw_plot.py`, `cli/plugins/map_plot.py`, etc. Missing plugins only emit a stderr warning.

---

## 2. CLI Syntax & Defaults

```
/mhw --bbox lon0,lat0[,lon1,lat1]
     [--start YYYY[-MM[-DD]]] [--end YYYY[-MM[-DD]]]
     [--fields sst,sst_anomaly,level,td]
     [--csv] [--raw]
     [--plot series|month|map] [--plot-field …]
     [--periods "comma-separated period tokens"]
     [--map-method cartopy|basemap|plain]
     [--cmap NAME] [--vmin X] [--vmax Y]
     [--outfile path.png]
```

Key behaviors:

- `--bbox` with two numbers means lon/lat point; four numbers produce a rectangular bbox. Missing bbox defaults to `[118,21,123,26]` (Taiwan).
- `--fields` defaults to `"sst,sst_anomaly"`. Allowed fields: `{sst, sst_anomaly, level, td}`.
- `--start/--end` accept `YYYY`, `YYYY-MM`, `YYYYMM`, `YYYY-MM-DD`, `YYYYMMDD`. When absent, `_latest_12_complete_months()` picks a 12-month window ending before the most recent update.
- `--periods` overrides start/end and may contain `YYYY`, `YYYYMM`, `YYYY-YYYY`, `YYYYMM-YYYYMM`. Each token expands to `(start_iso, end_iso, label)`.
- `--csv` switches API output to CSV (no client plotting). `--raw` prints the first few rows of the merged DataFrame for inspection.
- Plot routing:
  - `--plot series` → multi-field time series.
  - `--plot month` → monthly climatology (requires `--periods` for labels; otherwise uses `_latest_12_complete_months`).
  - `--plot map` → spatial field averaged over `start` window.
  - `--plot-field` selects the primary plot field (e.g., `sst`, `sst_anomaly`, `level`, `td`).
  - `--map-method`, `--cmap`, `--vmin`, `--vmax` feed through to map plugin; values are optional.
  - `--outfile` is passed to plugins to save PNG/SVG.

`MHW_HELP` documents the syntax in the CLI `/help` output.

---

## 3. Period & BBox Expansion

- `_parse_periods_csv(csv, default_start, default_end)`:
  - Parses the user-provided `--periods` CSV list.
  - Each token is normalized via `_norm_month_first` (supports multiple date formats).
  - With no `--periods`, uses `_latest_12_complete_months()` or `start/end` pair.
  - Returns a list of tuples `(start_iso, end_iso, label)`.

- `_split_bbox_by_crossing(lon0, lat0, lon1, lat1)`:
  - Handles antimeridian spans (`|lon0 - lon1| > 180`) and zero-crossing BBoxes by splitting into two requests with epsilon offsets.
  - Stores the last computed mode in `bboxMode` to inform map plotting (displaying antimeridian vs zero-crossing cases).

For each `(period, bbox)` combination, `_canonical_fetch_arglist()` produces a dict:

```json
{
  "lon0": <float>, "lat0": <float>,
  "lon1": <float|null>, "lat1": <float|null>,
  "start": "YYYY-MM-DD",
  "end": "YYYY-MM-DD",
  "append": "comma-separated field list",
  "output": "csv|json",
  "include_data": true|false
}
```

These dictionaries are validated via `validate_fetch` before any MCP call.

---

## 4. MCP Call & Response Handling

- `_mcp_call(cli, "mhw_fetch", args)`:
  - Validates CLI is connected (`cli.client`).
  - Calls the tool; if it fails, falls back to `mhw_query` for backward compatibility.
  - Returns structured dicts by inspecting `FastMCP` response objects (`result.data`, `.structured_content`, `.content text`, `.text`, or fallback to `{"raw": …}`).
  - Errors print a red ❌ message to stderr.

- CSV mode: prints JSON serialization of the raw response(s) (if multiple bbox/period splits, prints a list).
- JSON mode: expects `{"data": [...]}` with each entry containing at least `date, lon, lat`. Converts to pandas via `_records_to_df`, adds a `period` column (`YYYY-MM` label), and concatenates all frames.
- Summary: if only one API response, prints its metadata (minus `data`). Otherwise prints `{"note": "merged N API responses", "components": [...]}`.

---

## 5. Plotting Pipeline

`plot_cfg` is derived from CLI flags, canonicalized via `shared.schemas.canonicalize_plot`, and validated with `validate_plot`. Then:

| Mode | Function | Module search order |
| --- | --- | --- |
| `series` | `_plot_series(df, fields, outfile=…)` | `cli.plugins.mhw_plot`, `plugins/mhw_plot.py`, `cli.mhw_plot`, `mhw_plot` |
| `month` | `_plot_month(df, field, periods, outfile=…)` | same as `series` but calls `plot_month_climatology` |
| `map` | `_plot_map(df, field, bbox, start, bbox_mode, outfile, method, cmap, vmin, vmax)` | `cli.plugins.map_plot`, `plugins/map_plot.py`, `cli.map_plot`, `map_plot` |

`_load_symbol()` implements the plugin loader:

- Tries to import from package modules first (`importlib.import_module`).
- Falls back to loading `.py` files relative to `cli/`.
- Returns `None` if no callable is found; the calling helper prints `[plot plugin missing] …`.

The plotting helpers are responsible for actually rendering (matplotlib/basemap/cartopy) and writing to disk when `outfile` is provided.

---

## 6. Validation & Notices

- Fetch requests are validated per chunk. If any chunk fails validation, `/mhw` aborts before calling MCP.
- `_warn_limits` inspects bbox size and time span to remind users about server-side limits (e.g., `>90×90° bbox → ~1 month max`).
- Plot configurations are validated; invalid configs stop the command with `❌ invalid plot config: …`.
- When in JSON mode without plotting, `--raw` prints the first rows for debugging.

---

## 7. Extensibility Hooks

- `patch_mhw_command(cli)` monkey-patches `_handle_command` for legacy CLIs that lack `register_command`. Preferred path is `install_mhw_command(cli)` which calls `register_command` directly.
- Plot functions are intentionally decoupled so future refactors can drop them into a dedicated module (e.g., `cli/extensions/mhw_plot.py`) without touching the command handler.
- Shared schema helpers (`shared.schemas.canonicalize_fetch`, `validate_fetch`, `canonicalize_plot`, `validate_plot`) centralize argument validation; any redesign should preserve these contracts or supply equivalents.

---

This document should serve as the basis for extracting `/mhw` plotting/viewer behavior into a standalone module or package while maintaining CLI compatibility at v0.3.7.
