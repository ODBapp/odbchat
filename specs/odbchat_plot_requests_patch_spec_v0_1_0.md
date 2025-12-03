# ODBchat Plot Requests Patch Spec v0.1.0

**Goal:**  
Integrate the new `odbViz` viewer/plotter into `odbchat_cli` and the MCP server,
so that:

1. `/mhw` continues to fetch MHW data from ODB APIs, but delegates all plotting
   to `odbViz` (no more local `mhw_plot.py` / `map_plot.py`).
2. `odbchat_cli` also exposes `/view` commands that use `odbViz`, consistent with
   `odbargo_cli`.
3. `router.answer` (MCP tool) can propose **plot actions** via `plan.plot_request`,
   and `odbchat_cli` can execute them as CLI commands.

This document focuses on behavior and integration; it does **not** prescribe
implementation details or prompt engineering.


---

## 1. ODBchat CLI ↔ Viewer Integration (odbViz)

### 1.1 Viewer registration

`odbchat_cli` MUST support registering a viewer plugin (odbViz) similarly to
`odbargo_cli`:

- At first use of `/view` or `/mhw` that requires plotting:
  - Attempt to connect to an already-registered `odbViz` over WS.
  - If none is found, spawn `odbViz` as a stdio plugin.
- Expect the same `plugin.hello_ok` message and capabilities as specified in
  `ODB Viz Spec v0.3.0`.

Once registered, `odbchat_cli` maintains a client object (e.g. `ViewerClient`) that provides:

- `open_records(datasetKey, source, schema, records)`
- `plot(datasetKey, plotConfig)`
- `preview` / `subset` / `export` (for `/view` commands)


### 1.2 Enabling `/view` in `odbchat_cli`

`odbchat_cli` SHOULD expose the same `/view` command family as `odbargo_cli`,
backed by `odbViz`:

- `/view open <path> as <ds>` – optional if local NetCDF files are desired.
- `/view preview <ds> [as <ds2>] [--filter ...] [--bbox ...] [--cols ...]`
- `/view plot <ds> <kind> [plot options...]`
- `/view export <ds> ...`

This allows:

- Power users to re-use all viewer features (timeseries, climatology, map, profile).
- MHW datasets (fetched via `/mhw`) to be examined and re-plotted using `/view`.


---

## 2. `/mhw` Command Behavior with odbViz

### 2.1 External flags: keep `/mhw` backward compatible

The public interface of `/mhw` in `odbchat_cli` remains largely unchanged,
especially for first adoption:

```bash
/mhw \
  --bbox lon0,lat0,lon1,lat1 \
  --fields sst,sst_anomaly \
  --plot {series|map|month} \
  --plot-field sst_anomaly \
  [--periods "YYYYMMDD-YYYYMMDD(,...)"] \
  [--start YYYY-MM] \
  [--end YYYY-MM] \
  [--outfile path] \
  [--map-method {cartopy|basemap|plain}] \
  [--cmap <name>] \
  [--vmin <num>] \
  [--vmax <num>]
````

* `--plot` modes:

  * `series` → timeseries
  * `map` → map
  * `month` → monthly climatology (renamed `climatology` in viewer semantics)
* `--plot-field` stays as the flag name for now (mapped to `y` inside `odbViz`).
* `--periods` remains a **fetch parameter** for `/mhw` (see §2.3).

### 2.2 Internal mapping: `/mhw` → MHW API → odbViz

Internally, `/mhw` does:

1. Parse CLI flags.

2. Use MCP / MHW API tools to fetch data:

   * Single or multiple periods, depending on `--periods`.

3. Convert JSON into a table (e.g. pandas DataFrame).

4. Attach any extra fields needed for plotting, e.g.:

   * `period_label` when `--periods` is used.

5. Choose a `datasetKey`:

   * Temporary (e.g. `_mhw_tmp1`) for one-shot plots, **or**
   * Named via `as ds1` for reuse (see §2.4).

6. Call `open_records(datasetKey, source="mhw", schema, records)` on `odbViz`.

7. Build a `plotConfig` consistent with `ODB Viz Spec v0.3.0`:

   * `kind`:

     * `series` → `"timeseries"`
     * `map` → `"map"`
     * `month` → `"climatology"`
   * `y` ← `--plot-field` (e.g. `"sst_anomaly"`)
   * `groupBy`:

     * Typically `null`, or `"period_label"` if `--periods` was used.
   * `style.cmap`, `style.vmin`, `style.vmax` from CLI flags or defaults.
   * `params.bboxMode`, `params.timeResolution` as hints (optional).

8. Call `plot(datasetKey, plotConfig)` on `odbViz`.

`mhw_plot.py` and `map_plot.py` are no longer required or bundled in
`odbchat_cli`; all plotting is delegated to `odbViz`.

### 2.3 Handling `--periods` via `groupBy`

`--periods` remains a **/mhw-only parameter** and is **not** added to the
`odbViz` API.

Semantics:

* `--periods "20100101-20191231,20200101-20250601"` indicates that the user
  wants to compare multiple time windows on the same plot.
* `/mhw` parses this into a list of periods:

  * Period 1: 2010-01-01 to 2019-12-31
  * Period 2: 2020-01-01 to 2025-06-01

Implementation on CLI side:

1. For each period, call the MHW API (via MCP tool) and fetch its data.

2. For each row in that period’s data, attach a new field:

   ```python
   df["period_label"] = "2010–2019"
   # or "2010-01-01–2019-12-31"; the exact label is up to CLI
   ```

3. Concatenate data from all periods into a single dataset.

4. Use `groupBy="period_label"` in `plotConfig` so that `odbViz` plots one series
   per period (both in `"timeseries"` and `"climatology"` modes).

Effect:

* For `--plot series` + `--periods`, `odbViz` draws multiple time series, each
  labeled by `period_label`.
* For `--plot month` + `--periods`, `odbViz` (in `kind="climatology"`) will:

  * Derive `month` from `timeField`.
  * For each `period_label`, compute month-wise means of `y`.
  * Draw multiple 12-point climatology curves, one per period.

`odbViz` does not know about `--periods`; it only sees:

* a dataset with a `period_label` column.
* `groupBy="period_label"` in the plot request.

### 2.4 Named datasets: `/mhw ... as ds1`

`odbchat_cli` SHOULD support a syntax similar to `odbargo_cli`:

```bash
/mhw [fetch args...] as ds1
```

Semantics:

* `/mhw` fetches MHW data (with `--periods` if provided).
* It calls `open_records(datasetKey="ds1", source="mhw", ...)`.
* It stores `ds1` in CLI state.
* No plot is required unless additional `--plot` options are provided.

Follow-up commands:

```bash
/view preview ds1 --filter ...
/view preview ds1 as ds2 --filter ... --bbox ...
/view plot ds2 timeseries --y sst_anomaly --group-by period_label
```

Optionally, `/mhw` MAY support:

```bash
/mhw [fetch args...] as ds1 --plot series --plot-field sst_anomaly
```

which would be implemented as:

1. Fetch + `open_records(datasetKey="ds1", ...)`
2. Call `plot(kind="timeseries")` on `ds1`.

This creates a **soft migration path**:

* `/mhw` continues to be a “one-shot fetch+plot” command for simple usage.
* Advanced users can treat `/mhw` as a data source and use `/view` for plotting.

---

## 3. Plot Requests from MCP (`router.answer` → CLI)

### 3.1 Existing `router.answer` shape

`router.answer` currently returns a structure like:

```jsonc
{
  "mode": "explain" | "code" | "mcp_tools",
  "text": "Natural language answer...",
  "code": null,
  "citations": [],
  "plan": { ... },
  "mcp": { ... },
  "debug": { ... }
}
```

This remains valid. We only define a convention for `plan.plot_request`.

### 3.2 New field: `plan.plot_request`

When the one-pass router decides that the user primarily wants a plot, it MAY
include:

```jsonc
{
  "mode": "mcp_tools",
  "text": "I'll plot the MHW anomaly map for January 2024 around Taiwan.",
  "plan": {
    "plot_request": {
      "kind": "mhw.map",   // hint: "<source>.<plotKind>"
      "cli_command": "/mhw --bbox 118,21,123,26 --start 2024-01 --end 2024-01 --plot map --plot-field sst_anomaly",
      "notes": "User asked for Taiwan MHW anomaly map for 2024-01."
    }
  }
}
```

Fields:

* `kind`:

  * Informal tag, such as `"mhw.map"`, `"mhw.climatology"`, `"argo.profile"`.
  * For logging and debugging only.
* `cli_command`:

  * A full CLI command string valid in `odbchat_cli`.
  * Typically of the form `/mhw ...` including any `--periods` needed.
* `notes`:

  * Optional free-text remark for debugging.

Future extensions (not required in v0.1.0):

* `viewer`:

  * An optional object containing a partial `plotConfig` for direct viewer use.
  * For v0.1.0 it can be ignored; CLI is the main execution path.

### 3.3 CLI behavior for `plot_request`

Upon receiving `router.answer`:

1. `odbchat_cli` prints `text` to the user as usual.

2. If `plan.plot_request.cli_command` is present:

   * If **auto-plot** is enabled:

     * The CLI executes `cli_command` internally as if the user had typed it.

       * e.g. `/mhw --bbox ... --plot map ...`
   * If auto-plot is disabled:

     * The CLI may show:

       > Suggested command:
       > `/mhw --bbox 118,21,123,26 --start 2024-01 --end 2024-01 --plot map --plot-field sst_anomaly`

3. If execution of `cli_command` fails, the CLI:

   * Reports an error message to the user.
   * Optionally logs the failure for debugging.

Configuration:

* Auto-plot behavior MAY be controlled by:

  * An environment variable (e.g. `ODBCHAT_AUTOPLOT=1`), or
  * A CLI command (e.g. `/config auto-plot on|off`).

This patch does not mandate a specific mechanism; it only defines behavior when
auto-plot is active.

---

## 4. Help Text and Migration Guidelines

### 4.1 `/help /mhw`

Help text SHOULD reflect that:

* `/mhw` still accepts its historical flags, including `--plot`, `--plot-field`,
  `--periods`, `--cmap`, `--vmin`, `--vmax`, etc.

* Internally, plotting is done via `odbViz` and follows a unified viewer model:

  * `--plot series` → timeseries
  * `--plot map` → map
  * `--plot month` → monthly climatology
  * `--plot-field` indicates the data variable to plot (mapped to `y` in viewer).

* For advanced usage, users can:

  * Use `as dsN` to name datasets.
  * Use `/view preview` / `/view plot` for further operations.

### 4.2 `/help /view` in `odbchat_cli`

`/view` help in `odbchat_cli` SHOULD align closely with `odbargo_cli`, with notes
that:

* Any dataset created from `/mhw` with `as dsN` can be inspected with `/view`.

* Plot kinds available:

  * `timeseries`
  * `climatology` (monthly)
  * `map`
  * `profile`

* Plot options (`--y`, `--group-by`, `--agg`, `--bins`, `--cmap`, etc.) share a
  common semantics with `odbargo_cli` and are all implemented by `odbViz`.

---

## 5. Summary

* `odbViz` becomes the shared viewer/plotter for both `odbargo_cli` and
  `odbchat_cli`.
* `/mhw` in `odbchat_cli`:

  * Keeps its existing user-facing flags (including `--periods`).
  * Internally, fetches data via MCP tools and calls `odbViz` via `open_records`

    * `plot`.
  * Uses a `period_label` + `groupBy` pattern to implement multiple-period
    comparisons in both timeseries and climatology plots.
* `odbchat_cli` gains `/view` commands powered by `odbViz`, enabling users to
  treat MHW data like any other dataset.
* The MCP `router.answer` tool gains a `plan.plot_request` field, allowing LLMs
  to suggest and trigger plotting actions as `/mhw` CLI commands.


