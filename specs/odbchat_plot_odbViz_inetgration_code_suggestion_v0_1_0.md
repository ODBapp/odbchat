Yes, please go ahead and implement the passive bridge. The goal is:

> **Make `odbchat_cli` behave like `odbargo_cli`: the viewer attaches over WS/NDJSON, and `/mhw` only uses an already-attached viewer (no default spawn).**

Below is the minimal design I’d like you to implement.

---

## 1. Overall architecture we want

1. `odbchat_cli` runs as usual and starts a **viewer bridge server** (WebSocket) in the background.

2. `odbViz` (via `view_entry.py`) connects to that server and sends a `plugin.register` / `plugin.hello_ok` style handshake, including `capabilities`.

3. `odbchat_cli` stores that viewer instance on `cli.viewer` + `cli.viewer_caps`, and prints something like:

   ```text
   [view] viewer ready; capabilities = {...}
   ```

4. `/mhw` plotting then:

   * uses `cli.viewer` (passive attach) to call `open_records` + `plot`,
   * or, if `cli.viewer` is `None`, falls back to legacy plotting or prints “viewer not mounted”.

Spawn-based `PluginClient` is *optional* and can be kept only as a secondary / debug path if you think it’s useful, but the **default path should be passive attach**.

---

## 2. Reuse the Argo viewer bridge pattern

You don’t need to invent a new protocol; please **mirror what’s already done in `odbargo_cli`**:

* `ArgoCLI.run_server()` spins up a WebSocket server:

  ```python
  self._ws_server = await websockets.serve(self._ws_handler, "localhost", self.port)
  ```

* `_ws_handler()` has a branch that handles `{"type": "plugin.register", ...}`: it

  * validates the token (if any),
  * sets `self._plugin_ws` and `self._plugin_caps`,
  * replies with `{"type": "plugin.register_ok", ...}`,
  * then hands control to `_plugin_reader()` so that only one coroutine reads that WS.

For `odbchat_cli`, please do the same shape:

1. Add a `run_viewer_server()` coroutine that calls `websockets.serve(self._viewer_ws_handler, "localhost", VIEW_PORT)`.

   * `VIEW_PORT` can be a fixed default or read from an env var (e.g. `ODBCHAT_VIEW_PORT`).

2. In `ODBChatClient.__init__` add:

   ```python
   self.viewer = None          # attached viewer object
   self.viewer_caps = {}
   self._viewer_ws = None
   self._viewer_pending = {}   # if you implement request/response correlation
   ```

3. Implement `_viewer_ws_handler()` similar to `ArgoCLI._ws_handler()`:

   * Read JSON messages from the socket.
   * On the first `{"type": "plugin.register", ...}`:

     * Optionally validate a token.
     * Store the socket on `self._viewer_ws`.
     * Store `capabilities` on `self.viewer_caps`.
     * Create an **AttachedViewer** object that wraps this WS and assign it to `self.viewer`.
     * Print `[view] viewer ready; capabilities = ...`.
     * Send `{"type": "plugin.register_ok", ...}`.
     * Then call `_viewer_reader(websocket)` and `return` from `_viewer_ws_handler()` so there is only one reader coroutine.

4. Implement `_viewer_reader()` that continuously consumes viewer→CLI messages and matches them to pending requests (similar to `_plugin_reader()` in `odbargo_cli`). When the WS closes, clear `self.viewer`, `self.viewer_caps` and pending maps and print a “viewer disconnected” message.

The message **framing / NDJSON payloads** between CLI and viewer can be exactly the same as what you already use with the stdio `PluginClient` (type+msgId+payload). The difference is only the transport (WebSocket instead of subprocess pipes).

---

## 3. Expose the attached viewer on `cli.viewer`

Please add a small interface for the attached viewer, so `/mhw` doesn’t need to know about WebSockets:

```python
class BaseViewer:
    async def open_records(self, payload: dict) -> dict: ...
    async def plot(
        self,
        payload: dict,
        on_header: Callable[[dict], Awaitable[None]],
        on_binary: Callable[[bytes], Awaitable[None]],
    ) -> None: ...
```

Then implement:

```python
class AttachedViewer(BaseViewer):
    def __init__(self, ws, capabilities: dict, client: ODBChatClient):
        self._ws = ws
        self.capabilities = capabilities
        self._client = client
        self._counter = 0
        self._pending: Dict[str, asyncio.Future] = {}
```

* `open_records()` should:

  * allocate a new `msgId`,
  * send a JSON message like `{"type": "view.open_records", "msgId": "m1", "payload": ...}`,
  * await the matching response from `_viewer_reader()` (resolve a Future).
* `plot()` should:

  * send `{"type": "view.plot", "msgId": "m2", "payload": ...}`,
  * and then `_viewer_reader()` should drive `on_header` + `on_binary` when it receives `view.plot.header` / `view.plot.binary` messages.

This can very closely follow the existing `viewer_client.PluginClient._request()` & plot mode handling, just adapted to WS rather than stdio.

`ODBChatClient.attach_viewer()` can then simply do:

```python
def attach_viewer(self, ws, capabilities: dict) -> None:
    self.viewer = AttachedViewer(ws, capabilities, self)
    self.viewer_caps = capabilities or {}
    print(f"[view] viewer ready; capabilities = {self.viewer_caps}")
```

---

## 4. Refactor `/mhw` to use `cli.viewer`

In `mhw_cli_patch.py`:

1. Change `_plot_with_viewer` to take `cli` instead of instantiating its own `PluginClient`:

   ```python
   async def _plot_with_viewer(cli, df, plot_cfg, *, bbox_mode: str) -> bool:
       viewer = getattr(cli, "viewer", None)
       if not viewer:
           return False  # no attached viewer; let caller fall back

       # build open_records + plot payloads as you already do
       ...
       await viewer.open_records(open_payload)
       await viewer.plot(plot_payload, on_header=..., on_binary=...)
       return True
   ```

   The existing logic for constructing `records`, `datasetKey`, `plot` payloads can stay almost the same; only the actual “send to viewer” calls change from `PluginClient` to `cli.viewer`.

2. In `handle_mhw()`:

   ```python
   used_view = await _plot_with_viewer(cli, big, plot_cfg, bbox_mode=bbox_mode_local)
   if used_view:
       return
   # else fall back to legacy mhw_plot/map_plot as you already do
   ```

This gives us the behavior we want: if `odbViz` is attached, `/mhw` uses it; if not, we fall back to the old local plotting.

---

## 5. About tests

A pytest that spins up a local WS bridge and asserts the `plugin.register/plugin.register_ok` + a simple `open_records/plot` round-trip would be great, but it’s **optional** compared to getting the bridge working.

If you do write a test, a minimal shape could be:

* Start a dummy “viewer” WS client that:

  * connects to the CLI bridge,
  * sends `{"type": "plugin.register", "capabilities": {"dummy": true}}`,
  * responds to one `view.open_records` with a canned `view.open_records.ok`,
  * responds to one `view.plot` with `view.plot.header` + `view.plot.binary`.
* Assert that:

  * `ODBChatClient.attach_viewer()` was called and `cli.viewer_caps` has `"dummy": true`,
  * `_plot_with_viewer` returns `True` and the plot bytes were received.

---

If you implement the bridge as above (copying the Argo pattern) and refactor `/mhw` to use `cli.viewer`, the user-facing behavior will match what I want: I can start `odbchat_cli` and `odbViz` independently, and as soon as `odbViz` connects, `/mhw` will automatically use it.
