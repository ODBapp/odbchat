先講結論：**現在的 odbchat_cli ↔ odbViz 溝通路徑是「CLI 生小孩（PluginClient 起子行程）」的模式**，而你要的是 **「兩個獨立 process，由 odbViz 主動掛上 CLI（跟 odbargo_cli 的 viewer bridge 一樣）」**。
要讓 Codex 朝這個方向改，關鍵是：**把 viewer_client 變成「連到現有 viewer 的橋接器」，而不是「spawn viewer」**，並且讓 `/mhw` 的繪圖走這個 bridge。

下面用「改法步驟＋要改的具體函式」的形式寫，方便丟給 Codex 實作。

---

## 1. 先說明目前設計（給 Codex 的現況診斷）

1. `odbchat_cli.ODBChatClient.main()` 在啟動時會載入 `cli.mhw_cli_patch.install_mhw_command(cli)`，把 `/mhw` 註冊進 CLI。
2. `/mhw` 真正的實作在 `mhw_cli_patch.handle_mhw()`，裡面在需要畫圖時會呼叫 `_plot_with_viewer(big, plot_cfg, bbox_mode=bbox_mode_local)`。
3. `_plot_with_viewer()` 目前是這樣運作：

   * 用 `_get_viewer()` 產生全域的 `PluginClient()` 實例。
   * 呼叫 `ensure_viewer_available(viewer, verbose=True)`，而 `ensure_viewer_available()` 其實就是 `viewer.ensure_started()`，這會 **spawn 一個子行程**：`python -m odbViz.plugin`，透過 stdio 做 NDJSON handshake，收到 `plugin.hello_ok` 後把 `capabilities` 存到 `viewer.capabilities`。
   * 然後 `_plot_with_viewer()` 呼叫 `viewer.open_records(payload)`、`viewer.plot(viewer_payload, on_header, on_binary)`，最後用 `display_plot_window()` 開檔案。

也就是說：**目前的 viewer 是由 `/mhw` 這條路「按需 spawn 子行程」**，完全沒有「odbViz 先啟動，再掛上 CLI」這回事。

---

## 2. 你要的模式（對 Codex 說清楚目標）

我們要改成和 `odbargo_cli ↔ odbViz` 一樣的語意：

1. `odbchat_cli` 啟動時不會自動 spawn viewer，只是「開一個 listener / bridge」，等待有 viewer 連進來。
2. 你另外在同一台機器跑 `python argo/view_entry.py`（或未來的 `odbViz` entry）時，**odbViz 主動連到 CLI**，送出 `plugin.hello_ok` 與 `capabilities`。
3. 一旦掛上來，`odbchat_cli` 印出像：

   ```text
   [view] viewer ready; capabilities = {...}
   ```

   之後 `/mhw` 在需要畫圖時就透過這個已掛載的 viewer 做 `open_records` + `plot`。
4. 若目前沒有 viewer 掛上來，`/mhw` 仍可以：

   * 要嘛 fallback 到 legacy `map_plot.py / mhw_plot.py`（你現在已有這個機制）；
   * 要嘛印出友善訊息：「目前沒有 viewer 掛載」。

> 關鍵：**odbViz 的生命週期獨立於 CLI**，隨時可以先起 CLI、再起 viewer；或先起 viewer、再起 CLI，只要兩邊都在，就很快完成互相註冊。

---

## 3. 建議的實作策略（給 Codex 的具體修改步驟）

### Step 0. 不要再讓 `/mhw` 自己 new `PluginClient()`

現在的 `_plot_with_viewer()` 直接呼叫 `_get_viewer()` 產生 `PluginClient`，這是「spawn 小孩」模式。

> 建議：改成 **由 CLI 持有 viewer 物件**，`/mhw` 只透過 `cli.viewer` 這種抽象來呼叫。

要做的事情：

1. 把 `_viewer_client` 全域變數和 `_get_viewer()` 函式看作「舊路徑」，未來只保留給「spawn fallback」（如果你還想保留）。

2. 把 `_plot_with_viewer(df, plot_cfg, *, bbox_mode)` 的介面改成：

   ```python
   async def _plot_with_viewer(cli, df: pd.DataFrame, plot_cfg: Dict[str, Any], *, bbox_mode: str) -> bool:
       viewer = getattr(cli, "viewer", None)
       ...
   ```

   然後在 `handle_mhw()` 裡改成：

   ```python
   used_view = await _plot_with_viewer(cli, big, plot_cfg, bbox_mode=bbox_mode_local)
   ```

3. 在 `_plot_with_viewer()` 裡，如果 `viewer is None` 或 `viewer.capabilities` 還沒準備好，就直接 `return False`，讓後面的 legacy plot branch 繼續跑。

這樣 `/mhw` 跟「viewer 的取得方式」就解耦了，接下來就能把重點放在「怎麼讓 `cli.viewer` 被一個外部的 odbViz 實作填進來」。

---

### Step 1. 在 ODBChatClient 裡加一個 viewer slot + 註冊 API

在 `ODBChatClient.__init__` 加上兩個欄位：

```python
self.viewer = None          # type: Optional[SomeViewerInterface]
self.viewer_caps = {}       # type: Dict[str, Any]
```

並加一個 public 方法給外部（或內部 bridge）註冊 viewer：

```python
def attach_viewer(self, viewer, capabilities: Dict[str, Any]) -> None:
    self.viewer = viewer
    self.viewer_caps = capabilities or {}
    print(f"[view] viewer ready; capabilities = {self.viewer_caps}")
```

如果需要，也可以加 `detach_viewer()` 在 viewer 斷線時清掉。

> **給 Codex 的提示**：可以仿照 `ensure_viewer_available()` 在 `viewer_client.py` 取得 `capabilities` 的方式，只是這次不是 `PluginClient.ensure_started()`，而是來自「外部連線的 handshake」。

---

### Step 2. 把 `PluginClient` 轉成「可重用的 viewer 介面」，而不是一定 spawn

`viewer_client.PluginClient` 現在是「內建 spawn + stdio NDJSON client」：

* `ensure_started()` 裡面決定 `_launch_cmd` 然後 `create_subprocess_exec(...)`。
* `_request()` 把 payload 寫到 `stdin`，從 `stdout` 讀 NDJSON / binary blob。

我們想做的是：

1. **把 `PluginClient` 拆成兩層**：

   * 底層是一個通用的「request/response viewer interface」，有這幾個 async 方法：

     * `open_dataset(payload)`
     * `open_records(payload)`
     * `preview(payload)`
     * `plot(payload, on_header, on_binary)`
   * 「spawn 型」只是這個介面的其中一個實作（保留現在的 stdio 版邏輯即可，改名為 `StdioPluginClient` 之類）。

2. 再新增一個「已連線 viewer」版本，代表「某個外部啟動的 odbViz 已經透過 WebSocket / pipe 接上來」：

   * 這個實作可以沿用 Argo 專案中給 `odbargo_cli` 用的那一份 viewer bridge（Codex 可以直接去那邊抄概念）。
   * 比方說定義一個 `AttachedViewer` 類別，內部包著「跟 odbViz 的 WebSocket 連線」，然後提供 `open_records(...)` / `plot(...)` API。

3. 在 ODBchat 裡，`cli.viewer` 應該指向「一個已 attach 的 viewer 物件」（`AttachedViewer`），**而不是 `PluginClient()`**。
   如果你仍然想保留「沒有外部 viewer 時自動 spawn 一個」的能力，可以在 attach 失敗時才 new `StdioPluginClient`，但那是「第二優先」。

給 Codex 的 pseudo-code 指南：

```python
class BaseViewer:
    async def open_records(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...
    async def plot(...): ...

class StdioViewer(BaseViewer):  # 現在的 PluginClient 改名＋整理
    ...

class AttachedViewer(BaseViewer):
    # 用 websocket / pipe 把 open_records / plot 轉送給已啟動的 odbViz
    ...
```

---

### Step 3. 建立「viewer attach server」，讓 odbViz 主動連進來

這部份就直接對 Codex 說：「**仿照 odbargo_cli 的作法**」。

大致步驟：

1. 在 `odbchat_cli.py` 裡（或新的模組 `viewer_bridge.py`）啟動一個小型 asyncio server（WebSocket 或 TCP），例如在 `ODBChatClient.connect()` 成功後啟動一個背景 task：

   ```python
   self._viewer_server_task = asyncio.create_task(run_viewer_server(self))
   ```

2. `run_viewer_server(self)` 的行為類似：

   * 在某個固定 port / 路徑（可用 env `ODBCHAT_VIEW_PORT` 或 `ODB_VIEW_BRIDGE_URL` 控制）接受連線。

   * 第一個 message 必須是 NDJSON 的 handshake：

     ```json
     {"type": "plugin.hello_ok", "capabilities": {...}}
     ```

   * 收到後，用上述 `AttachedViewer` 包起來，然後呼叫 `self.attach_viewer(...)`，印出 capabilities。

   * 之後任何從 CLI 發出的 `open_records` / `plot`，都透過這個 `AttachedViewer` 發送到 viewer side。

3. odbViz 那邊（`view_entry.py` / `plugin.py`）本來就有一套 WebSocket / stdio 介面可供 Argo 用，Codex 只要**沿用同一個 handshake & message 格式**就可以同時支援 `odbargo_cli` 和 `odbchat_cli`。

> 關鍵是：**不要再要求使用者改 PYTHONPATH 或從 CLI 去 spawn viewer**；而是在 CLI 啟動時就把「viewer attach server」跑起來，讓 `python argo/view_entry.py` 主動來註冊。

---

### Step 4. 更新 `/mhw` → viewer 呼叫流程

在完成 Step 1–3 後，`mhw_cli_patch.handle_mhw()` 要做的修改就很少：

1. 把 `_plot_with_viewer()` 改成接受 `cli`，用 `cli.viewer`：

   ```python
   async def _plot_with_viewer(cli, df, plot_cfg, *, bbox_mode):
       viewer = getattr(cli, "viewer", None)
       if not viewer:
           return False
       # 把現在組的 payload 原樣丟給 viewer
       ...
       await viewer.open_records(open_payload)
       await viewer.plot(plot_payload, on_header=_on_header, on_binary=_on_binary)
   ```

2. `handle_mhw()` 中維持現有邏輯，只是改呼叫方式：

   ```python
   used_view = await _plot_with_viewer(cli, big, plot_cfg, bbox_mode=bbox_mode_local)
   if used_view:
       return
   # fallback legacy plotters (現有的 _plot_series/_plot_month/_plot_map)
   ```

這樣 `/mhw` 的旗標設計（`--bbox`, `--plot`, `--plot-field`, `--cmap`, `--vmin/vmax`, `--periods` 等）都可以保留，畫圖本身交給 odbViz 的 `map/timeseries/profile/climatology` 實作。

---

### Step 5. 確認使用流程符合你的需求

最後給 Codex 一個「驗收腳本」：

1. **同一台機器，先開 CLI：**

   ```bash
   cd ~/proj/odbchat
   python -m cli.odbchat_cli --debug
   ```

   * CLI 啟動後會啟動 MCP 連線＆ viewer attach server，但此時沒有 viewer 挂上來，`/mhw` 若請它畫圖，會 fallback 到 legacy plotters。

2. **在另一個 shell 啟動 odbViz：**

   ```bash
   cd ~/proj/argo
   python view_entry.py  # 或你既有的 entry
   ```

   * viewer 啟動後會主動連到 CLI 的 viewer server，送 `plugin.hello_ok`。
   * CLI 應該立刻印出：

     ```text
     [view] viewer ready; capabilities = {...}
     ```

3. **回到 CLI 測試：**

   ```text
   /mhw --bbox 135,-25,-60,25 --fields sst_anomaly --plot map \
        --plot-field sst_anomaly --start 2007-12 --map-method basemap \
        --cmap coolwarm --vmin -3 --vmax 3
   ```

   * `/mhw` 會照舊呼叫 MCP 抓 MHW JSON，組成 dataframe，呼叫 `_plot_with_viewer(cli, ...)`。
   * `_plot_with_viewer()` 用 `cli.viewer` → `viewer.open_records` & `viewer.plot` → 顯示圖片。
   * 如果此時你把 odbViz 關掉，再跑一次 `/mhw`，應該會自動 fallback 到 legacy `map_plot` 行為（或給出「viewer 未掛載」訊息）。

---

這樣 Codex 就有一份「**從現況到目標的具體重構計畫**」，包含：

* 要改的檔案與函式 (`ODBChatClient`, `PluginClient`, `_plot_with_viewer`, `handle_mhw`)。
* 哪些東西要抽象化（viewer interface）、哪些要變成 attach server、哪些保留為 fallback。

