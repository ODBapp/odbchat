# ODBchat CLI — MHW 快速繪圖指南

## CLI工具：odbchat_cli《README.md》（繁中），內容涵蓋：

* `odbchat_cli.py` 的使用方式與鍵盤操作（歷史、游標修正、輸出對齊）
* `/mhw` 指令語法、參數、可複製範例
* `mhw_cli_patch.py` 的 bbox 判斷與 `bbox_mode` 傳遞
* `mhw_plot.py`（時序）與 `map_plot.py`（地圖）的設計重點
* 地圖後端（cartopy/basemap/plain）與跨 0°/跨 180° 處理
* level 顏色與圖例規則、連續變量的 cmap/vmin/vmax
* 非阻塞繪圖、刻度與版面調整、常見錯誤排除
* RAG 片段建議與 Cheatsheet

## ODBchat CLI — MHW 快速繪圖指南

> 本文件是針對 **ODBchat** 專案、在「海洋熱浪（MHW）」主題下，指導使用者如何透過 `odbchat_cli.py` 及其 MHW 插件快速下指令、取用 ODB API 並繪製圖表的 **README**。  
> 你可以在此檔案 **最上方** 再加上 RAG 用的 YAML front‑matter（例如 `---\nid: mhw_cli_guides\n...`）後，另存為 `mhw_cli_guides.yml` 供向量化或段落檢索使用。

---

### 1. 目的與範圍

- 讓終端使用者以 **一行指令** 取得 ODB MHW API 結果，並繪製 **時序圖** 或 **地圖**。  
- 提供 **RAG 友善** 的結構、範例與關鍵語彙，方便後續把此內容放入知識庫。  
- 試作主題限定為 **海洋熱浪（Marine Heatwaves, MHW）**，涵蓋欄位：`sst`、`sst_anomaly`、`level`、`td` 等。

---

### 2. 目錄與元件

```
proj/odbchat/
├─ cli/
│  ├─ odbchat_cli.py          # 互動式 CLI
│  ├─ plugins/
│  │  ├─ mhw_cli_patch.py     # MHW 指令解譯與 ODB API 參數整備
│  │  └─ map_plot.py          # 地圖繪圖（cartopy / basemap / plain）
│  └─ ...
├─ mhw_plot.py                # 時序圖繪圖
└─ README.md                  # 本指南（可轉成 mhw_cli_guides.yml）
```

---

### 3. 系統需求

- Python 3.9+（推薦 3.10/3.11）  
- 必備：`numpy`, `pandas`, `matplotlib`  
- 選配地圖後端：
  - `cartopy`（較佳海岸線；需 GEOS/PROJ）
  - `basemap`（相容 contourf；較易安裝）
  - 沒有上述套件時自動退回 `plain`（無投影、快速）

> 後端選擇規則在 `cli/plugins/map_plot.py::_backend()`：
> 1) 若使用者以 `--map-method` 指名則優先；2) 否則依序嘗試 cartopy → basemap → plain。

---

### 4. 啟動方式

在專案根目錄：

```bash
python odbchat_cli.py
```

看到提示後即可輸入命令（任何不以斜線 `/` 起頭的文字會視為一般聊天）。

---

### 5. 互動小技巧（鍵盤操作）

- **歷史**：輸入 `/` 後按 **↑** / **↓** 可循環上一筆/下一筆指令。  
- **編輯**：**←** / **→** 正確移動至當前編輯位置（已修正游標顯示/刪除錯位問題）。  
- **輸出對齊**：CLI 會在 **下一行、第一欄** 開始列印回應，避免縮排跑位。

> 圖窗為 **非阻塞** 顯示：繪圖完成後 CLI 立即可再下指令。不會強迫視窗置頂。

---

### 6. 指令總覽

基本格式：

```
/mhw --bbox lon0,lat0,lon1,lat1      --fields <csv>      --plot {series|map}      --plot-field <csv>      [--periods "YYYYMMDD-YYYYMMDD(,...)"] [--start YYYY-MM] [--end YYYY-MM]      [--outfile path] [--map-method {cartopy|basemap|plain}]      [--cmap <name>] [--vmin <num>] [--vmax <num>]
```

#### 主要參數

- `--bbox lon0,lat0,lon1,lat1`：查詢/繪圖範圍（度，-180~180）。
- `--fields`：API 回傳欄位（例：`sst,sst_anomaly,level`）。
- `--plot`：`series` 時序圖、`map` 地圖。
- `--plot-field`：實際要繪製的欄位，可為 CSV（例：`sst,sst_anomaly`）。
- 期間指定：
  - `--periods "YYYYMMDD-YYYYMMDD,YYYYMMDD-YYYYMMDD"`（可多段；CLI 會逐段請求並合併顯示）
  - 或 `--start YYYY-MM` / `--end YYYY-MM`
- 繪圖選項（地圖連續變量）：`--cmap` / `--vmin` / `--vmax`
- 地圖後端：`--map-method cartopy|basemap|plain`
- 匯出：`--outfile path`（若不指定則直接顯示）

---

### 7. 常用範例（可直接複製）

#### 7.1 多期間時序圖（兩子圖）
```text
/mhw --bbox 119,20,122,23 --fields sst,sst_anomaly      --plot series --plot-field sst,sst_anomaly      --periods "20100101-20191231,20200101-20250601"
```
**行為**：
- `mhw_plot.py` 繪製兩個子圖：上方 `sst`、下方 `sst_anomaly`（寬>高，約 12×4）。  
- `sst_anomaly` 顏色：>0 用 `#f8766d`、<0 用 `#619cff`、=0 灰色，便於判讀。

#### 7.2 單月地圖（自動跨 180°/0°）
```text
/mhw --bbox 135,-25,-60,25 --fields level      --plot map --plot-field level --start 2007-12      --map-method cartopy
```
**行為**：
- `mhw_cli_patch.py` 會偵測 bbox 是否 **跨本初子午線(0°)** 或 **跨反子午線(±180°)**：  
  - 以 `abs(lon0-lon1)>180` 判定 **反子午線**  
  - 兩端異號判定 **跨 0°**  
- 此偵測結果（`bbox_mode`）會傳遞給 `map_plot.py`，在繪圖時採用對應經度模式與縫合邏輯。

---

### 8. 繪圖設計細節

#### 8.1 時序圖（`mhw_plot.py`）
- 版面：兩子圖、共享 X 軸（日期），**寬形**（預設 ~12×4）。
- `sst_anomaly`：以顏色區分正負；0 值灰。

#### 8.2 地圖（`map_plot.py`）
- **離散等級 `level`（1..4）**：
  - 色盤：`["#f5c268","#ec6b1a","#cb3827","#7f1416"]`（Moderate→Extreme）
  - 僅顯示 1–4 的圖例（0=無 MHW 不列入）
  - 圖例在下方置中；窄圖自動換成短標籤（`Mod/Str/Sev/Ext`）避免擁擠
- **連續變量**（如 `sst_anomaly`）：支援 `--cmap`/`--vmin`/`--vmax`，且提供預設配色與合理範圍
- **經度縫合**：
  - `bbox_mode == "crossing-zero"`：內部以 **[-180,180)** 建格避免「遠端帶狀」
  - `bbox_mode == "antimeridian"`：
    - Cartopy：以兩塊 `pcolormesh` 分段繪製
    - Basemap：以 `contourf` 繪製，不跨 seam
    - Plain：先把網格 **旋轉到資料弧中心**，再 **排序欄位** 與 **重排 Z**，確保 `pcolormesh` 單調遞增，不出現垂直接縫或重複
- **軸刻度**：依 **實際座標區塊大小** 動態決定 3–8 個刻度，避免擁擠/太稀
- **版面**：自動增大下邊界，避免圖例/色條與 X 軸重疊

---

### 9. RAG 知識庫建議切片

為了讓助理能「引導使用者組指令」，建議將下列片段納入向量索引：

- **命令模板**：
  - 時序：`/mhw --bbox <…> --fields sst,sst_anomaly --plot series --plot-field sst,sst_anomaly [--periods "..."]`
  - 地圖：`/mhw --bbox <…> --fields <field> --plot map --plot-field <field> --start YYYY-MM [--map-method ...]`
- **參數對照表**與 **範例解說**
- **bbox 判斷邏輯** 與 `bbox_mode` 傳遞（跨 0° / 反子午線）
- **level 色盤與圖例規則**
- **常見錯誤排除**（見下節）

> 產生 `mhw_cli_guides.yml`：在本 README 最上方加上 YAML front‑matter（`--- ... ---`），再以你慣用的切片工具製作。建議以章節作段落切分（H2/H3）。

---

### 10. 常見錯誤與排除

- **`pcolormesh are not monotonically increasing` 警告（plain+反子午線）**  
  代表 X 未單調。新版 `map_plot.py` 會在重心旋轉後 **排序欄位並同步重排 Z**，請確認已更新。
- **地圖刻度過於密集/稀疏**  
  新版已依軸尺寸自動調整；若仍不理想，縮放視窗或改用 `cartopy` 後端。
- **關閉單一圖窗後 CLI 卡住**  
  同行程 GUI 偶有此現象。建議：一次關閉全部圖窗，或未來啟用「外部子行程繪圖」模式（可加環境變數觸發）。
- **圖窗離開 CLI 後一起關閉**  
  目前圖窗與 CLI 同進程，結束 CLI 會連帶關閉。建議先 `--outfile` 存檔。

---

### 11. 參考色標與單位

- `sst`（°C），`sst_anomaly`（°C，預設色盤 `coolwarm`）  
- `level`：1..4（Moderate/Strong/Severe/Extreme）  
- `td`：Thermal Displacement（km）

---

### 12. 版本記錄（重點）

- **CLI 鍵盤體驗**：新增歷史「`/` + ↑/↓」、修正游標位置與刪除；輸出首欄對齊。  
- **多期間**：`--periods` 支援 CSV，多段自動請求與標示。  
- **非阻塞繪圖**：繪圖後不阻塞 CLI；禁用自動置頂。  
- **地圖縫合**：跨 0° 與跨 180° 各自處理；`plain` 後端改為旋轉+排序，避免接縫。  
- **`level` 顏色**：離散 1..4，圖例自動短標籤。  
- **時序設計**：雙子圖、寬畫面、異常值色彩（紅/藍/灰）。

---

### 13. 速查表（Cheatsheet）

```text
# A) 時序：台灣外海、兩段期間
/mhw --bbox 119,20,122,23 --fields sst,sst_anomaly --plot series --plot-field sst,sst_anomaly --periods "20100101-20191231,20200101-20250601"

# B) 地圖：跨 180° 的太平洋，單月 level
/mhw --bbox 135,-25,-60,25 --fields level --plot map --plot-field level --start 2007-12 --map-method cartopy

# C) 地圖：連續變量 + 自訂色階
/mhw --bbox 135,-25,-60,25 --fields sst_anomaly --plot map --plot-field sst_anomaly --start 2007-12 --map-method basemap --cmap coolwarm --vmin -3 --vmax 3

# D) 輸出檔
... --outfile result.png
```

---

### 14. 授權與資料來源

- API 與資料來源：Ocean Data Bank (ODB), Taiwan. www.odb.ntu.edu.tw  
- 本 CLI 與繪圖程式碼以專案授權條款為準，引用可參考: https://www.odb.ntu.edu.tw/odb-services/#service_citations



