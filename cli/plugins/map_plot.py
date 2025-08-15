# cli/plugins/map_plot.py
from __future__ import annotations
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

plt.rcParams["figure.raise_window"] = False

# ---------------- backend selection (keep your version) ----------------

def _backend(prefer: str | None) -> str:
    """
    Return 'cartopy' | 'basemap' | 'plain'.
    - If prefer is provided: honor it; error if not installed.
    - Else: try cartopy, then basemap, else plain.
    """
    if prefer:
        p = prefer.lower()
        if p == "cartopy":
            try:
                import cartopy.crs as ccrs  # noqa
                import cartopy.feature as cfeature  # noqa
                from cartopy.io import DownloadWarning
                warnings.simplefilter("ignore", DownloadWarning)
                return "cartopy"
            except Exception as e:
                raise RuntimeError("Cartopy not available. Install 'cartopy' or choose --map-method basemap/plain.") from e
        if p == "basemap":
            try:
                from mpl_toolkits.basemap import Basemap  # noqa
                return "basemap"
            except Exception as e:
                raise RuntimeError("Basemap not available. Install 'basemap' or choose --map-method cartopy/plain.") from e
        if p == "plain":
            return "plain"
        raise RuntimeError(f"Unknown --map-method: {prefer}")

    try:
        import cartopy.crs as ccrs  # noqa
        import cartopy.feature as cfeature  # noqa
        from cartopy.io import DownloadWarning
        warnings.simplefilter("ignore", DownloadWarning)
        return "cartopy"
    except Exception:
        try:
            from mpl_toolkits.basemap import Basemap  # noqa
            return "basemap"
        except Exception:
            return "plain"

# ---------------- helpers ----------------

def _delta_lon_shortest(l0: float, l1: float) -> float:
    d = ((l1 - l0 + 180.0) % 360.0) - 180.0
    return abs(d)

def _default_cmap_norm(field: str):
    if field == "sst":
        return "turbo", mcolors.Normalize(vmin=0, vmax=35)
    if field == "sst_anomaly":
        return "RdYlBu_r", mcolors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    if field == "level":
        colors = ["#ffffff","#f5c268","#ec6b1a","#cb3827","#7f1416","#4b0f13"]  # 0..5
        cmap = mcolors.ListedColormap(colors, name="mhw_levels")
        norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5,5.6,1.0), ncolors=cmap.N)
        return cmap, norm
    if field == "td":
        return "BrBG", mcolors.TwoSlopeNorm(vmin=-2000, vcenter=0, vmax=2000)  # tip: set --vmin 0
    return "viridis", None

def _pick_month(df: pd.DataFrame, start: Optional[str]) -> pd.Timestamp:
    if "date" not in df.columns:
        raise ValueError("DataFrame must include 'date'")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    if start:
        for fmt in ("%Y-%m-%d","%Y%m%d","%Y-%m","%Y%m","%Y"):
            try:
                dt = pd.to_datetime(start, format=fmt); return pd.Timestamp(dt.year, dt.month, 1)
            except Exception: pass
        dt = pd.to_datetime(start); return pd.Timestamp(dt.year, dt.month, 1)
    d0 = df["date"].min(); return pd.Timestamp(d0.year, d0.month, 1)

def _grid_month(df: pd.DataFrame, field: str, month: pd.Timestamp, lon_mode: str = "native"):
    sel = df[df["date"].dt.to_period("M") == month.to_period("M")].copy()
    if sel.empty:
        raise ValueError(f"No data for month {month.strftime('%Y-%m')}")
    if lon_mode == "360":
        sel["lon_plot"] = sel["lon"].apply(lambda x: x + 360 if x < 0 else x)
    else:
        sel["lon_plot"] = sel["lon"]
    lats = np.sort(sel["lat"].unique())
    lons = np.sort(sel["lon_plot"].unique())
    Z = np.full((lats.size, lons.size), np.nan)
    lat_idx = {v:i for i,v in enumerate(lats)}
    lon_idx = {v:i for i,v in enumerate(lons)}
    for lo, la, val in sel[["lon_plot","lat",field]].dropna().itertuples(index=False):
        Z[lat_idx[la], lon_idx[lo]] = float(val)
    LON, LAT = np.meshgrid(lons, lats)
    return LON, LAT, Z

# ---------------- main ----------------

def plot_map(
    df: pd.DataFrame,
    field: str = "sst_anomaly",
    bbox: Tuple[float,float,Optional[float],Optional[float]] | None = None,
    start: Optional[str] = None,
    outfile: Optional[str] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    method: Optional[str] = None,
) -> Optional[str]:
    """
    Render a monthly map for one variable.

    field: sst | sst_anomaly | level | td
    method: cartopy | basemap | plain

    Cmaps:
      - sst: turbo, viridis, cividis
      - sst_anomaly: RdYlBu_r, coolwarm, seismic
      - level: fixed palette
      - td: viridis, BrBG, PuOr, RdYlGn  (tip: set --vmin 0)
    """
    need = ("date","lon","lat",field)
    if any(c not in df.columns for c in need):
        raise ValueError("DataFrame must include columns: date, lon, lat, and the chosen field.")

    month = _pick_month(df, start)

    if bbox:
        lon0, lat0, lon1, lat1 = bbox
    else:
        lon0, lon1 = float(df["lon"].min()), float(df["lon"].max())
        lat0, lat1 = float(df["lat"].min()), float(df["lat"].max())
    wraps = (lon0 is not None and lon1 is not None and _delta_lon_shortest(lon0, lon1) > 180.0)

    backend = _backend(method)
    lon_mode = "native" if backend=="cartopy" else ("360" if wraps else "native")
    LON, LAT, Z = _grid_month(df, field, month, lon_mode=lon_mode)

    default_cmap, default_norm = _default_cmap_norm(field)
    if cmap is None: cmap = default_cmap
    vkw = {}
    if default_norm is not None and (vmin is None and vmax is None):
        vkw["norm"] = default_norm
    else:
        if vmin is not None: vkw["vmin"] = vmin
        if vmax is not None: vkw["vmax"] = vmax

    # CARTOPY
    if backend == "cartopy":
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.PlateCarree(central_longitude=180) if wraps else ccrs.PlateCarree()
        data_crs = ccrs.PlateCarree()  # data in [-180,180]
        fig = plt.figure(figsize=(10, 4.6))
        ax = plt.axes(projection=proj)
        if bbox:
            ax.set_extent([lon0, lon1, lat0, lat1], crs=data_crs)
        try:
            ax.coastlines(resolution="110m", linewidth=0.7)
            ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="lightgray")
        except Exception:
            pass
        pcm = ax.pcolormesh(LON, LAT, Z, cmap=cmap, transform=data_crs, shading="auto", **vkw)
        cbar = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, fraction=0.07)
        cbar.set_label({"sst":"SST (°C)","sst_anomaly":"SST Anomaly (°C)","level":"MHW Level","td":"Thermal Displacement (km)"}[field])
        ax.set_title(f"{field} — {month.strftime('%Y-%m')}")
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile, dpi=150); plt.close(fig); return outfile
        plt.show(block=False)
        try: fig.canvas.flush_events()
        except Exception: pass
        return None

    # BASEMAP (use contourf — not pcolormesh — per your preference)
    if backend == "basemap":
        from mpl_toolkits.basemap import Basemap
        fig, ax = plt.subplots(figsize=(10, 4.6))

        if wraps:
            llon = lon0 if lon0 >= 0 else lon0 + 360
            ulon = lon1 + 360 if lon1 < 0 else lon1
            m = Basemap(projection="cyl", llcrnrlon=llon, urcrnrlon=ulon,
                        llcrnrlat=lat0, urcrnrlat=lat1, resolution="l", ax=ax)
        else:
            m = Basemap(projection="cyl", llcrnrlon=lon0, urcrnrlon=lon1,
                        llcrnrlat=lat0, urcrnrlat=lat1, resolution="l", ax=ax)

        m.drawcoastlines(linewidth=0.7)
        try: m.fillcontinents(color="lightgray")
        except Exception: pass

        # Build levels for contourf if using norm-less numeric limits
        if "norm" in vkw:
            # choose ~21 evenly spaced contour levels around the norm’s domain if possible
            if hasattr(vkw["norm"], "vmin") and hasattr(vkw["norm"], "vmax") and vkw["norm"].vmin is not None and vkw["norm"].vmax is not None:
                lv = np.linspace(vkw["norm"].vmin, vkw["norm"].vmax, 21)
            else:
                lv = 21
            cs = m.contourf(LON, LAT, Z, levels=lv, cmap=cmap, latlon=True, extend="both")
        else:
            if vmin is not None and vmax is not None:
                levels = np.linspace(vmin, vmax, 21)
                cs = m.contourf(LON, LAT, Z, levels=levels, cmap=cmap, latlon=True, extend="both")
            else:
                cs = m.contourf(LON, LAT, Z, cmap=cmap, latlon=True, levels=21, extend="both")

        cbar = plt.colorbar(cs, orientation="horizontal", pad=0.05, fraction=0.07)
        cbar.set_label({"sst":"SST (°C)","sst_anomaly":"SST Anomaly (°C)","level":"MHW Level","td":"Thermal Displacement (km)"}[field])
        plt.title(f"{field} — {month.strftime('%Y-%m')}")
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile, dpi=150); plt.close(fig); return outfile
        plt.show(block=False)
        try: fig.canvas.flush_events()
        except Exception: pass
        return None

    # PLAIN fallback
    fig, ax = plt.subplots(figsize=(10, 4.6))
    im = ax.pcolormesh(LON, LAT, Z, cmap=cmap, shading="auto", **vkw)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"{field} — {month.strftime('%Y-%m')} (no coastline)")
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, fraction=0.07)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150); plt.close(fig); return outfile
    plt.show(block=False)
    try: fig.canvas.flush_events()
    except Exception: pass
    return None
