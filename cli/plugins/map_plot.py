# cli/plugins/map_plot.py
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams["figure.raise_window"] = False

# ------------------------------------------------------------
# Backend selection (UNCHANGED – your original)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
LEVEL_COLORS = ["#f5c268", "#ec6b1a", "#cb3827", "#7f1416"]  # 1..4
LEVEL_BOUNDS = [0.5, 1.5, 2.5, 3.5, 4.5]
LEVEL_LABELS_LONG = ["Moderate (1)", "Strong (2)", "Severe (3)", "Extreme (4)"]
LEVEL_LABELS_SHORT = ["Mod (1)", "Str (2)", "Sev (3)", "Ext (4)"]
LEVEL_CMAP = mcolors.ListedColormap(LEVEL_COLORS, name="mhw_level_1_4")
LEVEL_NORM = mcolors.BoundaryNorm(LEVEL_BOUNDS, LEVEL_CMAP.N)
LEVEL_CMAP_MASKED = LEVEL_CMAP.with_extremes(bad=(0, 0, 0, 0))  # masked=transparent

@dataclass
class Grid:
    LON: np.ndarray  # 2D lon mesh
    LAT: np.ndarray  # 2D lat mesh
    Z: np.ndarray    # 2D data
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

def _lon180(a: np.ndarray) -> np.ndarray:
    """Normalize longitudes to [-180, 180)."""
    return ((a + 180.0) % 360.0) - 180.0

def _lon360(a: np.ndarray) -> np.ndarray:
    """Normalize longitudes to [0, 360)."""
    return np.mod(a, 360.0)

def _default_cmap_norm(field: str):
    """Defaults for continuous fields; user cmap/vmin/vmax can override."""
    if field == "sst":
        return plt.get_cmap("viridis"), None, None, None
    if field == "sst_anomaly":
        cmap = plt.get_cmap("coolwarm")
        return cmap, None, -3.0, 3.0
    if field == "td":
        return plt.get_cmap("plasma"), None, None, None
    return plt.get_cmap("viridis"), None, None, None

def _resolve_month_from_start(start: Optional[str], df: pd.DataFrame) -> pd.Timestamp:
    if start:
        ts = pd.to_datetime(start, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Unrecognized start format: {start}")
        return pd.Timestamp(ts.year, ts.month, 1)
    if "date" in df.columns:
        ts = pd.to_datetime(df["date"]).sort_values().iloc[0]
        return pd.Timestamp(ts.year, ts.month, 1)
    now = pd.Timestamp.utcnow().normalize()
    return pd.Timestamp(now.year, now.month, 1)

def _grid_month(df: pd.DataFrame, field: str, month_ts: pd.Timestamp, lon_mode: str) -> Grid:
    """
    Build (LON, LAT, Z) for one month.
    lon_mode: 'native' (as-is), '180' (force [-180,180)), '360' (force [0,360))
    """
    g = df.copy()
    if "date" in g.columns:
        g["date"] = pd.to_datetime(g["date"], errors="coerce")
        g = g[g["date"].dt.to_period("M") == month_ts.to_period("M")]

    lon_key = "lon" if "lon" in g.columns else ("longitude" if "longitude" in g.columns else "lon")
    lat_key = "lat" if "lat" in g.columns else ("latitude" if "latitude" in g.columns else "lat")

    raw_lons = np.asarray(g[lon_key].unique())
    raw_lats = np.asarray(g[lat_key].unique())

    if lon_mode == "360":
        lvals = np.sort(_lon360(raw_lons))
        lon_map = {orig: _lon360(orig) for orig in raw_lons}
    elif lon_mode == "180":
        lvals = np.sort(_lon180(raw_lons))
        lon_map = {orig: _lon180(orig) for orig in raw_lons}
    else:  # native
        lvals = np.sort(raw_lons)
        lon_map = {orig: orig for orig in raw_lons}

    lat_vals = np.sort(raw_lats)
    LON, LAT = np.meshgrid(lvals, lat_vals)
    Z = np.full((lat_vals.size, lvals.size), np.nan, dtype=float)

    lon_index = {v: i for i, v in enumerate(lvals)}
    lat_index = {v: i for i, v in enumerate(lat_vals)}
    for _, row in g[[lon_key, lat_key, field]].dropna().iterrows():
        lo = lon_map[row[lon_key]]
        la = row[lat_key]
        i = lat_index[la]
        j = lon_index[lo]
        Z[i, j] = float(row[field])

    return Grid(
        LON=LON, LAT=LAT, Z=Z,
        lon_min=float(np.nanmin(LON)), lon_max=float(np.nanmax(LON)),
        lat_min=float(np.nanmin(LAT)), lat_max=float(np.nanmax(LAT)),
    )

def _am_blocks_from_lon180(LON_180: np.ndarray) -> List[slice]:
    """
    Given LON normalized to [-180,180), detect the 180° seam and return
    one or two column slices that don't cross the seam.
    """
    x = LON_180[0, :]
    x360 = np.mod(x, 360.0)
    d = np.diff(x360)
    seam = np.where(d < -180.0)[0]
    if seam.size:
        j = int(seam[0] + 1)
        return [slice(0, j), slice(j, None)]
    return [slice(None)]

def _recenter_lon_for_plain(LON2: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Recenter longitudes so the **data arc** (complement of largest gap) is centered at 0.
    Returns (Lrec[-180..180), xmin, xmax). (No sorting here; caller will sort & reorder Z.)
    """

    x = LON2[0, :]
    x360 = np.sort(np.unique(np.mod(x, 360.0)))
    if x360.size < 2:
        L = _lon180(LON2)
        return L, float(np.nanmin(L)), float(np.nanmax(L))

    diffs = np.diff(x360)
    wrap_gap = (x360[0] + 360.0) - x360[-1]
    gaps = np.concatenate([diffs, [wrap_gap]])
    k = int(np.argmax(gaps))
    gap = float(gaps[k])                        # largest gap length
    data_len = 360.0 - gap                      # contiguous arc length containing data
    start = x360[(k + 1) % x360.size]           # first lon **after** the gap
    center = (start + data_len / 2.0) % 360.0   # midpoint of data arc

    Lrec = ((LON2 - center + 180.0) % 360.0) - 180.0
    xmin, xmax = -data_len / 2.0, +data_len / 2.0
    return Lrec, xmin, xmax

# Enhanced plotting functions (reuse from previous code)
def _smart_geo_ticks(lo: float, hi: float, coord_type: str = 'lon') -> np.ndarray:
    """
    Generate smart ticks for geographic coordinates.
    For longitude: prioritize 5°, 10°, 15°, 20°, 30°, 45°, 60°, 90° intervals
    For latitude: prioritize 5°, 10°, 15°, 20°, 30° intervals
    """

    span = hi - lo
    if span <= 0 or not np.isfinite(span):
        return np.array([lo])
    
    # Define preferred intervals for each coordinate type
    if coord_type == 'lon':
        preferred_intervals = [5, 10, 15, 20, 30, 45, 60, 90, 180]
    else:  # latitude
        preferred_intervals = [5, 10, 15, 20, 30, 45, 90]
    
    # Find the best interval that gives us reasonable number of ticks (3-8)
    best_interval = None
    for interval in preferred_intervals:
        n_ticks = span / interval
        if 3 <= n_ticks <= 8:
            best_interval = interval
            break
    
    # If no preferred interval works, fall back to nice ticks
    if best_interval is None:
        return _nice_ticks(lo, hi, nticks=5)
    
    # Generate ticks with the chosen interval
    start = np.ceil(lo / best_interval) * best_interval
    end = np.floor(hi / best_interval) * best_interval
    ticks = np.arange(start, end + 0.5 * best_interval, best_interval)
    
    # Ensure we have at least 3 ticks
    if len(ticks) < 3:
        return _nice_ticks(lo, hi, nticks=5)
    
    return ticks

def _find_best_colorbar_position(ax, fig):
    """
    Find the best position for colorbar that doesn't overlap with ticks/labels.
    Returns orientation and positioning parameters.
    """

    # Get current axis position
    pos = ax.get_position()
    
    # Check available space
    right_space = 1 - pos.x1  # Space to the right
    bottom_space = pos.y0     # Space below
    
    # Prefer horizontal colorbar if there's enough space below
    # and the plot is wide enough
    fig_w, fig_h = fig.get_size_inches()
    aspect_ratio = fig_w / fig_h
    
    if bottom_space > 0.15 and aspect_ratio > 1.5:
        return "horizontal", {"pad": 0.12, "fraction": 0.06, "aspect": 30}
    elif right_space > 0.12:
        return "vertical", {"pad": 0.08, "fraction": 0.06, "aspect": 20}
    else:
        # Fallback to horizontal with adjusted spacing
        return "horizontal", {"pad": 0.15, "fraction": 0.07, "aspect": 25}

def _find_best_legend_position_cartopy(ax, fig, is_level: bool = False):
    """
    Find the best position for legend in cartopy plots.
    """

    pos = ax.get_position()
    bottom_space = pos.y0
    right_space = 1 - pos.x1
    
    if is_level:
        # For discrete level legend
        if bottom_space > 0.18:
            return (0.5, -0.14), "upper center", 4  # ncol=4
        elif right_space > 0.15:
            return (1.02, 0.5), "center left", 1   # ncol=1
        else:
            return (0.5, -0.18), "upper center", 4  # ncol=4, more space
    
    return None, None, None

def _find_best_colorbar_position_basemap(ax, fig):
    """
    Find the best position for colorbar in basemap plots.
    """

    pos = ax.get_position()
    bottom_space = pos.y0
    right_space = 1 - pos.x1
    fig_w, fig_h = fig.get_size_inches()
    aspect_ratio = fig_w / fig_h
    
    # For basemap, prefer horizontal colorbar for wide plots
    if bottom_space > 0.15 and aspect_ratio > 1.5:
        return "horizontal", {"pad": 0.12, "fraction": 0.07, "aspect": 30}
    elif right_space > 0.12:
        return "vertical", {"pad": 0.08, "fraction": 0.06, "aspect": 20}
    else:
        return "horizontal", {"pad": 0.15, "fraction": 0.08, "aspect": 25}

def _find_best_legend_position_basemap(ax, fig, is_level: bool = False):
    """
    Find the best position for legend in basemap plots.
    """
    pos = ax.get_position()
    bottom_space = pos.y0
    right_space = 1 - pos.x1
    
    if is_level:
        if bottom_space > 0.18:
            return (0.5, -0.14), "upper center", 4
        elif right_space > 0.15:
            return (1.02, 0.5), "center left", 1
        else:
            return (0.5, -0.18), "upper center", 4
    
    return None, None, None

def _find_best_legend_position(ax, is_level: bool = False):
    """
    Find the best position for legend that doesn't overlap with ticks/labels.
    Returns (bbox_to_anchor, loc) tuple.
    """
    # Get current axis limits and position
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    pos = ax.get_position()
    
    # Check if there's space below the plot
    bottom_space = pos.y0  # Space below the axes
    
    if is_level:
        # For discrete level legend (horizontal layout)
        if bottom_space > 0.15:  # Enough space below
            return (0.5, -0.12), "upper center"
        else:  # Place outside right side
            return (1.02, 0.5), "center left"
    else:
        # For colorbar, we'll handle this separately
        return None, None

# ------------------------------------------------------------
# Public API (original signature preserved)
# ------------------------------------------------------------
def plot_map(
    df: pd.DataFrame,
    field: str = "sst_anomaly",
    bbox: Tuple[float, float, Optional[float], Optional[float]] | None = None,
    start: Optional[str] = None,
    bbox_mode: Optional[str] = 'none',
    outfile: Optional[str] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    method: Optional[str] = None,
) -> Optional[str]:
    """
    Plot a 2D map for one month.

    - 'level' is rendered as 4 discrete categories with a legend (no colorbar).
    - Other fields get a continuous colorbar; user cmap/vmin/vmax are honored.
    """
    backend = _backend(method)
    month_ts = _resolve_month_from_start(start, df)

    # Choose lon_mode for building the grid:
    # - crossing-zero → force [-180,180) for ALL backends (prevents far-end wrap in plain/basemap)
    # - antimeridian → basemap/cartopy are fine with native; we’ll split at seam when needed
    if bbox_mode == "crossing-zero":
        lon_mode = "180"
    else:
        lon_mode = "360" if bbox_mode == "antimeridian" and backend != 'plain' else "native"

    # for Debugging:
    # print("Warning: internal plotting backend use: ", backend, " with longitude's range mode: ", lon_mode,
    #       " when detecting mode: ", bbox_mode)
    grid = _grid_month(df, field, month_ts, lon_mode=lon_mode)

    # Colormap / range
    is_level = (field == "level")
    if not is_level:
        cmap_default, norm_default, vmin_default, vmax_default = _default_cmap_norm(field)
        cmap_use = plt.get_cmap(cmap) if isinstance(cmap, str) else (cmap or cmap_default)
        vvmin = vmin if vmin is not None else vmin_default
        vvmax = vmax if vmax is not None else vmax_default
        vkw: dict = {}
        if vvmin is not None or vvmax is not None:
            vkw["vmin"] = vvmin
            vkw["vmax"] = vvmax
        elif norm_default is not None:
            vkw["norm"] = norm_default
    else:
        cmap_use = LEVEL_CMAP
        vkw = {}

    # ---------------- CARTOPY ----------------
    if backend == "cartopy":
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

        # Anti-meridian benefits from 180-centered projection; crossing-zero uses 0-centered
        use_am_proj = (bbox_mode == "antimeridian")
        proj = ccrs.PlateCarree(central_longitude=180) if use_am_proj else ccrs.PlateCarree()
        data_crs = ccrs.PlateCarree()

        # Adjust figure size for colorbar if needed
        fig_width = 10
        fig_height = 4.6
        if not is_level:
            # Add space for colorbar
            orientation, cbar_params = _find_best_colorbar_position(None, plt.figure(figsize=(fig_width, fig_height)))
            if orientation == "vertical":
                fig_width += 1.2

        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = plt.axes(projection=proj)

        # Extent: always from the grid (covers both halves cleanly)
        ax.set_extent([grid.lon_min, grid.lon_max, grid.lat_min, grid.lat_max], crs=data_crs)

        # Store the pcolormesh result for colorbar
        im = None

        if is_level:
            Zm = np.ma.masked_less(grid.Z, 0.5)  # mask "no MHW"
            if bbox_mode == "antimeridian":
                # split to avoid fat wrap cells
                L180 = _lon180(grid.LON)
                for s in _am_blocks_from_lon180(L180):
                    im = ax.pcolormesh(grid.LON[:, s], grid.LAT[:, s], Zm[:, s],
                                transform=data_crs, cmap=LEVEL_CMAP_MASKED, norm=LEVEL_NORM,
                                shading="auto", zorder=1)
            else:
                im = ax.pcolormesh(grid.LON, grid.LAT, Zm, transform=data_crs,
                            cmap=LEVEL_CMAP_MASKED, norm=LEVEL_NORM,
                            shading="auto", zorder=1)
            try:
                ax.add_feature(cfeature.LAND.with_scale("110m"),
                            facecolor="#d3d3d3", edgecolor="none", zorder=3)
            except Exception:
                pass
            ax.coastlines(resolution="110m", linewidth=0.7, zorder=4)

            # Enhanced legend positioning
            fig_w = fig.get_size_inches()[0]
            labels = LEVEL_LABELS_LONG if fig_w >= 7.5 else LEVEL_LABELS_SHORT
            
            import matplotlib.patches as mpatches
            patches = [mpatches.Patch(color=LEVEL_COLORS[i], label=labels[i]) for i in range(4)]
            
            bbox_anchor, loc, ncol = _find_best_legend_position_cartopy(ax, fig, is_level=True)
            
            legend = ax.legend(handles=patches, loc=loc, bbox_to_anchor=bbox_anchor,
                    ncol=ncol, frameon=False, fontsize=(10 if fig_w >= 7.5 else 9),
                    handlelength=1.6, columnspacing=0.8, labelspacing=0.6)
            
        else:
            if bbox_mode == "antimeridian":
                L180 = _lon180(grid.LON)
                for s in _am_blocks_from_lon180(L180):
                    im = ax.pcolormesh(grid.LON[:, s], grid.LAT[:, s], grid.Z[:, s],
                                transform=data_crs, cmap=cmap_use, shading="auto", **vkw)
            else:
                im = ax.pcolormesh(grid.LON, grid.LAT, grid.Z,
                            transform=data_crs, cmap=cmap_use, shading="auto", **vkw)
            try:
                ax.add_feature(cfeature.LAND.with_scale("110m"),
                            facecolor="#d3d3d3", edgecolor="none", zorder=3)
            except Exception:
                pass
            ax.coastlines(resolution="110m", linewidth=0.7, zorder=4)

            # Enhanced colorbar with triangular ends
            orientation, cbar_params = _find_best_colorbar_position(ax, fig)
            
            # Create colorbar with triangular end-style (extend='both' for triangular ends)
            if im is not None:
                # Determine if we need triangular ends based on data range vs colormap range
                data_min, data_max = np.nanmin(grid.Z), np.nanmax(grid.Z)
                cmap_min = vkw.get("vmin", data_min)
                cmap_max = vkw.get("vmax", data_max)
                
                # Use triangular ends if data extends beyond colormap range
                extend = 'neither'
                if data_min < cmap_min and data_max > cmap_max:
                    extend = 'both'
                elif data_min < cmap_min:
                    extend = 'min'
                elif data_max > cmap_max:
                    extend = 'max'
                else:
                    extend = 'both'  # Default to both for better visual appeal
                
                cbar = plt.colorbar(im, ax=ax, orientation=orientation, 
                                extend=extend, **cbar_params)
                
                # Set label
                label_dict = {"sst": "SST (°C)", "sst_anomaly": "SST Anomaly (°C)",
                            "level": "MHW Level", "td": "Thermal Displacement (km)"}
                label = label_dict.get(field, field.replace('_', ' ').title())
                
                if orientation == "horizontal":
                    cbar.set_label(label, labelpad=10)
                else:
                    cbar.set_label(label, rotation=90, labelpad=15)
                
                # Adjust tick label size
                cbar.ax.tick_params(labelsize=9)

        # Smart geographic ticks
        xt = _smart_geo_ticks(grid.lon_min, grid.lon_max, coord_type='lon')
        yt = _smart_geo_ticks(grid.lat_min, grid.lat_max, coord_type='lat')
        
        # Fallback to ensure we have ticks
        if xt.size < 3: 
            xt = _nice_ticks(grid.lon_min, grid.lon_max, nticks=5)
        if yt.size < 3: 
            yt = _nice_ticks(grid.lat_min, grid.lat_max, nticks=5)
        
        ax.set_xticks(xt, crs=data_crs)
        ax.set_yticks(yt, crs=data_crs)
        
        # Use Cartopy formatters for proper degree symbols
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())

        ax.set_title(f"{field} — {month_ts.strftime('%Y-%m')}")
        
        # Adjust layout based on legend/colorbar positioning
        if is_level:
            if bbox_anchor and bbox_anchor[1] < 0:  # Legend below
                plt.subplots_adjust(bottom=0.28)
            elif loc == "center left":  # Legend on right
                plt.subplots_adjust(right=0.85)
            else:
                plt.subplots_adjust(bottom=0.26)
        else:
            if orientation == "horizontal":
                plt.subplots_adjust(bottom=0.25)
            else:
                plt.tight_layout()
        
        if outfile:
            plt.savefig(outfile, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return outfile
        
        plt.show(block=False)
        try:
            fig.canvas.flush_events()
        except Exception:
            pass
        return None  

    # ---------------- BASEMAP ----------------
    if backend == "basemap":
        from mpl_toolkits.basemap import Basemap

        # Adjust figure size for colorbar if needed
        fig_width = 10
        fig_height = 4.6
        if not is_level:
            # Check if we need extra space for vertical colorbar
            temp_fig = plt.figure(figsize=(fig_width, fig_height))
            temp_ax = temp_fig.gca()
            orientation, _ = _find_best_colorbar_position_basemap(temp_ax, temp_fig)
            plt.close(temp_fig)
            if orientation == "vertical":
                fig_width += 1.2

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        m = Basemap(projection="cyl",
                    llcrnrlon=grid.lon_min, urcrnrlon=grid.lon_max,
                    llcrnrlat=grid.lat_min, urcrnrlat=grid.lat_max,
                    resolution="l", ax=ax)
        m.drawcoastlines(linewidth=0.7)
        try:
            m.fillcontinents(color="lightgray")
        except Exception:
            pass

        if is_level:
            cs = m.contourf(grid.LON, grid.LAT, grid.Z,
                       levels=LEVEL_BOUNDS, colors=LEVEL_COLORS, latlon=True, extend="neither")
            
            # Enhanced legend positioning
            fig_w = fig.get_size_inches()[0]
            labels = LEVEL_LABELS_LONG if fig_w >= 7.5 else LEVEL_LABELS_SHORT
            
            import matplotlib.patches as mpatches
            patches = [mpatches.Patch(color=LEVEL_COLORS[i], label=labels[i]) for i in range(4)]
            
            bbox_anchor, loc, ncol = _find_best_legend_position_basemap(ax, fig, is_level=True)
            
            legend = ax.legend(handles=patches, loc=loc, bbox_to_anchor=bbox_anchor,
                      ncol=ncol, frameon=False, fontsize=(10 if fig_w >= 7.5 else 9),
                      handlelength=1.6, columnspacing=0.8, labelspacing=0.6)
        else:
            # Build contour levels
            if vmin is not None or vmax is not None:
                lvmin = vmin if vmin is not None else np.nanmin(grid.Z)
                lvmax = vmax if vmax is not None else np.nanmax(grid.Z)
                levels = np.linspace(lvmin, lvmax, 21)
            else:
                cmap_default, norm_default, vmin_default, vmax_default = _default_cmap_norm(field)
                if norm_default is not None and hasattr(norm_default, "vmin"):
                    levels = np.linspace(norm_default.vmin, norm_default.vmax, 21)
                else:
                    zmin = np.nanmin(grid.Z); zmax = np.nanmax(grid.Z)
                    levels = np.linspace(zmin, zmax, 21) if np.isfinite(zmin) and np.isfinite(zmax) else 21
            
            cs = m.contourf(grid.LON, grid.LAT, grid.Z, levels=levels,
                       cmap=(plt.get_cmap(cmap) if isinstance(cmap, str) else (cmap or _default_cmap_norm(field)[0])),
                       latlon=True, extend="both")
            
            # Enhanced colorbar with better positioning and triangular ends
            orientation, cbar_params = _find_best_colorbar_position_basemap(ax, fig)
            
            # The triangular ends are already handled by extend="both" in contourf
            cbar = plt.colorbar(cs, orientation=orientation, **cbar_params)
            
            # Set label with better formatting
            label_dict = {"sst": "SST (°C)", "sst_anomaly": "SST Anomaly (°C)",
                         "level": "MHW Level", "td": "Thermal Displacement (km)"}
            label = label_dict.get(field, field.replace('_', ' ').title())
            
            if orientation == "horizontal":
                cbar.set_label(label, labelpad=10)
            else:
                cbar.set_label(label, rotation=90, labelpad=15)
            
            # Adjust tick label size
            cbar.ax.tick_params(labelsize=9)

        # Smart geographic ticks using enhanced functions
        xt = _smart_geo_ticks(grid.lon_min, grid.lon_max, coord_type='lon')
        yt = _smart_geo_ticks(grid.lat_min, grid.lat_max, coord_type='lat')
        
        # Fallback to ensure we have ticks
        if xt.size < 3: 
            xt = np.linspace(grid.lon_min, grid.lon_max, 3)
        if yt.size < 3: 
            yt = np.linspace(grid.lat_min, grid.lat_max, 3)
        
        try:
            m.drawmeridians(xt, labels=[0, 0, 0, 1], linewidth=0.3, color="#666666",
                            fontsize=9, dashes=[1, 0])
            m.drawparallels(yt, labels=[1, 0, 0, 0], linewidth=0.3, color="#666666",
                            fontsize=9, dashes=[1, 0])
        except Exception:
            pass

        plt.title(f"{field} — {month_ts.strftime('%Y-%m')}")
        
        # Adjust layout based on legend/colorbar positioning
        if is_level:
            if bbox_anchor and bbox_anchor[1] < 0:  # Legend below
                plt.subplots_adjust(bottom=0.28)
            elif loc == "center left":  # Legend on right
                plt.subplots_adjust(right=0.85)
            else:
                plt.subplots_adjust(bottom=0.26)
        else:
            if orientation == "horizontal":
                plt.subplots_adjust(bottom=0.25)
            else:
                plt.tight_layout()
        
        if outfile:
            plt.savefig(outfile, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return outfile
        
        plt.show(block=False)
        try:
            fig.canvas.flush_events()
        except Exception:
            pass
        return None

    # ---------------- PLAIN ----------------
    # Adjust figure size to accommodate legends/colorbars
    fig_width = 10
    fig_height = 4.6
    if not is_level:  # Add extra width for colorbar
        fig_width += 1.2
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if bbox_mode == "antimeridian":
        # Recenter around the **data arc** midpoint; then sort columns and reorder Z identically.
        Lrec, xmin, xmax = _recenter_lon_for_plain(grid.LON)
        order = np.argsort(Lrec[0, :])
        Lrec = Lrec[:, order]
        Zuse = grid.Z[:, order]
        if is_level:
            Zm = np.ma.masked_less(Zuse, 0.5)
            im = ax.pcolormesh(Lrec, grid.LAT, Zm, cmap=LEVEL_CMAP_MASKED, norm=LEVEL_NORM, shading="auto")
        else:
            im = ax.pcolormesh(Lrec, grid.LAT, Zuse,
                          cmap=(plt.get_cmap(cmap) if isinstance(cmap, str) else (cmap or _default_cmap_norm(field)[0])),
                          shading="auto", **({"vmin": vmin, "vmax": vmax} if (vmin is not None or vmax is not None) else {}))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(grid.lat_min, grid.lat_max)
    else:
        if is_level:
            Zm = np.ma.masked_less(grid.Z, 0.5)
            im = ax.pcolormesh(grid.LON, grid.LAT, Zm,
                          cmap=LEVEL_CMAP_MASKED, norm=LEVEL_NORM, shading="auto")
        else:
            im = ax.pcolormesh(grid.LON, grid.LAT, grid.Z,
                          cmap=(plt.get_cmap(cmap) if isinstance(cmap, str) else (cmap or _default_cmap_norm(field)[0])),
                          shading="auto", **({"vmin": vmin, "vmax": vmax} if (vmin is not None or vmax is not None) else {}))
        ax.set_xlim(grid.lon_min, grid.lon_max)
        ax.set_ylim(grid.lat_min, grid.lat_max)

    # Keep degrees roughly square & add smart ticks
    ax.set_aspect('equal', adjustable='box')
    
    # Use smart geographic ticks
    xt = _smart_geo_ticks(*ax.get_xlim(), coord_type='lon')
    yt = _smart_geo_ticks(*ax.get_ylim(), coord_type='lat')
    
    # Fallback to ensure we have ticks
    if xt.size < 3: 
        xt = _nice_ticks(*ax.get_xlim(), nticks=5)
    if yt.size < 3: 
        yt = _nice_ticks(*ax.get_ylim(), nticks=5)
    
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Handle legends and colorbars
    if is_level:
        # Discrete level legend
        fig_w = fig.get_size_inches()[0]
        labels = LEVEL_LABELS_LONG if fig_w >= 7.5 else LEVEL_LABELS_SHORT
        
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=LEVEL_COLORS[i], label=labels[i]) for i in range(4)]
        
        # Find best position for legend
        bbox_anchor, loc = _find_best_legend_position(ax, is_level=True)
        
        legend = ax.legend(handles=patches, loc=loc, bbox_to_anchor=bbox_anchor,
                  ncol=4 if loc == "upper center" else 1, 
                  frameon=False, fontsize=(10 if fig_w >= 7.5 else 9),
                  handlelength=1.6, columnspacing=0.8, labelspacing=0.6)
        
        # Adjust layout based on legend position
        if loc == "upper center":
            plt.subplots_adjust(bottom=0.25)
        else:
            plt.subplots_adjust(right=0.85)
            
    else:
        # Continuous colorbar for non-level fields
        # Create colorbar with proper positioning
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(field.replace('_', ' ').title(), rotation=90, labelpad=15)
        
        # Ensure colorbar doesn't overlap with anything
        cbar.ax.tick_params(labelsize=9)

    ax.set_title(f"{field} — {month_ts.strftime('%Y-%m')} (no coastline)")
    
    # Final layout adjustment
    if not is_level:
        plt.tight_layout()
    else:
        plt.tight_layout(rect=[0, 0.05 if loc == "upper center" else 0, 
                              0.85 if loc == "center left" else 1, 1])
