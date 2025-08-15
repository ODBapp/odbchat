# cli/plugins/mhw_plot.py
from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["figure.raise_window"] = False

def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("DataFrame must include 'date'")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    return df

def _area_mean(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    API returns 0.25° grid records; reduce to area-mean by date to avoid 'spiky'
    plots (you saw that when every grid cell got plotted).
    """
    use = ["date"] + [c for c in cols if c in df.columns]
    g = df[use].groupby("date", as_index=False).mean(numeric_only=True).sort_values("date")
    return g

def plot_series(df: pd.DataFrame, fields: List[str], title: str = "MHW Time Series", outfile: Optional[str] = None):
    df = _ensure_dt(df)
    fields = [f for f in fields if f in df.columns]
    if not fields:
        fields = [c for c in ("sst","sst_anomaly") if c in df.columns]
    if not fields:
        raise ValueError("No plot fields available.")

    # reduce to area-mean by date
    g = _area_mean(df, fields)

    if set(("sst","sst_anomaly")).issubset(g.columns):
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(g["date"], g["sst"], lw=1.6)
        axes[0].set_ylabel("SST (°C)")
        axes[0].grid(True, alpha=.3)

        axes[1].plot(g["date"], g["sst_anomaly"], lw=1.6)
        axes[1].axhline(0, color="k", lw=.8, alpha=.5)
        axes[1].set_ylabel("SST Anomaly (°C)")
        axes[1].grid(True, alpha=.3)
        axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[1].xaxis.get_major_locator()))
        fig.suptitle(title); fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(11,4))
        for f in fields:
            if f in g.columns:
                ax.plot(g["date"], g[f], label=f, lw=1.6)
        ax.grid(True, alpha=.3); ax.legend(); ax.set_title(title); ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        fig.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=150); plt.close(fig); return outfile
    plt.show(block=False)
    try: fig.canvas.flush_events()
    except Exception: pass
    return None

def plot_month_climatology(df: pd.DataFrame, field: str = "sst", periods: str = "", title: str = "Monthly climatology",
                           outfile: Optional[str] = None):
    df = _ensure_dt(df).copy()
    if field not in df.columns:
        raise ValueError(f"Field '{field}' not found in data.")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # parse requested periods (matches CLI)
    per_list = []
    if not periods:
        y0, y1 = int(df["date"].dt.year.min()), int(df["date"].dt.year.max())
        per_list = [(f"{y0}-{y1}", y0, y1)]
    else:
        for tok in [p.strip() for p in periods.split(",") if p.strip()]:
            if "-" in tok:
                a, b = tok.split("-", 1)
                y0, y1 = int(a[:4]), int(b[:4]); per_list.append((f"{y0}-{y1}", y0, y1))
            else:
                y = int(tok[:4]); per_list.append((str(y), y, y))

    fig, ax = plt.subplots(figsize=(11,5))
    for label, y0, y1 in per_list:
        sub = df[(df["year"] >= y0) & (df["year"] <= y1)]
        if sub.empty: continue
        monthly = sub.groupby("month")[field].mean().reindex(range(1,13))
        ax.plot(range(1,13), monthly.values, label=label, lw=1.6)

    ax.set_xticks(range(1,13)); ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_xlabel("Month"); ax.set_ylabel("SST (°C)" if field=="sst" else "SST Anomaly (°C)")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=.3); fig.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=150); plt.close(fig); return outfile
    plt.show(block=False)
    try: fig.canvas.flush_events()
    except Exception: pass
    return None
