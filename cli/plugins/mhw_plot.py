# Minimal plotting plugin for ODB MHW (matplotlib + pandas only)
from __future__ import annotations
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import re

# Do not auto-raise windows when showing/refreshing figures
plt.rcParams["figure.raise_window"] = False

def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column (YYYY-MM-DD).")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
    return df

def _area_mean_by_date(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    # 對同日期的多格點做面平均
    return (
        df.dropna(subset=[value_col])
          .groupby("date", as_index=False)[value_col]
          .mean()
          .sort_values("date")
    )

def plot_series(
    df: pd.DataFrame,
    fields: List[str],
    title: str = "MHW Time Series",
    outfile: Optional[str] = None,
) -> str | None:
    df = _ensure_datetime(df)

    valid_fields = [f for f in fields if f in df.columns]
    if not valid_fields:
        raise ValueError(f"No valid fields to plot from {fields}")

    # Wide & short (≈12:4 for two panels)
    nplots = len(valid_fields)
    if nplots == 1:
        figsize = (10, 3.2)
    elif nplots == 2:
        figsize = (10, 4.0)
    else:
        figsize = (10, 2.6 * nplots)

    fig, axes = plt.subplots(nplots, 1, figsize=figsize, sharex=True)
    if nplots == 1:
        axes = [axes]

    # date formatter
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    for ax, field in zip(axes, valid_fields):
        s = _area_mean_by_date(df, field)
        if s.empty:
            ax.set_visible(False)
            continue

        x = s["date"]
        if field == "sst_anomaly":
            ax.plot(x, s[field], linewidth=1.4, label=field, color='#fd8000')
            ax.axhline(0.0, color="#666666", linewidth=0.8)
            ax.set_ylabel("SST Anomalies")
        else:
            ax.plot(x, s[field], linewidth=1.4, label=field)
            ax.set_ylabel(f"{'SST' if field == 'sst' else field.apitalize()}")

        ax.legend(loc="upper right", frameon=False)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    axes[-1].set_xlabel("Date")
    fig.suptitle(title)
    fig.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=150)
        plt.close(fig)
        return outfile
    else:
        plt.show(block=False)
        try:
            fig.canvas.flush_events()   # no window raising
        except Exception:
            pass
        return None

def plot_month_climatology(
    df: pd.DataFrame,
    field: str = "sst",
    periods: Optional[List[str]] = None,
    title: str = "Monthly Climatology",
    outfile: Optional[str] = None,
) -> str | None:
    """
    月份氣候平均圖（把資料依 month 聚合）。
    periods:
      - None：用整段資料一組線
      - ["2012-2021","2022","2023"]：可混合單一年份或年份區間
    """
    df = _ensure_datetime(df)
    if field not in df.columns:
        raise ValueError(f"Field '{field}' not in DataFrame.")

    df = df.dropna(subset=[field]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    def _parse_period(p: str) -> pd.Series:
        """
        Accept rich period formats and return a boolean mask over df:
        - YYYY (single year)
        - YYYY-YYYY (year range)
        - YYYYMM or YYYY-MM (single month)
        - YYYYMM-YYYYMM or YYYY-MM-YYYY-MM (month range)
        - YYYYMMDD or YYYY-MM-DD (single day)
        - YYYYMMDD-YYYYMMDD or YYYY-MM-DD-YYYY-MM-DD (day range)
        """
        s = p.strip().strip('"').strip("'")
        # Single year
        if re.fullmatch(r"\d{4}", s):
            y = int(s)
            return (df["year"] == y)
        # Year range
        m = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", s)
        if m:
            y0, y1 = int(m.group(1)), int(m.group(2))
            return (df["year"] >= y0) & (df["year"] <= y1)

        # Build date bounds for other formats
        def _to_date(x: str) -> pd.Timestamp:
            x = x.strip()
            for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m", "%Y%m", "%Y"):
                try:
                    dt = datetime.strptime(x, fmt)
                    # Normalize month/year to first day
                    if fmt in ("%Y-%m", "%Y%m"):
                        dt = dt.replace(day=1)
                    if fmt == "%Y":
                        dt = dt.replace(month=1, day=1)
                    return pd.Timestamp(dt.date())
                except ValueError:
                    continue
            raise ValueError(f"Invalid date format: {x}")

        # Day range / month range including hyphen variants
        rng = re.fullmatch(r"(.+?)\s*-\s*(.+)", s)
        if rng:
            s0, s1 = rng.group(1), rng.group(2)
            d0 = _to_date(s0)
            d1 = _to_date(s1)
            # Ensure d1 >= d0
            if d1 < d0:
                d0, d1 = d1, d0
            return (df["date"] >= d0) & (df["date"] <= d1)

        # Single month/day
        try:
            d = _to_date(s)
            # Single day
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{8}", s):
                return (df["date"] == d)
            # Single month (YYYYMM/ YYYY-MM) or year
            if re.fullmatch(r"\d{4}-\d{2}|\d{6}", s):
                return (df["date"].dt.year == d.year) & (df["date"].dt.month == d.month)
            # Year (already handled above), fallback: match year
            return (df["date"].dt.year == d.year)
        except ValueError:
            raise ValueError(f"Invalid period format: {s}")

    plt.figure(figsize=(10, 5))
    if not periods:
        grp = df.groupby("month", as_index=False)[field].mean().sort_values("month")
        plt.plot(grp["month"], grp[field], label="all")
    else:
        for p in periods:
            mask = _parse_period(p)
            sub = df.loc[mask]
            if sub.empty:
                continue
            grp = sub.groupby("month", as_index=False)[field].mean().sort_values("month")
            plt.plot(grp["month"], grp[field], label=p)

    plt.xticks(range(1,13), ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    plt.xlabel("Month")
    plt.ylabel(field)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=150)
        plt.close()
        return outfile
    else:
        plt.show(block=False)
        try:
            fig.canvas.flush_events()
        except Exception:
            pass
        return None
