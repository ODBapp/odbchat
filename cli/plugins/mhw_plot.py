# Minimal plotting plugin for ODB MHW (matplotlib + pandas only)
from __future__ import annotations
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt

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
    """
    以時間序列呈現 sst / sst_anomaly 的面平均。
    fields: e.g. ["sst", "sst_anomaly"]
    """

    df = _ensure_datetime(df)

    # Filter only valid fields present in DataFrame
    valid_fields = [f for f in fields if f in df.columns]
    if not valid_fields:
        raise ValueError(f"No valid fields to plot from {fields}")

    nplots = len(valid_fields)
    fig, axes = plt.subplots(
        nplots, 1, figsize=(10, 4 * nplots), sharex=True
    )
    if nplots == 1:
        axes = [axes]  # make iterable

    for ax, field in zip(axes, valid_fields):
        s = _area_mean_by_date(df, field)
        if s.empty:
            continue
        ax.plot(s["date"], s[field], label=field)
        ax.set_ylabel(field)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Date")
    fig.suptitle(title)
    fig.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=150)
        plt.close(fig)
        return outfile
    else:
        plt.show()
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

    import re
    def _parse_period(p: str) -> pd.Series:
        p = p.strip()
        if re.fullmatch(r"\d{4}", p):
            y = int(p)
            return (df["year"] == y)
        m = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", p)
        if m:
            y0, y1 = int(m.group(1)), int(m.group(2))
            return (df["year"] >= y0) & (df["year"] <= y1)
        raise ValueError(f"Invalid period format: {p}")

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
        plt.show()
        return None
