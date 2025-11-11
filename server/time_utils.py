from __future__ import annotations

from datetime import datetime, timezone

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

_TPE_TZ = ZoneInfo("Asia/Taipei") if ZoneInfo else None


def today_tpe() -> str:
    """Return today's date in Asia/Taipei (YYYY-MM-DD)."""
    tz = _TPE_TZ or timezone.utc
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d")
