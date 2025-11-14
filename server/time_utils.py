from __future__ import annotations

from datetime import datetime, timezone
import os

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

_TPE_TZ = ZoneInfo("Asia/Taipei") if ZoneInfo else None
_DEFAULT_TZ_NAME = os.getenv("ODBCHAT_TZ", "Asia/Taipei")


def today_tpe() -> str:
    """Return today's date in Asia/Taipei (YYYY-MM-DD)."""
    tz = _TPE_TZ or timezone.utc
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d")


def resolve_tz_name(preferred: str | None = None) -> str:
    """Return a safe TZ name, falling back to env default or Asia/Taipei."""
    name = (preferred or "").strip()
    if name:
        return name
    return _DEFAULT_TZ_NAME or "Asia/Taipei"


def _tzinfo_from_name(name: str) -> timezone:
    if ZoneInfo:
        try:
            return ZoneInfo(name)
        except Exception:  # pragma: no cover
            pass
    return timezone.utc


def today_in_tz(tz_name: str | None = None) -> str:
    """Return YYYY-MM-DD string for the given timezone (defaults to env)."""
    name = resolve_tz_name(tz_name)
    tzinfo = _tzinfo_from_name(name)
    now = datetime.now(tzinfo)
    return now.strftime("%Y-%m-%d")


def now_iso_in_tz(tz_name: str | None = None) -> str:
    """Return ISO-8601 timestamp with offset for the given timezone."""
    name = resolve_tz_name(tz_name)
    tzinfo = _tzinfo_from_name(name)
    now = datetime.now(tzinfo)
    return now.isoformat()
