"""Shared keyword/phrase sets (and regex helpers) for router and one-pass logic."""

from __future__ import annotations

import re
from typing import Sequence


def _regex(patterns: Sequence[str]) -> re.Pattern:
    joined = "|".join(patterns)
    return re.compile(f"(?:{joined})", re.IGNORECASE)


# General "user wants code" cues
CODE_KEYWORDS = (
    "code",
    "script",
    "腳本",
    "python",
    "程式",
    "程式碼",
    "寫",
    "寫程式",
    "示範程式",
    "程式範例",
    "繪圖",
    "plot",
    "畫圖",
    "畫出",
    "製圖",
    "chart",
    "map",
    "timeseries",
    "time series",
)

CODE_REGEX = _regex(tuple(re.escape(term) for term in CODE_KEYWORDS))

# Follow-up indicators (e.g., "改成", "上一個", "again")
FOLLOWUP_KEYWORDS = (
    "改",
    "換",
    "更新",
    "修改",
    "再",
    "重新",
    "同樣",
    "上一",
    "繼續",
    "程式碼有錯",
    "follow",
    "again",
    "update",
    "change",
    "same",
    "revise",
)

PLOT_TERM_REGEX = _regex((r"plot", r"map", r"圖", r"畫", r"繪圖", r"製圖"))

# CLI/GUI tool mentions that should bias toward explain mode
CLI_TOOL_REGEX = _regex(
    (
        r"odbchat(?:_|\s*)cli",
        r"hidy(?:\s*viewer)?",
        r"圖形介面",
        r"圖台",
        r"命令列",
        r"cli 工具",
        r"命令列工具",
        r"gui",
        r"網站",
    )
)

# Explicit no-code phrases
NO_CODE_REGEX = _regex(
    (
        r"不用程式",
        r"不寫程式",
        r"不會寫程式",
        r"如果不用程式",
        r"如果不會寫程式",
        r"without\s+code",
        r"without\s+programming",
        r"no\s+coding",
        r"no\s+code",
    )
)

# Sea-state words we currently cannot answer
SEA_STATE_REGEX = _regex((r"海況", r"sea\s*state"))

# GHRSST/SST cues
GHRSST_REGEX = _regex(
    (
        r"海(?:表|水)?溫(?:度)?",
        r"海水表面溫度",
        r"水溫(?:度)?",
        r"\bsst\b",
        r"sea(?:\s|-)?surface\s+temp(?:erature)?",
        r"sea\s+temp(?:erature)?",
    )
)

GHRSST_GEO_REGEX = _regex(
    (
        r"\blon(?:gitude)?\b",
        r"\blat(?:itude)?\b",
        r"\bbox\b",
        r"經緯",
        r"座標",
        r"範圍",
        r"經度",
        r"緯度",
    )
)

TIME_HINT_REGEX = _regex(
    (
        r"今天",
        r"今日",
        r"現在",
        r"目前",
        r"最近",
        r"即時",
        r"最新",
        r"today",
        r"current",
        r"now",
        r"recent(?:ly)?",
        r"latest",
    )
)

# Tide cues
TIDE_REGEX = _regex(
    (
        r"潮(?:汐|高|位|況)?",
        r"(?:滿|乾|漲|退|大|小)潮",
        r"高低潮",
        r"日出",
        r"日落",
        r"夕陽",
        r"曙光",
        r"暮光",
        r"月(?:出|落|相)",
        r"盈虧",
        r"sun\s*(?:rise|set)",
        r"moon\s*(?:rise|set|phase)",
        r"\bhigh\s*tide\b",
        r"\blow\s*tide\b",
        r"\btide\b",
        r"\btidal\b",
    )
)

TIDE_STRONG_REGEX = _regex(
    (
        r"潮(?:汐|高|位|況)?",
        r"(?:滿|乾|漲|退|大|小)潮",
        r"高低潮",
        r"\btide(?:s)?\b",
        r"\btidal\b",
        r"\bhigh\s*tide\b",
        r"\blow\s*tide\b",
        r"\bwave\s*height\b",
        r"浪(?:高|況)?",
        r"swell",
        r"surf",
        r"tidal\s*height",
        r"tide\s*gauge",
    )
)

__all__ = [
    "CODE_KEYWORDS",
    "CODE_REGEX",
    "FOLLOWUP_KEYWORDS",
    "PLOT_TERM_REGEX",
    "CLI_TOOL_REGEX",
    "NO_CODE_REGEX",
    "SEA_STATE_REGEX",
    "GHRSST_REGEX",
    "GHRSST_GEO_REGEX",
    "TIME_HINT_REGEX",
    "TIDE_REGEX",
    "TIDE_STRONG_REGEX",
]
