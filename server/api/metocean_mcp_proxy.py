"""Proxy registrations for Metocean MCP tools (tide, sun, moon)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import mcp.types
from fastmcp import FastMCP
from fastmcp.client import Client as MCPClient
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError

logger = logging.getLogger(__name__)

METOCEAN_MCP_URL = os.getenv(
    "METOCEAN_MCP_URL",
    "https://eco.odb.ntu.edu.tw/mcp/metocean",
)
METOCEAN_UA = os.getenv("METOCEAN_MCP_USER_AGENT", "metocean-mcp")


async def _call_metocean_tool(tool_name: str, payload: dict[str, Any]) -> Any:
    headers = {"User-Agent": METOCEAN_UA}
    try:
        transport = StreamableHttpTransport(METOCEAN_MCP_URL, headers=headers)
        async with MCPClient(transport) as client:
            result = await client.call_tool(tool_name, payload)
    except ToolError:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Metocean MCP call failed: %s", exc)
        raise ToolError(f"Failed upstream metocean tool {tool_name!r}: {exc}")

    if result.data is not None:
        return result.data

    if result.structured_content:
        return result.structured_content

    if result.content:
        texts: list[str] = []
        for block in result.content:
            if isinstance(block, mcp.types.TextContent):
                texts.append(block.text)
        if texts:
            combined = "".join(texts).strip()
            if combined:
                try:
                    return json.loads(combined)
                except json.JSONDecodeError:
                    return combined

    return {
        "content": result.content,
        "structured_content": result.structured_content,
        "is_error": result.is_error,
    }


async def _tide_forecast(**payload: Any) -> Any:
    clean = {k: v for k, v in payload.items() if v is not None}
    return await _call_metocean_tool("tide.forecast", clean)


def register_metocean_tools(mcp: FastMCP) -> None:
    """Expose metocean upstream tools locally."""

    @mcp.tool(name="tide.forecast")
    async def tide_forecast(
        longitude: float | None = None,
        latitude: float | None = None,
        station_id: str | None = None,
        date: str | None = None,
        query_time: str | None = None,
        tz: str | None = None,
    ) -> Any:
        payload = {
            "longitude": longitude,
            "latitude": latitude,
            "station_id": station_id,
            "date": date,
            "query_time": query_time,
            "tz": tz,
        }
        return await _tide_forecast(**payload)
