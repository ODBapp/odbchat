"""Proxy registrations for GHRSST MCP tools.

This module wires the upstream GHRSST MCP server hosted at
https://eco.odb.ntu.edu.tw/mcp/ghrsst into the local odbchat FastMCP server.
The upstream server already implements the `ghrsst.point_value` and
`ghrsst.bbox_mean` tools – we simply forward calls so clients can treat them as
first-class tools exposed by this deployment.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import mcp.types
from fastmcp import FastMCP
from fastmcp.client import Client as MCPClient
from fastmcp.exceptions import ToolError

logger = logging.getLogger(__name__)

# Public endpoint for the upstream GHRSST MCP tools. Allow overriding via env.
GHRSST_MCP_URL = os.getenv(
    "GHRSST_MCP_URL",
    "https://eco.odb.ntu.edu.tw/mcp/ghrsst",
)


async def _call_ghrsst_tool(tool_name: str, payload: dict[str, Any]) -> Any:
    """Call a GHRSST MCP tool on the upstream server and normalize the result.

    The upstream tool already returns structured JSON; we preserve that structure
    when available and gracefully fall back to decoded text content.
    """

    try:
        async with MCPClient(GHRSST_MCP_URL) as client:
            result = await client.call_tool(tool_name, payload)
    except ToolError:
        # ToolError already carries a helpful message; re-raise for FastMCP to surface.
        raise
    except Exception as exc:  # pragma: no cover - network issues surfaced at runtime
        logger.exception("GHRSST MCP call failed: %s", exc)
        raise ToolError(f"Failed to call upstream GHRSST tool {tool_name!r}: {exc}")

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

    # Fallback: return the upstream protocol payload untouched.
    return {
        "content": result.content,
        "structured_content": result.structured_content,
        "is_error": result.is_error,
    }


async def _ghrsst_point_value(
    longitude: float,
    latitude: float,
    date: str | None = None,
    fields: list[str] | None = None,
    method: str | None = None,
) -> Any:
    payload: dict[str, Any] = {
        "longitude": longitude,
        "latitude": latitude,
    }
    if date:
        payload["date"] = date
    if fields:
        payload["fields"] = fields
    if method:
        payload["method"] = method
    return await _call_ghrsst_tool("ghrsst.point_value", payload)


async def _ghrsst_bbox_mean(
    bbox: list[float],
    date: str,
    fields: list[str] | None = None,
    method: str | None = None,
) -> Any:
    if len(bbox) != 4:
        raise ToolError("bbox must contain exactly four numbers: [lon0, lat0, lon1, lat1]")

    payload: dict[str, Any] = {
        "bbox": bbox,
        "date": date,
    }
    if fields:
        payload["fields"] = fields
    if method:
        payload["method"] = method
    return await _call_ghrsst_tool("ghrsst.bbox_mean", payload)


def register_ghrsst_tools(mcp: FastMCP) -> None:
    """Register proxy wrappers for the upstream GHRSST tools."""

    @mcp.tool(name="ghrsst.point_value")
    async def ghrsst_point_value(
        longitude: float,
        latitude: float,
        date: str | None = None,
        fields: list[str] | None = None,
        method: str | None = None,
    ) -> Any:
        return await _ghrsst_point_value(longitude, latitude, date, fields, method)

    @mcp.tool(name="ghrsst.bbox_mean")
    async def ghrsst_bbox_mean(
        bbox: list[float],
        date: str,
        fields: list[str] | None = None,
        method: str | None = None,
    ) -> Any:
        return await _ghrsst_bbox_mean(bbox, date, fields, method)
