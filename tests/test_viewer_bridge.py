import asyncio
import json
import socket

import pytest
import websockets

from cli.odbchat_cli import ODBChatClient


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.mark.asyncio
async def test_viewer_attach_and_plot_roundtrip():
    client = ODBChatClient(server_url="http://localhost:0")
    client.viewer_port = _free_port()
    await client._start_viewer_server()
    ws_url = f"ws://127.0.0.1:{client.viewer_port}"

    async def fake_viewer():
        async with websockets.connect(ws_url) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "plugin.register",
                        "pluginProtocolVersion": "1.0",
                        "capabilities": {"open_records": True, "plot": True},
                    }
                )
            )
            ack = json.loads(await ws.recv())
            assert ack.get("type") == "plugin.register_ok"

            open_msg = json.loads(await ws.recv())
            assert open_msg["op"] == "open_records"
            await ws.send(
                json.dumps(
                    {
                        "op": "open_records.ok",
                        "msgId": open_msg["msgId"],
                        "datasetKey": open_msg.get("datasetKey"),
                    }
                )
            )

            plot_msg = json.loads(await ws.recv())
            assert plot_msg["op"] == "plot"
            await ws.send(
                json.dumps(
                    {
                        "op": "plot_blob",
                        "msgId": plot_msg["msgId"],
                        "contentType": "image/png",
                        "size": 4,
                    }
                )
            )
            await ws.send(b"png!")

    viewer_task = asyncio.create_task(fake_viewer())

    # Wait briefly for registration to complete
    await asyncio.sleep(0.05)
    assert client.viewer is not None
    assert client.viewer_caps.get("plot") is True

    res = await client.viewer.open_records({"datasetKey": "ds", "records": []})
    assert res.get("op") in {"open_records.ok", "open_records_ok"}

    seen: dict = {}

    async def on_header(header):
        seen["header"] = header

    async def on_binary(data: bytes):
        seen["png"] = data

    await client.viewer.plot({"datasetKey": "ds", "kind": "map", "y": "sst"}, on_header, on_binary)

    assert seen.get("header", {}).get("op") == "plot_blob"
    assert seen.get("png") == b"png!"

    await client.disconnect()
    await viewer_task
