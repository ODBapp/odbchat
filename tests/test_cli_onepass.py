import pytest

from cli.odbchat_cli import ODBChatClient


class DummyMCPClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def call_tool(self, name, payload):
        self.calls.append((name, payload))
        return self.responses[name]


@pytest.mark.asyncio
async def test_cli_chat_explain(monkeypatch, capsys):
    monkeypatch.setattr("cli.odbchat_cli.OllamaAsyncClient", None, raising=False)
    responses = {
        "router.answer": {
            "mode": "explain",
            "text": "Marine heatwaves explained",
            "citations": [{"title": "Doc", "source": "https://example", "chunk_id": 0}],
        }
    }
    client = ODBChatClient()
    client.client = DummyMCPClient(responses)
    client.default_tz = "Asia/Taipei"
    client._current_query_time = lambda: "2025-01-10T00:00:00+08:00"
    output = await client.chat("什麼是海洋熱浪?")
    captured = capsys.readouterr().out
    assert "Marine heatwaves explained" in captured
    assert "Citations" in captured
    assert "https://example" in captured
    assert "Marine heatwaves" in output


@pytest.mark.asyncio
async def test_cli_chat_code(monkeypatch, capsys):
    monkeypatch.setattr("cli.odbchat_cli.OllamaAsyncClient", None, raising=False)
    responses = {
        "router.answer": {
            "mode": "code",
            "plan": {"endpoint": "/api/mhw", "params": {"lon0": 120, "lat0": 23}},
            "code": "import requests\nprint('ok')",
            "citations": [{"title": "Spec", "source": "https://spec", "chunk_id": 1}],
        }
    }
    client = ODBChatClient()
    client.client = DummyMCPClient(responses)
    client.default_tz = "Asia/Taipei"
    client._current_query_time = lambda: "2025-01-10T00:00:00+08:00"
    output = await client.chat("請提供程式碼")
    captured = capsys.readouterr().out
    assert "Plan" not in captured
    assert "import requests" in captured
    assert "Citations" in captured
    assert "import requests" in output


@pytest.mark.asyncio
async def test_cli_chat_mcp(monkeypatch, capsys):
    monkeypatch.setattr("cli.odbchat_cli.OllamaAsyncClient", None, raising=False)
    responses = {
        "router.answer": {
            "mode": "mcp_tools",
            "tool": "ghrsst.point_value",
            "arguments": {
                "longitude": 123.0,
                "latitude": 23.0,
                "date": "2025-01-10",
                "method": "nearest",
            },
            "text": "2025-01-10｜point [123.0, 23.0]: SST ≈ 26.40 °C, Anomaly +0.12 °C",
            "result": {
                "region": [123.0, 23.0],
                "date": "2025-01-10",
                "sst": 26.4,
                "sst_anomaly": 0.12,
            },
            "citations": [],
        }
    }
    client = ODBChatClient()
    client.client = DummyMCPClient(responses)
    client.default_tz = "Asia/Taipei"
    client._current_query_time = lambda: "2025-01-10T00:00:00+08:00"
    output = await client.chat("今天台灣外海(23N, 123E)海溫多少？")
    captured = capsys.readouterr().out
    assert "SST" in captured
    assert "2025-01-10" in output
    assert client.client.calls
    _, payload = client.client.calls[0]
    assert payload["tz"] == "Asia/Taipei"
    assert payload["query_time"] == "2025-01-10T00:00:00+08:00"


@pytest.mark.asyncio
async def test_cli_chat_refusal_no_citations(monkeypatch, capsys):
    monkeypatch.setattr("cli.odbchat_cli.OllamaAsyncClient", None, raising=False)
    responses = {
        "router.answer": {
            "mode": "explain",
            "text": "抱歉，我無法提供該資訊。",
            "citations": [
                {"title": "Marine Heatwaves", "source": "https://example", "chunk_id": 1}
            ],
        }
    }
    client = ODBChatClient()
    client.client = DummyMCPClient(responses)
    client.default_tz = "Asia/Taipei"
    client._current_query_time = lambda: "2025-01-10T00:00:00+08:00"
    await client.chat("問題")
    captured = capsys.readouterr().out
    assert "抱歉" in captured
    assert "Citations" not in captured


@pytest.mark.asyncio
async def test_cli_llm_status(monkeypatch, capsys):
    monkeypatch.setattr("cli.odbchat_cli.OllamaAsyncClient", None, raising=False)
    responses = {
        "config.llm_status": {
            "provider": "llama-cpp",
            "model": "gemma3:12b",
            "timeout": 60,
            "reachable": True,
            "healthy": False,
            "last_error": "ReadTimeout",
            "available": ["gemma3:12b"],
        }
    }
    client = ODBChatClient()
    client.client = DummyMCPClient(responses)
    await client._cmd_llm(["status"])
    captured = capsys.readouterr().out
    assert "llm backend status" in captured.lower()
    assert "gemma3:12b" in captured
