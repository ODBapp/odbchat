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
        "rag.onepass_answer": {
            "mode": "explain",
            "text": "Marine heatwaves explained",
            "citations": [{"title": "Doc", "source": "https://example", "chunk_id": 0}],
        }
    }
    client = ODBChatClient()
    client.client = DummyMCPClient(responses)
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
        "rag.onepass_answer": {
            "mode": "code",
            "plan": {"endpoint": "/api/mhw", "params": {"lon0": 120, "lat0": 23}},
            "code": "import requests\nprint('ok')",
            "citations": [{"title": "Spec", "source": "https://spec", "chunk_id": 1}],
        }
    }
    client = ODBChatClient()
    client.client = DummyMCPClient(responses)
    output = await client.chat("請提供程式碼")
    captured = capsys.readouterr().out
    assert "Plan" not in captured
    assert "import requests" in captured
    assert "Citations" in captured
    assert "import requests" in output
