import sys
import types
import importlib
import pytest


class DummyAsyncClient:
    """Fallback AsyncClient to allow importing without ollama installed."""

    def __init__(self, host: str):
        self.host = host

    async def list(self):
        return {"models": []}

    async def chat(self, model, messages, options):
        return {"message": {"content": "ok"}}


def import_server_with_dummy_ollama():
    """Import server module with a dummy 'ollama' module injected."""
    sys.modules.pop("odbchat_mcp_server", None)
    sys.modules["ollama"] = types.SimpleNamespace(AsyncClient=DummyAsyncClient)
    return importlib.import_module("odbchat_mcp_server")


@pytest.mark.asyncio
async def test_check_ollama_status_running(monkeypatch):
    server = import_server_with_dummy_ollama()

    class FakeClient(DummyAsyncClient):
        async def list(self):
            return {"models": [{"name": "gemma3:4b"}]}

    monkeypatch.setattr(server, "ollama_client", FakeClient(host="http://x"))

    data = await server.check_ollama_status()
    assert data["status"] == "running"
    assert data["models_count"] == 1
    assert "gemma3:4b" in data["available_models"]


@pytest.mark.asyncio
async def test_list_available_models(monkeypatch):
    server = import_server_with_dummy_ollama()

    class FakeClient(DummyAsyncClient):
        async def list(self):
            return {"models": [{"name": "gemma3:4b"}, {"name": "llama3:8b"}]}

    monkeypatch.setattr(server, "ollama_client", FakeClient(host="http://x"))

    models = await server.list_available_models()
    # Should include installed and suggested candidate
    assert set(["gemma3:4b", "llama3:8b"]).issubset(set(models))
    assert "gpt-oss:20b" in models


@pytest.mark.asyncio
async def test_chat_with_odb_success(monkeypatch):
    server = import_server_with_dummy_ollama()

    class FakeClient(DummyAsyncClient):
        async def chat(self, model, messages, options):
            return {"message": {"content": "Hello, ODB!"}}

    monkeypatch.setattr(server, "ollama_client", FakeClient(host="http://x"))

    text = await server.chat_with_odb("hi")
    assert "Hello, ODB!" in text


@pytest.mark.asyncio
async def test_chat_with_odb_error(monkeypatch):
    server = import_server_with_dummy_ollama()

    class FakeClient(DummyAsyncClient):
        async def chat(self, model, messages, options):
            raise RuntimeError("boom")

    monkeypatch.setattr(server, "ollama_client", FakeClient(host="http://x"))

    text = await server.chat_with_odb("hi")
    assert text.startswith("Error:")
