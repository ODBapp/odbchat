# tests/test_server_tools.py
import sys
import types
import importlib
import pytest


class DummyAsyncClient:
    """Fallback AsyncClient to allow importing without Ollama installed."""

    def __init__(self, host: str):
        self.host = host

    async def list(self):
        # default: no models
        return {"models": []}

    async def chat(self, model, messages, options):
        # default chat behavior
        return {"message": {"content": "ok"}}


def import_server_with_dummies(models_payload=None, chat_raises: Exception | None = None):
    """
    Import 'odbchat_mcp_server' with dummy 'ollama'.
    You can pass:
      - models_payload: dict returned by DummyAsyncClient.list()
      - chat_raises: Exception to raise from DummyAsyncClient.chat()
    """
    # Clean module caches so we re-import fresh
    sys.modules.pop("odbchat_mcp_server", None)

    # Prepare a dummy AsyncClient that returns the desired stubs
    class _Client(DummyAsyncClient):
        async def list(self):
            return models_payload if models_payload is not None else super().list()

        async def chat(self, model, messages, options):
            if chat_raises:
                raise chat_raises
            return {"message": {"content": "Hello, ODB!"}}

    # Inject a minimal 'ollama' shim
    sys.modules["ollama"] = types.SimpleNamespace(AsyncClient=_Client)

    # Import the target module
    server = importlib.import_module("odbchat_mcp_server")
    return server


@pytest.mark.asyncio
async def test_check_ollama_status_running():
    # Simulate one installed model
    server = import_server_with_dummies(models_payload={"models": [{"name": "gemma3:4b"}]})

    data = await server.check_ollama_status_impl()
    assert data["status"] == "running"
    assert data["models_count"] == 1
    assert "gemma3:4b" in data["available_models"]


@pytest.mark.asyncio
async def test_check_ollama_status_error():
    # Simulate no server by making list() raise
    class Boom(Exception):
        pass

    server = import_server_with_dummies(models_payload=None, chat_raises=None)
    # Patch the injected client to raise on list()
    class RaisingClient(DummyAsyncClient):
        async def list(self):
            raise Boom("down")

    server.ollama_client = RaisingClient(host="http://x")

    data = await server.check_ollama_status_impl()
    assert data["status"] == "error"
    assert "Make sure Ollama is running" in data["suggestion"]


@pytest.mark.asyncio
async def test_list_available_models_includes_suggested():
    # Simulate two installed models; server should merge with SUGGESTED_MODELS
    server = import_server_with_dummies(
        models_payload={"models": [{"name": "gemma3:4b"}, {"name": "llama3:8b"}]}
    )
    models = await server.list_available_models_impl()
    # Installed + suggested merge; at least these should be present:
    assert "gemma3:4b" in models
    assert "llama3:8b" in models
    assert "gpt-oss:20b" in models  # suggested by server


@pytest.mark.asyncio
async def test_get_model_info_installed():
    # Simulate an installed model so server marks installed=True
    server = import_server_with_dummies(models_payload={"models": [{"name": "gpt-oss:20b", "size": 12345}]})
    info = await server.get_model_info_impl("gpt-oss:20b")
    assert info.get("installed") is True
    assert info.get("name") == "gpt-oss:20b"


@pytest.mark.asyncio
async def test_get_model_info_not_installed():
    # No models => not installed
    server = import_server_with_dummies(models_payload={"models": []})
    info = await server.get_model_info_impl("gpt-oss:20b")
    assert info.get("installed") is False
    assert info.get("name") == "gpt-oss:20b"
    # message should advise pulling the model
    assert "ollama pull" in info.get("message", "")


@pytest.mark.asyncio
async def test_chat_with_odb_success():
    # Replace chat() to return a friendly string
    server = import_server_with_dummies(models_payload={"models": []})
    text = await server.chat_with_odb_impl("hi")
    assert "Hello, ODB!" in text


@pytest.mark.asyncio
async def test_chat_with_odb_error():
    # Make chat() raise to exercise server's error formatting
    class Boom(Exception):
        pass

    # Inject a client that raises on chat()
    server = import_server_with_dummies(models_payload={"models": []}, chat_raises=Boom("boom"))
    # Ensure server uses our raising client instance
    # (This is already wired by import_server_with_dummies, but we keep the call simple)
    text = await server.chat_with_odb_impl("hi")
    assert text.startswith("Error: Unable to process request -")
