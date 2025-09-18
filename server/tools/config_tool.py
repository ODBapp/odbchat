# server/tools/config_tool.py
from fastmcp import FastMCP
from server.llm_adapter import LLM

def register_rag_config_tool(mcp: FastMCP):
    @mcp.tool(name="config.get_model", description="Get current LLM provider/model and available models.")
    def get_model() -> dict:
        try:
            available = LLM.list_models()
        except Exception:
            available = [LLM.model]
        return {"provider": LLM.provider, "model": LLM.model, "available": available}

    @mcp.tool(name="config.set_model", description="Set current model for the active provider.")
    def set_model(model: str) -> dict:
        ok = False
        try:
            ok = LLM.set_model(model)
        except Exception:
            ok = False
        return {"ok": bool(ok), "provider": LLM.provider, "model": LLM.model}

    @mcp.tool(name="config.set_provider", description="Set active LLM provider (ollama|llama-cpp).")
    def set_provider(provider: str) -> dict:
        ok = False
        try:
            ok = LLM.set_provider(provider)
        except Exception:
            ok = False
        return {"ok": bool(ok), "provider": LLM.provider}

