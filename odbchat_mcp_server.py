# odbchat_mcp_server.py
"""
MCP Server for ODB Chat with Ollama integration
This creates a proper MCP server that can be used with mcp-use or other MCP clients
"""
import asyncio
import json
import sys
import logging
from typing import Any, Dict, List, Optional
from fastmcp import FastMCP
from ollama import AsyncClient as OllamaAsyncClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Ollama client
OLLAMA_URL = "http://127.0.0.1:11434"
ollama_client = OllamaAsyncClient(host=OLLAMA_URL)

# Create MCP server
mcp = FastMCP("odb-chat-server")

# mcp function plugins
from api.mhw_mcp import register_mhw_tools
register_mhw_tools(mcp)

# Suggested model candidates (not guaranteed installed)
SUGGESTED_MODELS = [
    "gemma3:4b",
    "gpt-oss:20b",
]

@mcp.tool()
async def ping() -> str:
    "Return 'pong' to verify the MCP server is alive."
    return "pong"

@mcp.tool()
async def chat_with_odb(
    query: str,
    model: str = "gemma3:4b",
    context: Optional[str] = None,
    temperature: float = 0.7
) -> str:
    """
    Chat with ODB knowledge using local Ollama model
    
    Args:
        query: User's question about ODB, database, API, or science communication
        model: Ollama model to use (default: gemma2:4b)
        context: Optional context from previous conversations or knowledge base
        temperature: Response creativity (0.0-1.0)
    
    Returns:
        AI response about ODB-related topics
    """
    try:
        # Prepare system prompt for ODB context
        system_prompt = """You are an expert assistant for ODB (Ocean Data Bank) and related topics including:
- ODB database systems and architecture
- API development and integration
- Knowledge base management
- Science communication
- Data management best practices

Provide accurate, helpful responses based on your knowledge of these domains."""
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add context if provided
        if context:
            messages.append({
                "role": "system", 
                "content": f"Additional context: {context}"
            })
        
        messages.append({"role": "user", "content": query})
        
        # Call Ollama
        response = await ollama_client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature}
        )
        
        return response['message']['content']
        
    except Exception as e:
        logger.error(f"Error in chat_with_odb: {e}")
        return f"Error: Unable to process request - {str(e)}"

@mcp.tool()
async def list_available_models() -> List[str]:
    """
    Return unique model names (installed + suggested). Avoids verbose reprs.
    """
    try:
        models_response = await ollama_client.list()
        logger.debug(f"Ollama list response: {models_response}")
        names: List[str] = []
        for item in models_response.get('models', []):
            name = None
            if isinstance(item, dict):
                name = item.get('name') or item.get('model')
            elif isinstance(item, str):
                name = item
            else:
                name = getattr(item, 'name', None) or getattr(item, 'model', None)
            if name:
                names.append(name)
        merged = sorted({*names, *SUGGESTED_MODELS})
        return merged
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return [f"Error: {str(e)}"]

@mcp.tool()
async def get_model_info(model: str) -> Dict[str, Any]:
    """Return details for a specific model name."""
    try:
        models_response = await ollama_client.list()
        for item in models_response.get('models', []):
            if isinstance(item, dict):
                name = item.get('name') or item.get('model')
                if name == model:
                    data = dict(item)
                    data['installed'] = True
                    return data
            else:
                name = None
                if isinstance(item, str):
                    name = item
                else:
                    name = getattr(item, 'name', None) or getattr(item, 'model', None)
                if name == model:
                    return {'name': name, 'installed': True}
        return {
            'name': model,
            'installed': False,
            'suggested': model in SUGGESTED_MODELS,
            'message': "Model not installed. Use 'ollama pull <model>' if available."
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {'error': str(e), 'name': model}

@mcp.tool()
async def check_ollama_status() -> Dict[str, Any]:
    """
    Check if Ollama service is running and accessible.
    """
    try:
        models_response = await ollama_client.list()
        
        available_models = []
        for model in models_response.get('models', []):
            if isinstance(model, dict) and "name" in model:
                available_models.append(model["name"])
            elif isinstance(model, str):
                available_models.append(model)
            else:
                available_models.append(str(model))

        return {
            "status": "running",
            "url": OLLAMA_URL,
            "models_count": len(available_models),
            "available_models": available_models
        }
    
    except Exception as e:
        logger.error(f"Error in check_ollama_status: {e}")
        return {
            "status": "error",
            "url": OLLAMA_URL,
            "error": str(e),
            "suggestion": "Make sure Ollama is running with 'ollama serve'"
        }

@mcp.resource("odb://knowledge-base")
async def odb_knowledge_base() -> str:
    """
    Provides access to ODB knowledge base information
    """
    return """
    ODB (Ocean Data Bank) Knowledge Base:
    
    1. Database Architecture:
       - PostgreSQL for structured data
       - Vector databases for embeddings
       - Time-series data handling
       
    2. API Design:
       - RESTful APIs for data access
       - GraphQL for complex queries
       - WebSocket for real-time updates
       
    3. Science Communication:
       - Data visualization techniques
       - Interactive dashboards
       - Public engagement strategies
       
    4. Data Management:
       - ETL pipelines
       - Data quality assurance
       - Metadata standards
    """

# For standalone MCP server (HTTP mode)
def main():
    """Run the MCP server in HTTP mode for mcp-use compatibility"""
    # Use FastMCP's HTTP transport so clients can POST/HTTP
    mcp.run(transport="http", host="127.0.0.1", port=8045, path="/mcp")

if __name__ == "__main__":
    main()
