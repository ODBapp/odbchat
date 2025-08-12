# odbchat_cli.py
"""
CLI client for ODB Chat MCP server
Works with the HTTP FastMCP server running on port 8045
"""

import asyncio
import json
import sys
from typing import Optional
import argparse
from fastmcp import Client
try:
    from ollama import AsyncClient as OllamaAsyncClient  # for streaming
except Exception:
    OllamaAsyncClient = None

SUGGESTED_MODELS = [
    "gemma3:4b",
    "gpt-oss:20b",
]
OLLAMA_URL = "http://127.0.0.1:11434"

class ODBChatClient:
    def __init__(self, server_url: str = "http://localhost:8045/mcp", default_model: str = "gemma3:4b"):
        # HTTP transport expects base endpoint (no /sse)
        self.server_url = server_url
        self.current_model = default_model
        self.client = None
    
    async def connect(self):
        """Connect to the MCP server"""
        try:
            self.client = Client(self.server_url)
            await self.client.__aenter__()
            print(f"✅ Connected to ODB MCP server at {self.server_url}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            print("Make sure the server is running: python odbchat_mcp_server.py")
            return False
    
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.client:
            await self.client.__aexit__(None, None, None)
    
    async def list_tools(self):
        """List available tools on the server"""
        try:
            tools = await self.client.list_tools()
            print("\n📋 Available tools:")
            for tool in tools:
                print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
            return tools
        except Exception as e:
            print(f"❌ Error listing tools: {e}")
            return []
    
    async def check_status(self):
        """Check Ollama and server status"""
        try:
            result = await self.client.call_tool("check_ollama_status", {})
            
            # Handle different result formats
            if hasattr(result, 'text'):
                status_text = result.text
            elif hasattr(result, 'content') and isinstance(result.content, list) and len(result.content) > 0:
                status_text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
            else:
                status_text = str(result)
            
            status_data = json.loads(status_text)
            
            print(f"\n🔍 Server Status:")
            print(f"  Status: {status_data.get('status', 'unknown')}")
            print(f"  Ollama URL: {status_data.get('url', 'unknown')}")
            if status_data.get('status') != 'running':
                print(f"  Error: {status_data.get('error', 'Unknown error')}")
                print(f"  Suggestion: {status_data.get('suggestion', '')}")
            
            return status_data.get('status') == 'running'
        except Exception as e:
            print(f"❌ Error checking status: {e}")
            return False
    
    async def list_models(self):
        """List available Ollama models"""
        try:
            result = await self.client.call_tool("list_available_models", {})
            
            # Handle different result formats
            if hasattr(result, 'text'):
                models_text = result.text
            elif hasattr(result, 'content') and isinstance(result.content, list) and len(result.content) > 0:
                models_text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
            else:
                models_text = str(result)
                
            models = json.loads(models_text)
            # Names only, unique and sorted
            names = sorted({str(m) for m in models}) if isinstance(models, list) else []
            print(f"\n🤖 Available models ({len(names)}):")
            for i, name in enumerate(names, 1):
                print(f"  {i}. {name}")
            return names
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
    
    async def chat(self, query: str, model: Optional[str] = None, context: Optional[str] = None, temperature: float = 0.7):
        """Send a chat message; stream tokens when possible for low latency.

        If model is None, uses the client's current_model.
        """
        use_model = model or getattr(self, 'current_model', None) or "gemma3:4b"
        if OllamaAsyncClient is not None:
            try:
                system_prompt = (
                    "You are an expert assistant for ODB (Ocean Data Bank) and related topics including:\n"
                    "- ODB database systems and architecture\n"
                    "- API development and integration\n"
                    "- Knowledge base management\n"
                    "- Science communication\n"
                    "- Data management best practices\n\n"
                    "Provide accurate, helpful responses based on your knowledge of these domains."
                )
                messages = [{"role": "system", "content": system_prompt}]
                if context:
                    messages.append({"role": "system", "content": f"Additional context: {context}"})
                messages.append({"role": "user", "content": query})

                client = OllamaAsyncClient(host=OLLAMA_URL)
                full_response = ""
                async for chunk in await client.chat(
                    model=use_model,
                    messages=messages,
                    options={"temperature": temperature},
                    stream=True
                ):
                    if chunk['message']['content']:
                        content = chunk['message']['content']
                        print(content, end="", flush=True)
                        full_response += content

                print()  # New line after streaming
                return full_response

            except Exception as e:
                print(f"\n❌ Streaming error: {e}. Falling back to MCP.")
                return f"Error: {str(e)}"
        try:
            result = await self.client.call_tool(
                "chat_with_odb",
                {
                    "query": query,
                    "model": use_model,
                    "context": context,
                    "temperature": temperature
                }
            )
            if hasattr(result, 'text'):
                text = result.text
            elif hasattr(result, 'content'):
                if isinstance(result.content, list) and len(result.content) > 0:
                    text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                else:
                    text = str(result.content)
            elif isinstance(result, str):
                text = result
            else:
                text = str(result)
            print(text)
            return text
        except Exception as e:
            print(f"❌ Error in chat: {e}")
            return f"Error: {str(e)}"

    # stream_chat_direct is no longer needed; streaming is handled in chat()
    
    async def interactive_chat(self, model: str = "gemma3:4b", temperature: float = 0.7):
        """Start an interactive chat session"""
        # Initialize current model for the session
        self.current_model = model
        print(f"\n🗣️  Starting interactive chat with {self.current_model}")
        print("Type 'quit', 'exit', or 'q' to exit")
        print("Type 'models' to see available models")
        print("Type 'status' to check server status")
        print("Type 'help' for more commands")
        print("-" * 50)
        
        context = None  # Store conversation context
        
        while True:
            try:
                user_input = input(f"\n🧑 You ({self.current_model}): ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() in ['models', ':models']:
                    await self.list_models()
                    continue
                elif user_input.lower() in ['status', ':status']:
                    await self.check_status()
                    print(f"  Current model: {self.current_model}")
                    continue
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.startswith(':'):
                    # Unified colon-command handler supports optional spaces (e.g., ": select model")
                    cmdline = user_input[1:].strip()
                    if not cmdline:
                        continue
                    parts = cmdline.split(maxsplit=1)
                    cmd = parts[0].lower()
                    arg = parts[1].strip() if len(parts) > 1 else ''
                    if cmd == 'models':
                        await self.list_models()
                    elif cmd == 'status':
                        await self.check_status()
                        print(f"  Current model: {self.current_model}")
                    elif cmd == 'model':
                        if not arg:
                            print("❌ Usage: :model <model_name>")
                        else:
                            await self.show_model_info(arg)
                    elif cmd == 'select':
                        if not arg:
                            print("❌ Usage: :select <model_name>")
                        else:
                            self.current_model = arg
                            print(f"🔄 Switched to model: {self.current_model}")
                    else:
                        print(f"❓ Unknown command: :{cmd}")
                    continue
                elif user_input.startswith('/model '):
                    new_model = user_input[7:].strip()
                    self.current_model = new_model
                    print(f"🔄 Switched to model: {self.current_model}")
                    continue
                elif user_input.startswith('/temp '):
                    try:
                        temp = float(user_input[6:].strip())
                        if 0.0 <= temp <= 1.0:
                            temperature = temp
                            print(f"🌡️  Temperature set to: {temperature}")
                        else:
                            print("❌ Temperature must be between 0.0 and 1.0")
                    except ValueError:
                        print("❌ Invalid temperature value")
                    continue
                
                # Send chat message
                print("🤖 Assistant: ", end="", flush=True)
                response = await self.chat(user_input, None, context, temperature=temperature)
                
                # Update context with recent conversation
                if context:
                    context = f"{context}\nUser: {user_input}\nAssistant: {response[-200:]}"
                else:
                    context = f"User: {user_input}\nAssistant: {response[-200:]}"
                
                # Keep context manageable
                if len(context) > 1000:
                    context = context[-800:]
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def show_help(self):
        """Show help commands"""
        print("\n📖 Available commands:")
        print("  quit, exit, q     - Exit the chat")
        print("  models | :models  - List available models (names only)")
        print("  status | :status  - Show server status (no model list)")
        print("  :model <name>     - Show details of a model")
        print("  :select <name>    - Change current model")
        print("  /model <name>     - Switch model (legacy)")
        print("  /temp <value>     - Set temperature (0.0-1.0)")

    async def show_model_info(self, name: str):
        """Fetch and display concise model details."""
        try:
            result = await self.client.call_tool("get_model_info", {"model": name})
            if hasattr(result, 'text'):
                info_text = result.text
            elif hasattr(result, 'content') and isinstance(result.content, list) and len(result.content) > 0:
                info_text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
            else:
                info_text = str(result)
            data = json.loads(info_text)
            if isinstance(data, dict) and 'error' in data:
                print(f"❌ {data['error']}")
                return
            print("\n🧩 Model Info:")
            print(f"  Name: {data.get('name', name)}")
            print(f"  Installed: {data.get('installed', False)}")
            if 'modified_at' in data:
                print(f"  Modified: {data['modified_at']}")
            if 'size' in data:
                print(f"  Size: {data['size']}")
            if 'size_bytes' in data and 'size' not in data:
                print(f"  Size: {data['size_bytes']} bytes")
            if 'message' in data and not data.get('installed', False):
                print(f"  Note: {data['message']}")
        except Exception as e:
            print(f"❌ Error fetching model info: {e}")
        

async def main():
    parser = argparse.ArgumentParser(description="ODB Chat CLI Client")
    parser.add_argument("--server", "-s", default="http://localhost:8045/mcp",
                       help="MCP server URL (default: http://localhost:8045/mcp)")
    parser.add_argument("--model", "-m", default="gemma3:4b",
                       help="Default model to use (default: gemma3:4b)")
    parser.add_argument("--query", "-q", help="Single query instead of interactive mode")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                       help="Temperature for responses (default: 0.7)")
    # Streaming is default via Ollama AsyncClient when available; no flag needed.
    parser.add_argument("--list-tools", action="store_true",
                       help="List available tools and exit")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    parser.add_argument("--status", action="store_true",
                       help="Check server status and exit")
    
    args = parser.parse_args()
    
    # Create client and set initial model
    client = ODBChatClient(args.server, default_model=args.model)
    
    # Connect to server
    if not await client.connect():
        sys.exit(1)
    
    try:
        # Handle different modes
        if args.status:
            await client.check_status()
        elif args.list_tools:
            await client.list_tools()
        elif args.list_models:
            await client.list_models()
        elif args.query:
            # Single query mode (streaming by default when possible)
            client.current_model = args.model
            print(f"🤖 Using model: {client.current_model}")
            print("🤖 Assistant: ", end="", flush=True)
            await client.chat(args.query, None, temperature=args.temperature)
        else:
            # Interactive mode
            await client.interactive_chat(args.model, temperature=args.temperature)
    
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
