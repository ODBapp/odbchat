"""
CLI client for ODB Chat MCP server (unified /command style)
- Uses FastMCP HTTP transport at http://localhost:8045/mcp by default
- Streams responses via local Ollama when available

Commands (type /help for this list):
  /server connect <url>     Connect (or reconnect) to an MCP server URL
  /server set <url>         Set server URL without connecting yet
  /server show              Show current server URL and connection status
  /tools                    List available MCP tools
  /tool <name> k=v ...      Call a tool with key=value arguments
  /models                   List available local Ollama models
  /info <model>             Show details (installed, size, etc.)
  /model <model>            Set chat model (e.g., gemma3:4b, gpt-oss:20b)
  /temp <0..1>              Set temperature for chat
  /status                   Check Ollama server via MCP tool
  /help                     Show commands
  /quit | /exit | q         Exit

Anything else is treated as a chat prompt (agent-less direct chat via Ollama stream,
then falls back to MCP tool if streaming fails).
"""

import asyncio
import json
import sys
from typing import Optional, Dict, Any
import argparse
from fastmcp import Client

try:
    from ollama import AsyncClient as OllamaAsyncClient  # for streaming
except Exception:  # pragma: no cover
    OllamaAsyncClient = None

DEFAULT_SERVER_URL = "http://localhost:8045/mcp"
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_TEMPERATURE = 0.7
OLLAMA_URL = "http://127.0.0.1:11434"

class ODBChatClient:
    def __init__(self, server_url: str = DEFAULT_SERVER_URL, default_model: str = DEFAULT_MODEL):
        self.server_url = server_url
        self.current_model = default_model
        self.temperature = DEFAULT_TEMPERATURE
        self.client: Optional[Client] = None

    # ----------------------
    # Connection management
    # ----------------------
    async def connect(self) -> bool:
        """Connect to the MCP server at self.server_url."""
        await self.disconnect()  # ensure clean state
        try:
            self.client = Client(self.server_url)
            await self.client.__aenter__()
            print(f"✅ Connected to ODB MCP server at {self.server_url}")
            return True
        except Exception as e:  # pragma: no cover
            print(f"❌ Failed to connect to server: {e}")
            print("Make sure the server is running: python odbchat_mcp_server.py")
            self.client = None
            return False

    async def disconnect(self):
        if self.client is not None:
            try:
                await self.client.__aexit__(None, None, None)
            finally:
                self.client = None

    # ----------------------
    # Tool helpers
    # ----------------------
    async def list_tools(self):
        if not self.client:
            print("❌ Not connected. Use /server connect <url> first.")
            return []
        try:
            tools = await self.client.list_tools()
            print("\n📋 Available tools:")
            for tool in tools:
                # Support both object and dict responses
                name = getattr(tool, "name", None)
                if name is None and isinstance(tool, dict):
                    name = tool.get("name")
                    if name is None:
                        name = str(tool)

                desc = getattr(tool, "description", None)
                if desc is None and isinstance(tool, dict):
                    desc = tool.get("description")
                print(f"  - {name}: {desc or 'No description'}")
            return tools

        except Exception as e:
            print(f"❌ Error listing tools: {e}")
            return []

    async def check_status(self) -> bool:
        if not self.client:
            print("❌ Not connected. Use /server connect <url> first.")
            return False
        try:
            result = await self.client.call_tool("check_ollama_status", {})
            status_text = self._extract_text(result)
            status_data = json.loads(status_text)
            print(f"\n🔍 Server Status:")
            print(f"  Status: {status_data.get('status', 'unknown')}")
            print(f"  Ollama URL: {status_data.get('url', 'unknown')}")
            if status_data.get('status') != 'running':
                print(f"  Error: {status_data.get('error', 'Unknown error')}")
                print(f"  Suggestion: {status_data.get('suggestion', '')}")
            print(f"  Current model: {self.current_model}")
            return status_data.get('status') == 'running'
        except Exception as e:
            print(f"❌ Error checking status: {e}")
            return False

    async def list_models(self):
        if not self.client:
            print("❌ Not connected. Use /server connect <url> first.")
            return []
        try:
            result = await self.client.call_tool("list_available_models", {})
            models_text = self._extract_text(result)
            models = json.loads(models_text)
            names = sorted({str(m) for m in models}) if isinstance(models, list) else []
            print(f"\n🤖 Available models ({len(names)}):")
            for i, name in enumerate(names, 1):
                print(f"  {i}. {name}")
            return names
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []

    async def show_model_info(self, name: str):
        if not self.client:
            print("❌ Not connected. Use /server connect <url> first.")
            return
        try:
            result = await self.client.call_tool("get_model_info", {"model": name})
            info_text = self._extract_text(result)
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

    async def call_tool(self, name: str, args: Dict[str, Any]):
        if not self.client:
            print("❌ Not connected. Use /server connect <url> first.")
            return
        try:
            res = await self.client.call_tool(name, args)
            text = self._extract_text(res)
            print(text)
        except Exception as e:
            print(f"❌ Tool error: {e}")

    # ----------------------
    # Chat (stream via Ollama → fallback to MCP tool)
    # ----------------------
    async def chat(self, query: str, model: Optional[str] = None, context: Optional[str] = None):
        use_model = model or self.current_model or DEFAULT_MODEL
        # Try local streaming first
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
                    options={"temperature": self.temperature},
                    stream=True,
                ):
                    content = chunk.get('message', {}).get('content')
                    if content:
                        print(content, end="", flush=True)
                        full_response += content
                print()  # newline after streaming
                return full_response
            except Exception as e:  # pragma: no cover
                print(f"\n❌ Streaming error: {e}. Falling back to MCP tool.")
                # fall through to MCP tool

        # Fallback: call MCP tool (non-streaming)
        if not self.client:
            print("❌ Not connected. Use /server connect <url> first.")
            return ""
        try:
            result = await self.client.call_tool(
                "chat_with_odb",
                {
                    "query": query,
                    "model": use_model,
                    "context": context,
                    "temperature": self.temperature,
                },
            )
            text = self._extract_text(result)
            print(text)
            return text
        except Exception as e:
            print(f"❌ Error in chat: {e}")
            return f"Error: {str(e)}"

    # ----------------------
    # Interactive shell
    # ----------------------
    async def interactive_chat(self):
        print(f"\n🗣️  Starting ODBChat CLI")
        print("Type /help for commands. Anything else is sent as a prompt.\n")

        context = None  # basic rolling context

        while True:
            try:
                user_input = input(f"\n🧑 You ({self.current_model} @ {self.temperature}): ").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"/quit", "/exit", "q", "quit", "exit"}:
                    print("👋 Goodbye!")
                    break

                if user_input.startswith('/'):
                    await self._handle_command(user_input)
                    continue

                # Chat prompt
                print("🤖 Assistant: ", end="", flush=True)
                response = await self.chat(user_input, None, context)
                # maintain short rolling context (tail only)
                if response:
                    snippet = response[-200:]
                    context = (context + "\n" if context else "") + f"User: {user_input}\nAssistant: {snippet}"
                    if len(context) > 1000:
                        context = context[-800:]

            except KeyboardInterrupt:  # pragma: no cover
                print("\n👋 Goodbye!")
                break
            except Exception as e:  # pragma: no cover
                print(f"\n❌ Error: {e}")

    # ----------------------
    # Command parser
    # ----------------------
    async def _handle_command(self, line: str):
        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "/help":
            print(self._help_text())
        elif cmd == "/server":
            await self._cmd_server(args)
        elif cmd == "/tools":
            await self.list_tools()
        elif cmd == "/tool":
            if not args:
                print("Usage: /tool <name> k=v …")
            else:
                name = args[0]
                kv = self._parse_kv(args[1:])
                await self.call_tool(name, kv)
        elif cmd == "/models":
            await self.list_models()
        elif cmd == "/info":
            if not args:
                print("Usage: /info <model>")
            else:
                await self.show_model_info(args[0])
        elif cmd == "/model":
            if not args:
                print(f"Current model: {self.current_model}")
            else:
                self.current_model = args[0]
                print(f"🔄 Switched to model: {self.current_model}")
        elif cmd == "/temp":
            if not args:
                print(f"Current temperature: {self.temperature}")
            else:
                try:
                    t = float(args[0])
                    if 0.0 <= t <= 1.0:
                        self.temperature = t
                        print(f"🌡️  Temperature set to: {self.temperature}")
                    else:
                        print("❌ Temperature must be between 0.0 and 1.0")
                except ValueError:
                    print("❌ Invalid temperature value")
        elif cmd == "/status":
            await self.check_status()
        else:
            print("❓ Unknown command. /help for help.")

    async def _cmd_server(self, args: list[str]):
        if not args:
            conn = "connected" if self.client else "disconnected"
            print(f"Server URL: {self.server_url} ({conn})")
            return
        sub = args[0].lower()
        if sub == "connect":
            url = args[1] if len(args) > 1 else self.server_url
            if url != self.server_url:
                self.server_url = url
            await self.connect()
        elif sub == "set":
            if len(args) < 2:
                print("Usage: /server set <url>")
                return
            self.server_url = args[1]
            print(f"Server URL set to: {self.server_url}. Use '/server connect' to connect.")
        elif sub == "show":
            conn = "connected" if self.client else "disconnected"
            print(f"Server URL: {self.server_url} ({conn})")
        else:
            print("Usage: /server connect <url> | /server set <url> | /server show")

    # ----------------------
    # Utilities
    # ----------------------
    def _parse_kv(self, parts: list[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            # strip quotes
            if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
                v = v[1:-1]
            # try coercion
            if v.isdigit():
                v = int(v)
            else:
                try:
                    v_float = float(v)
                    v = v_float
                except ValueError:
                    pass
            out[k] = v
        return out

    def _extract_text(self, result: Any) -> str:
        if hasattr(result, 'text') and result.text is not None:
            return result.text
        if hasattr(result, 'content'):
            c = result.content
            if isinstance(c, list) and c:
                first = c[0]
                if hasattr(first, 'text') and first.text is not None:
                    return first.text
                return str(first)
            return str(c)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    def _help_text(self) -> str:
        return (
            "\n📖 Commands:\n"
            "  /server connect <url>     Connect (or reconnect) to an MCP server URL\n"
            "  /server set <url>         Set server URL without connecting yet\n"
            "  /server show              Show current server URL and status\n"
            "  /tools                    List available MCP tools\n"
            "  /tool <name> k=v ...      Call a tool with key=value arguments\n"
            "  /models                   List available local Ollama models\n"
            "  /info <model>             Show model details\n"
            "  /model <model>            Set chat model\n"
            "  /temp <0..1>              Set temperature for chat\n"
            "  /status                   Check Ollama via MCP tool\n"
            "  /help                     Show this help\n"
            "  /quit | /exit | q         Exit\n"
        )


async def main():
    parser = argparse.ArgumentParser(description="ODB Chat CLI (unified /command style)")
    parser.add_argument("--server", "-s", default=DEFAULT_SERVER_URL,
                        help=f"MCP server URL (default: {DEFAULT_SERVER_URL})")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Default model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--query", "-q", help="Single query instead of interactive mode")
    parser.add_argument("--temperature", "-t", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Temperature for responses (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--list-tools", action="store_true", help="List available tools and exit")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--status", action="store_true", help="Check server status and exit")

    args = parser.parse_args()

    cli = ODBChatClient(args.server, default_model=args.model)
    cli.temperature = args.temperature

    # Always connect on startup so non-chat commands work
    connected = await cli.connect()
    if not connected:
        sys.exit(1)

    try:
        from mhw_cli_patch import install_mhw_command
        install_mhw_command(cli)

        if args.status:
            await cli.check_status()
        elif args.list_tools:
            await cli.list_tools()
        elif args.list_models:
            await cli.list_models()
        elif args.query:
            # Single query mode
            print(f"🤖 Using model: {cli.current_model} (T={cli.temperature})")
            print("🤖 Assistant: ", end="", flush=True)
            await cli.chat(args.query)
        else:
            # Interactive mode
            await cli.interactive_chat()
    finally:
        await cli.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
