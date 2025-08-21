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
import os
try:  # platform-dependent
    import tty  # type: ignore
    import termios  # type: ignore
except Exception:  # pragma: no cover
    tty = None  # type: ignore
    termios = None  # type: ignore
import argparse
import select
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
        # Session-only history for slash-commands (e.g., "/mhw ...")
        self._cmd_history: list[str] = []
        self._commands = {}  # name -> async handler
        self._help = {}      # name -> help string

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
                user_input = self._readline_with_cmd_history(
                    f"\n🧑 You ({self.current_model} @ {self.temperature}): "
                ).strip()
                if not user_input:
                    continue
                if user_input.lower() in {"/quit", "/exit", "q", "quit", "exit"}:
                    print("\n👋 Goodbye!")
                    break

                if user_input.startswith('/'):
                    # Keep session command history for slash-commands
                    if not self._cmd_history or self._cmd_history[-1] != user_input:
                        self._cmd_history.append(user_input)
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
    # Minimal line editor with conditional Up/Down for slash-commands
    # ----------------------
    def _readline_with_cmd_history(self, prompt: str) -> str:
        """
        Read a line from TTY and support Up/Down navigation ONLY for slash-commands.
        - Up/Down cycle through history iff the current buffer starts with '/'
        - Left/Right/Home/End for in-line editing
        - Bracketed paste supported
        Also services matplotlib GUI while waiting for keystrokes.
        Falls back to built-in input() if no TTY or on unsupported platforms.
        """
        try:
            if (not sys.stdin.isatty() or not sys.stdout.isatty() or os.name == 'nt' or
                tty is None or termios is None):
                return input(prompt)

            # Split prompt into prefix (incl. any newlines) and the final single-line prompt
            nl_idx = prompt.rfind('\n')
            if nl_idx >= 0:
                prompt_prefix = prompt[: nl_idx + 1]
                prompt_line = prompt[nl_idx + 1 :]
            else:
                prompt_prefix = ''
                prompt_line = prompt

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)

            # Optional: GUI tick (only if matplotlib is importable and figures exist)
            # More robust against transient window-close states.
            def _gui_tick():
                try:
                    import matplotlib  # type: ignore
                    managers = list(getattr(matplotlib._pylab_helpers.Gcf, "get_all_fig_managers")() or [])
                    for m in managers:
                        try:
                            if getattr(m, "canvas", None) is not None:
                                m.canvas.flush_events()
                        except Exception:
                            # Window might be closing; ignore to keep CLI responsive
                            pass
                except Exception:
                    # matplotlib not available or backend quirks – ignore
                    pass

            # ---- display-width helper (handles emoji, CJK, combining marks) ----
            import unicodedata
            def _cols(s: str) -> int:
                w = 0
                for ch in s:
                    if unicodedata.combining(ch):
                        continue
                    w += 2 if unicodedata.east_asian_width(ch) in ('W','F') else 1
                return w

            # Print prefix + prompt line once; save anchor *after* the prompt
            if prompt_prefix:
                sys.stdout.write(prompt_prefix)
            sys.stdout.write(prompt_line)
            sys.stdout.flush()
            sys.stdout.write('\x1b[s')  # anchor just after prompt
            sys.stdout.flush()

            buf: list[str] = []
            cursor = 0
            hist_index: Optional[int] = None
            saved_before_history: str = ""

            def redraw():
                import shutil
                # terminal width and available input columns (display-width aware)
                try:
                    width = shutil.get_terminal_size().columns or 80
                except Exception:
                    width = 80
                width = max(20, width)
                avail = max(10, width - _cols(prompt_line) - 1)

                text = ''.join(buf)
                cur = max(0, cursor)
                # choose a start index so that text[start:cur] fits within avail
                start = 0
                if _cols(text) > avail:
                    # slide window so that cursor is kept in view
                    start = 0
                    while start < cur and _cols(text[start:cur]) > (avail - 1):
                        start += 1

                # build visible slice that fits in avail columns
                vis_chars = []
                used = 0
                for ch in text[start:]:
                    ch_w = 0 if unicodedata.combining(ch) else (2 if unicodedata.east_asian_width(ch) in ('W','F') else 1)
                    if used + ch_w > avail:
                        break
                    vis_chars.append(ch)
                    used += ch_w
                visible = ''.join(vis_chars)

                # columns from start→cursor inside the visible slice
                cur_cols = _cols(text[start:cur])

                # draw caret exactly at the insertion point (no visual fudge)
                cur_cols = _cols(text[start:cur])
                caret_cols = _cols(prompt_line) + min(cur_cols, _cols(visible))

                # redraw single line at saved anchor
                sys.stdout.write('\x1b[u')   # restore to anchor
                sys.stdout.write('\x1b[?7l')    # disable line wrap
                sys.stdout.write('\r')       # column 0
                sys.stdout.write('\x1b[2K')  # clear line
                sys.stdout.write(prompt_line)
                sys.stdout.write(visible)

                # place caret: "next edit position" (more intuitive)
                sys.stdout.write('\x1b[u')
                sys.stdout.write('\r')
                if caret_cols > 0:
                    sys.stdout.write(f'\x1b[{caret_cols}C')
                sys.stdout.write('\x1b[?7h')    # re‑enable wrap
                sys.stdout.flush()

            def set_buffer(s: str):
                nonlocal buf, cursor
                buf = list(s)
                cursor = len(buf)
                redraw()

            # raw mode + bracketed paste
            tty.setraw(fd)
            try:
                sys.stdout.write("\x1b[?2004h")
                sys.stdout.flush()
            except Exception:
                pass

            try:
                while True:
                    # Non-blocking wait for keypress with short timeout; tick GUI while idle
                    r, _, _ = select.select([fd], [], [], 0.05)
                    if not r:
                        _gui_tick()
                        continue

                    ch = sys.stdin.read(1)
                    if ch == '\r' or ch == '\n':
                        sys.stdout.write('\r\n')
                        sys.stdout.flush()
                        return ''.join(buf)
                    if ch == '\x03':  # Ctrl-C
                        raise KeyboardInterrupt
                    # Ctrl-D when buffer empty (we also move to col 0)
                    if ch == '\x04':  # Ctrl-D
                        if not buf:
                            sys.stdout.write('\r\n')
                            sys.stdout.flush()
                            return ''
                        continue
                    if ch in ('\x7f', '\b'):  # Backspace
                        if cursor > 0:
                            del buf[cursor-1]
                            cursor -= 1
                            hist_index = None
                            redraw()
                        continue
                    if ch == '\x1b':  # CSI / ESC sequence
                        seq1 = sys.stdin.read(1)
                        if seq1 != '[':
                            continue
                        seq = ''
                        while True:
                            c = sys.stdin.read(1)
                            seq += c
                            if c.isalpha() or c in '~':
                                break
                        # Left / Right
                        if seq == 'D':
                            if cursor > 0:
                                cursor -= 1
                                redraw()
                            continue
                        if seq == 'C':
                            if cursor < len(buf):
                                cursor += 1
                                redraw()
                            continue
                        # Up / Down (history iff buffer starts with '/')
                        if seq in ('A', 'B'):
                            current = ''.join(buf)
                            if not current.startswith('/') or not self._cmd_history:
                                continue
                            if seq == 'A':
                                if hist_index is None:
                                    hist_index = len(self._cmd_history) - 1
                                    saved_before_history = current
                                elif hist_index > 0:
                                    hist_index -= 1
                                set_buffer(self._cmd_history[hist_index])
                            else:
                                if hist_index is None:
                                    continue
                                if hist_index < len(self._cmd_history) - 1:
                                    hist_index += 1
                                    set_buffer(self._cmd_history[hist_index])
                                else:
                                    hist_index = None
                                    set_buffer(saved_before_history)
                            continue
                        # Home / End
                        if seq == 'H':
                            if cursor != 0:
                                cursor = 0
                                redraw()
                            continue
                        if seq == 'F':
                            if cursor != len(buf):
                                cursor = len(buf)
                                redraw()
                            continue
                        # Delete key: 3~
                        if seq == '3~':
                            if cursor < len(buf):
                                del buf[cursor]
                                redraw()
                            continue
                        # Bracketed paste start 200~ ... end 201~
                        if seq == '200~':
                            paste_buf: list[str] = []
                            while True:
                                x = sys.stdin.read(1)
                                if x == '\x1b':
                                    y = sys.stdin.read(1)
                                    if y != '[':
                                        continue
                                    seq2 = ''
                                    while True:
                                        z = sys.stdin.read(1)
                                        seq2 += z
                                        if z.isalpha() or z in '~':
                                            break
                                    if seq2 == '201~':
                                        break
                                    else:
                                        continue
                                else:
                                    paste_buf.append(x)
                            if paste_buf:
                                buf[cursor:cursor] = paste_buf
                                cursor += len(paste_buf)
                                hist_index = None
                                redraw()
                            continue
                        continue
                    # printable
                    if ' ' <= ch <= '~':
                        buf[cursor:cursor] = [ch]
                        cursor += 1
                        hist_index = None
                        redraw()
                        continue
                    # ignore others
            finally:
                try:
                    sys.stdout.write("\x1b[?2004l")
                    sys.stdout.flush()
                except Exception:
                    pass
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except EOFError:
            return ''
        except KeyboardInterrupt:
            raise
        except Exception:
            return input(prompt)

    # ----------------------
    # Register method
    # ----------------------
    def register_command(self, name: str, handler, help_text: str = ""):
        key = name.lower()
        self._commands[key] = handler
        if help_text:
            self._help[key] = help_text

    # ----------------------
    # Command parser
    # ----------------------
    async def _handle_command(self, line: str):
        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        # command in other registered tool patch, e.g. /mhw
        if cmd in self._commands:
            await self._commands[cmd](self, line)
            return

        if cmd == "/help":
            # support: "/help" (list) and "/help /mhw" (detail)
            if len(parts) == 1:
                print("Available commands:")
                for k in sorted(self._commands.keys()):
                    one = self._help.get(k, "").splitlines()[0] if k in self._help else ""
                    print(f"  {k:8} {one}")
                print('Tip: "/help /mhw" for detailed help.')
                return
            else:
                target = parts[1].lower()
                txt = self._help.get(target)
                if txt:
                    print(txt)
                else:
                    txt = self._help.get(f"/{target}")
                    if txt:
                        print(txt)
                    else:
                        print(f"No help for {target}")
                return
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
        try:
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            print("You current at: ", project_root)                
            from cli.mhw_cli_patch import install_mhw_command
            install_mhw_command(cli)
        except Exception:
            print("Warning: cannot load MHW CLI plugin")
            pass
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
