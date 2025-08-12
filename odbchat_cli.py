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

class ODBChatClient:
    def __init__(self, server_url: str = "http://localhost:8045/mcp"):
        self.server_url = server_url
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
            
            if status_data.get('status') == 'running':
                print(f"  Available models: {len(status_data.get('available_models', []))}")
                for model in status_data.get('available_models', []):
                    print(f"    - {model}")
            else:
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
            print(f"\n🤖 Available models ({len(models)}):")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
            return models
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
    
    async def chat(self, query: str, model: str = "gemma3:4b", context: Optional[str] = None, temperature: float = 0.7):
        """Send a chat message to the ODB assistant"""
        try:
            result = await self.client.call_tool(
                "chat_with_odb",
                {
                    "query": query,
                    "model": model,
                    "context": context,
                    "temperature": temperature
                }
            )
            # Handle different possible result formats
            if hasattr(result, 'text'):
                return result.text
            elif hasattr(result, 'content'):
                if isinstance(result.content, list) and len(result.content) > 0:
                    return result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                else:
                    return str(result.content)
            elif isinstance(result, str):
                return result
            else:
                return str(result)
        except Exception as e:
            print(f"❌ Error in chat: {e}")
            return f"Error: {str(e)}"
    
    async def interactive_chat(self, model: str = "gemma3:4b"):
        """Start an interactive chat session"""
        print(f"\n🗣️  Starting interactive chat with {model}")
        print("Type 'quit', 'exit', or 'q' to exit")
        print("Type 'models' to see available models")
        print("Type 'status' to check server status")
        print("Type 'help' for more commands")
        print("-" * 50)
        
        context = None  # Store conversation context
        
        while True:
            try:
                user_input = input(f"\n🧑 You ({model}): ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'models':
                    await self.list_models()
                    continue
                elif user_input.lower() == 'status':
                    await self.check_status()
                    continue
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.startswith('/model '):
                    new_model = user_input[7:].strip()
                    model = new_model
                    print(f"🔄 Switched to model: {model}")
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
                response = await self.chat(user_input, model, context)
                print(response)
                
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
        print("  models           - List available models")
        print("  status           - Check server status")
        print("  help             - Show this help")
        print("  /model <name>    - Switch to different model (e.g., /model gemma2:12b)")
        print("  /temp <value>    - Set temperature (0.0-1.0, e.g., /temp 0.8)")
        

async def main():
    parser = argparse.ArgumentParser(description="ODB Chat CLI Client")
    parser.add_argument("--server", "-s", default="http://localhost:8045/mcp",
                       help="MCP server URL (default: http://localhost:8045/mcp)")
    parser.add_argument("--model", "-m", default="gemma3:4b",
                       help="Default model to use (default: gemma3:4b)")
    parser.add_argument("--query", "-q", help="Single query instead of interactive mode")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                       help="Temperature for responses (default: 0.7)")
    parser.add_argument("--list-tools", action="store_true",
                       help="List available tools and exit")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    parser.add_argument("--status", action="store_true",
                       help="Check server status and exit")
    
    args = parser.parse_args()
    
    # Create client
    client = ODBChatClient(args.server)
    
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
            # Single query mode
            print(f"🤖 Using model: {args.model}")
            response = await client.chat(args.query, args.model, temperature=args.temperature)
            print(f"\n🤖 Assistant: {response}")
        else:
            # Interactive mode
            await client.interactive_chat(args.model)
    
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
