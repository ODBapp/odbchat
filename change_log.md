#### v0.0.1 first commit to setup simple mcp
#### v0.0.2 fix CLI basic functions (stream, status, change model..) to make it work
#### v0.0.3 First MHW API mcp and plot plugin for CLI/n1
#### v0.0.4 CLI: add slash-command history with Up/Down; ignore in chat mode
#### v0.0.5 CLI line editor: correct cursor across wrapped lines; MHW plots now non-blocking
#### v0.1.0 CLI major refactor to move API to server-side, fix antimeridian problem in basemap/cartopy and various bugs
#### v0.1.1 Add shared schemas for client/server MCP interface and validation/add batch fetch tool
#### v0.1.2 Build pytest test files and env/n1
#### v0.1.3 Big refactor cli/plugins/map_plot.py for basemap/cartopy/plain crossing-zero, antimeridian cases
#### v0.1.4 Simplify map_plot backend helpers: unify legend/colorbar placement into reusable functions; add _nice_ticks; no behavior change intended
#### v0.1.5 CLI GUI stability: harden matplotlib GUI tick to avoid stalls when closing one of multiple windows; add optional detached plotting via ODBCHAT_PLOT_DETACH=1 so plot windows live independently of the CLI
#### v0.1.6 Start switching to llama.cpp/build/set-up RAG docs/n1
#### v0.1.7 Setup RAG docs/n2/rag_cli.py to test ingested RAG doc
#### v0.1.8 Setup RAG docs with FAQ, purpose/n3/ingest qdrant with overwite/additive mode
#### v0.1.9 New Ingestion so that it can handle multiple YAML with OAS/and rag_cli.py can find OAS parameters/n4
    - A temporary version to improve rag_cli.py capabilities and analysis Qdrant hits/n5,n6
#### v0.2.0 More statble LLM chat combine with code and explin mode. Better restructure by checking Qdrant statistics/n7
    - a more stable version that split code/explain mode by LLM itself 
    - a temp version that provide code template to plot map or timeseries, need tuned/n1
    - deprecate code template if LLM not reply empty code, need tuned/n2
    - deprecate second_try LLM to reduce response time, need tuned/n3
	- small fix for more accurate code rag extract, remove imports/n4, small fix: scoring and rerank for qdrant/n5
#### v0.2.1 A new scoring-rerank-diversify way for qdrant and try to improve first-round answer/n1
#### v0.2.2 A new One-pass LLM structure try to reduce response time/n1
    - try to improve code/explain mode in one-pass LLM structure, need tuned/n2 
    - improve sharing sysrules module for one-pass LLM, need tuned/n3
#### v0.2.3 Staged-like structured one-pass LLM improvement/n1
    - continue improving sharing sysrules.../n2 
#### v0.2.4 Determine plot_rules, more robust sysrules for staged-structure one-pass LLM/n3
    - continue improving sharing sysrules before ingest woa23.../n3 
    - a more loose extract-plan-and-code block parser and chat-message completion for llama-cpp 
#### v0.3.0 MCP/CLI refactor to centralized one-pass pipeline
    - add generalised ingest/ingest.py with multi-doc YAML + dry-run support
    - extract server/rag/onepass_core.py and expose rag.onepass_answer MCP tool
    - move mhw MCP helpers under server/api, wire CLI to rag.onepass_answer
    - add schemas, prompt templates, and new pytest coverage for guardrails and CLI UX
#### v0.3.1 Debug plumbing for one-pass pipeline and CLI integration
    - rag.onepass MCP schema/tool accept debug flag and emit timing + whitelist diagnostics
    - run_onepass attaches detailed debug payload on demand and handles empty LLM replies gracefully
    - CLI adds --debug, prints server plan/code/debug info, and syncs provider:model via config tools
#### v0.3.2 Harden one-pass code generation and plan handling
    - Aligned server prompts with dev CLI logic: reuse previous plans, inject OAS server URLs, and enforce chunk_rule/plot_rule behaviour (timeseries loops, map pivot+pcolormesh, no handcrafted query strings).
    - Normalised llama.cpp inline `code {…}` responses into formatted Python (including plan-less fallbacks), appended guard/warning comments when code skips required plan params, chunk loops, or plotting steps, and surfaced warnings in CLI debug output.
    - Suppressed plan display in normal CLI output, preserved it in debug diagnostics, and ensured debug payload validation passes even for explain-mode replies.
#### v0.3.3 Proxy GHRSST MCP tools via odbchat server
    - Registered upstream `ghrsst.point_value` and `ghrsst.bbox_mean` FastMCP tools so clients can query eco.odb.ntu.edu.tw/mcp/ghrsst through the local server.
    - Added proxy helper tests covering payload forwarding and validation.
    - Forced uvicorn to use the classic `websockets` backend for compatibility with the deployment environment.
#### v0.3.4 Add lightweight MCP router
    - Introduced `router.answer` tool that classifies queries into `mcp_tools|explain|code` and calls GHRSST tools when appropriate.
    - Updated CLI chat flow to use the router and render SST results directly.
    - Added router unit tests plus CLI coverage for the new MCP branch.
    - Improved GHRSST tool handling with optional `method="nearest"` fallback and user-facing error messages when data is unavailable.
