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
