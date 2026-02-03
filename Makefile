UV=uv

.PHONY: install test lint server cli

install:
	$(UV) sync --all-extras

test:
	PYTHONWARNINGS=ignore MPLBACKEND=Agg PYTHONPATH=. $(UV) run pytest -q

server:
	$(UV) run python -m server.odbchat_mcp_server

cli:
	$(UV) run python -m cli.odbchat_cli
