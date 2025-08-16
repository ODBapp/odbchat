PY=python
PIP=pip

.PHONY: install test lint server cli

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt || true

test:
	PYTHONWARNINGS=ignore MPLBACKEND=Agg PYTHONPATH=. pytest -q

server:
	$(PY) odbchat_mcp_server.py

cli:
	$(PY) cli/odbchat_cli.py
