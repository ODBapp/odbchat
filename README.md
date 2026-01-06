# Chat with AI about ODB

knowledge_base/
├── manuals/           # API docs, user guides
├── schema/            # DB schemas
├── code_snippets/     # FastAPI, ETL, configs
├── papers/            # Related PDFs or markdown
├── web_docs/          # Scraped web docs
├── data_summaries/    # CSV headers, .nc/.zarr metadata, etc.

## Quickstart (Qdrant + Omnipipe JSON)

1) Run Qdrant (test port 6334 to avoid collisions):

```bash
docker run -d --rm --name qdrant_test -p 6334:6333 -v /tmp/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

2) Ingest Omnipipe JSON artifacts:

```bash
python ingest/ingest_json.py --root /path/to/json_dir --mode overwrite
```

Supported arguments:

- `--root PATH` (scan for `.json` recursively)
- `--file PATH` (ingest a single JSON file)
- `--collection NAME` (defaults to filename or `OMNIPIPE_COLLECTION_PREFIX + filename`)
- `--mode {overwrite,upsert,insert,dry-run}`
- `--embedding-model NAME`
- `--embedding-device DEVICE`

Required env variables:

- `QDRANT_HOST` (e.g. `localhost`)
- `QDRANT_PORT` (e.g. `6334`)

Optional env variables:

- `QDRANT_API_KEY`
- `EMBED_MODEL`
- `EMBED_DEVICE`
- `OMNIPIPE_COLLECTION_PREFIX`

3) Run odbchat LLM mode by llama-cpp:

```bash
llama-server -m /home/odbadmin/proj/odbchat/models/google_gemma-3-12b-it-Q5_K_M.gguf -md /home/odbadmin/proj/odbchat/models/google_gemma-3-1b-it-Q4_K_M.gguf -c 6144 -ngl 99 -ngld 24 --parallel 1 -t 16 --threads-batch 16 --batch-size 384 --ubatch-size 384 --kv-unified --no-cont-batching --ctx-checkpoints 0 -fa off --cache-ram 0 --cache-reuse 0 --threads-http 1 --timeout 600 --port 8201
```

LLM env vars (llama-cpp):

- `ODB_LLM_PROVIDER=llama-cpp`
- `ODB_LLM_MODEL=gemma3:12b`

4) Run odbchat mcp-server:

```bash
python -m server.odbchat_mcp_server
```

5) Run odbchat client:

```bash
python -m cli.odbchat_cli
```

### One-pass retrieval env vars

Select data source:

- `ODB_ONEPASS_SOURCE=yaml` (default) or `omnipipe`
- `OMNIPIPE_COLLECTIONS=AI,AI2` (used when source is `omnipipe`)
- `OMNIPIPE_LINK_EXPANSION=1` (enable graph link expansion)

Qdrant connection:

- `QDRANT_HOST=localhost`
- `QDRANT_PORT=6334`
- `QDRANT_API_KEY` (optional)
