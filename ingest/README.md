# Ingestion Scripts

This directory contains ingestion utilities for loading RAG content into Qdrant.

## `ingest.py` (legacy YAML)

Ingests the legacy multi-document YAML files used by ODBchat.

## `ingest_json.py` (Omnipipe JSON)

Ingests Omnipipe JSON exports that contain an `artifacts` list with graph links/tags.

### Usage

```bash
python ingest/ingest_json.py --file /path/to/artifacts.json --collection my_collection --mode upsert
```

```bash
python ingest/ingest_json.py --root /path/to/json_dir --mode overwrite
```

### Modes

- `overwrite`: drop and recreate the collection before ingestion
- `upsert`: replace existing artifacts with the same `artifact_id`
- `insert`: error if an `artifact_id` already exists
- `dry-run`: print the ingestion plan without writing to Qdrant

### Environment

- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY`
- `EMBED_MODEL`, `EMBED_DEVICE`
- `OMNIPIPE_COLLECTION_PREFIX` (optional prefix for auto-named collections)

### Notes

- Collection defaults to the JSON filename (without extension) unless `--collection` is provided.
- `artifact_id` is stored in the payload, along with `links`, `tags`, and Omnipipe metadata fields.
