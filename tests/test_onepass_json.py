import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_onepass_retrieval_with_omnipipe_links(qdrant_service, tmp_path, monkeypatch):
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers is required for JSON retrieval tests")
    repo_root = Path(__file__).resolve().parents[2]
    ingest_script = repo_root / "odbchat" / "ingest" / "ingest_json.py"
    json_path = tmp_path / "sample_links.json"
    data = {
        "artifacts": [
            {
                "id": "artifact-a",
                "source_file": "sample.txt",
                "artifact_type": "text_chunk",
                "links": [
                    {
                        "target_id": "artifact-b",
                        "link_type": "references",
                        "metadata": {"confidence": 0.9},
                    }
                ],
                "tags": ["sample"],
                "text": "alpha unique token",
                "page_number": 1,
                "chunk_index": 0,
            },
            {
                "id": "artifact-b",
                "source_file": "sample.txt",
                "artifact_type": "text_chunk",
                "links": [],
                "tags": ["sample"],
                "text": "beta-only",
                "page_number": 2,
                "chunk_index": 1,
            },
        ]
    }
    _write_json(json_path, data)

    collection = "onepass_json_test"
    env = os.environ.copy()
    env.update(
        {
            "QDRANT_HOST": "localhost",
            "QDRANT_PORT": "6334",
            "EMBED_DEVICE": "cpu",
        }
    )
    subprocess.run(
        [
            sys.executable,
            str(ingest_script),
            "--file",
            str(json_path),
            "--collection",
            collection,
            "--mode",
            "overwrite",
            "--embedding-device",
            "cpu",
        ],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    monkeypatch.setenv("ODB_ONEPASS_SOURCE", "omnipipe")
    monkeypatch.setenv("OMNIPIPE_COLLECTIONS", collection)
    monkeypatch.setenv("OMNIPIPE_LINK_EXPANSION", "1")
    monkeypatch.setenv("QDRANT_HOST", "localhost")
    monkeypatch.setenv("QDRANT_PORT", "6334")
    monkeypatch.setenv("EMBED_DEVICE", "cpu")

    import importlib
    from server.rag import onepass_core

    importlib.reload(onepass_core)
    hits = onepass_core.search_qdrant("alpha unique token", k=4)
    artifact_ids = {hit.payload.get("artifact_id") for hit in hits}
    assert "artifact-a" in artifact_ids
    assert "artifact-b" in artifact_ids

    qdrant_service.delete_collection(collection_name=collection)
