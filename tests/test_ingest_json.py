import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _legacy_yaml_to_omnipipe_json(yaml_path: Path) -> dict:
    docs = list(yaml.safe_load_all(yaml_path.read_text(encoding="utf-8")))
    if not docs:
        return {"artifacts": []}
    front = docs[0] if isinstance(docs[0], dict) else {}
    source_file = front.get("source_file") or str(yaml_path)
    tags = front.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    artifacts = []
    chunk_index = 0
    for idx, doc in enumerate(docs):
        doc_meta = doc if isinstance(doc, dict) else {}
        text = doc_meta.get("content")
        if not text and isinstance(doc, str):
            text = doc
        if not text:
            text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
        if not isinstance(text, str) or not text.strip():
            continue
        artifact_id = hashlib.sha1(f"{source_file}:{idx}".encode("utf-8")).hexdigest()
        artifacts.append(
            {
                "id": artifact_id,
                "source_file": source_file,
                "artifact_type": "text_chunk",
                "links": [],
                "tags": tags,
                "text": text,
                "page_number": int(doc_meta.get("page_number") or 0),
                "chunk_index": chunk_index,
            }
        )
        chunk_index += 1
    return {"artifacts": artifacts}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_ingest_json_from_legacy_yamls(qdrant_service, tmp_path):
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers is required for JSON ingestion tests")
    repo_root = Path(__file__).resolve().parents[2]
    legacy_root = repo_root / "omnipipe" / "tests" / "rag" / "legacy_yamls"
    yaml_files = sorted(legacy_root.rglob("*.yml"))
    assert yaml_files, "Expected legacy YAML fixtures under omnipipe/tests/rag/legacy_yamls"

    ingest_script = repo_root / "odbchat" / "ingest" / "ingest_json.py"
    env = os.environ.copy()
    env.update(
        {
            "QDRANT_HOST": "localhost",
            "QDRANT_PORT": "6334",
            "EMBED_DEVICE": "cpu",
        }
    )

    client = qdrant_service
    for yaml_path in yaml_files:
        rel = yaml_path.relative_to(legacy_root).as_posix().replace("/", "__")
        json_path = tmp_path / f"{rel}.json"
        data = _legacy_yaml_to_omnipipe_json(yaml_path)
        _write_json(json_path, data)
        artifact_count = len(data["artifacts"])
        assert artifact_count > 0

        collection = f"test_{rel}".replace(".", "_")
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

        count = client.count(collection_name=collection, exact=True)
        assert count.count == artifact_count
        client.delete_collection(collection_name=collection)


def test_ingest_json_dry_run_outputs_plan(tmp_path):
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers is required for JSON ingestion tests")
    repo_root = Path(__file__).resolve().parents[2]
    ingest_script = repo_root / "odbchat" / "ingest" / "ingest_json.py"
    json_path = tmp_path / "sample.json"
    data = {
        "artifacts": [
            {
                "id": "artifact-1",
                "source_file": "sample.txt",
                "artifact_type": "text_chunk",
                "links": [],
                "tags": ["sample"],
                "text": "hello world",
                "page_number": 1,
                "chunk_index": 0,
            }
        ]
    }
    _write_json(json_path, data)

    result = subprocess.run(
        [
            sys.executable,
            str(ingest_script),
            "--file",
            str(json_path),
            "--collection",
            "dry_run_collection",
            "--mode",
            "dry-run",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert '"action": "plan"' in result.stdout
    assert "artifact-1" in result.stdout
