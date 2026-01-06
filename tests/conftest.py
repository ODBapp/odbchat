import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add project root to sys.path so `import cli`, `import shared` work.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def qdrant_service(tmp_path_factory):
    if not shutil.which("docker"):
        pytest.skip("docker is required for Qdrant integration tests")
    try:
        from qdrant_client import QdrantClient
    except Exception:
        pytest.skip("qdrant-client is required for Qdrant integration tests")

    storage = tmp_path_factory.mktemp("qdrant_storage")
    subprocess.run(["docker", "rm", "-f", "qdrant_test"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            "qdrant_test",
            "-p",
            "6334:6333",
            "-v",
            f"{storage}:/qdrant/storage",
            "qdrant/qdrant:latest",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to start Qdrant: {result.stderr.strip()}")

    client = QdrantClient(host="localhost", port=6334)
    ready = False
    for _ in range(40):
        try:
            client.get_collections()
            ready = True
            break
        except Exception:
            time.sleep(0.5)
    if not ready:
        subprocess.run(["docker", "stop", "qdrant_test"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pytest.skip("Qdrant did not become ready in time")

    try:
        yield client
    finally:
        subprocess.run(["docker", "stop", "qdrant_test"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
