from pathlib import Path

from ingest import ingest


def test_ingest_dry_run_outputs_plan(capsys):
    sample = Path(__file__).resolve().parents[1] / "rag" / "manuals" / "odb_mhw_openapi.yml"
    ingest.main(["--file", str(sample), "--mode", "dry-run"])
    captured = capsys.readouterr().out
    assert '"action": "plan"' in captured
