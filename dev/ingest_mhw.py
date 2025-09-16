#!/usr/bin/env python3
"""Shim that maps legacy ingest_mhw CLI options to ingest/ingest.py."""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List

from ingest.ingest import main as ingest_main


def _parse_legacy_args(argv: List[str]) -> List[str]:
    parser = argparse.ArgumentParser(
        description="Deprecated shim for ingesting MHW YAML files",
        add_help=True,
    )
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("rag"))
    parser.add_argument("--file", type=pathlib.Path, help="Single YAML file to ingest", default=None)
    parser.add_argument("--mode", choices=["overwrite", "add"], default="add")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without writing")
    parsed = parser.parse_args(argv)

    new_args: List[str] = []
    if parsed.file is not None:
        new_args.extend(["--file", str(parsed.file)])
    else:
        new_args.extend(["--root", str(parsed.root)])

    if parsed.dry_run:
        mode = "dry-run"
    elif parsed.mode == "overwrite":
        mode = "overwrite"
    else:
        mode = "upsert"
    new_args.extend(["--mode", mode])
    return new_args


def main(argv: List[str] | None = None) -> None:
    legacy_argv = sys.argv[1:] if argv is None else argv
    translated = _parse_legacy_args(legacy_argv)
    ingest_main(translated)


if __name__ == "__main__":
    main()
