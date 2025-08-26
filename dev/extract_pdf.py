#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
from pathlib import Path

try:
    import pymupdf  # PyMuPDF is imported as 'pymupdf' in recent versions
except Exception:
    # fallback for older installs
    import fitz as pymupdf  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser(description="Extract plain text from a PDF (UTF-8).")
    ap.add_argument("--pdf", required=True, help="Input PDF file path")
    ap.add_argument("--out", required=True, help="Output text file path")
    return ap.parse_args()


def extract_pdf_text(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    parts = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text()  # utf-8 plain text
        parts.append(f"\n===== Page {i} =====\n{text.strip()}\n")
    return "\n".join(parts).strip()


def main():
    args = parse_args()
    src = Path(args.pdf)
    dst = Path(args.out)
    if not src.exists():
        print(f"[ERROR] PDF not found: {src}")
        sys.exit(1)

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        text = extract_pdf_text(str(src))
    except Exception as e:
        print(f"[ERROR] Failed to read PDF: {e}")
        sys.exit(2)

    try:
        dst.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to write output: {e}")
        sys.exit(3)

    print(f"[OK] Wrote: {dst}")


if __name__ == "__main__":
    main()
