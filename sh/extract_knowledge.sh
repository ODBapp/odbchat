#!/bin/bash

# --- Extract structured and unstructured knowledge for RAG indexing ---
# Usage:
#   ./extract_knowledge.sh --path <path> --type <type> [--connect <conn_string>] [--out <output_file>] [--manifest manifest.yaml]
# Example:
# zarrdump "$INPUT_PATH" | sed 's/^/  /' >> "$OUTFILE"
# ✅ This adds 2 spaces to every line of the content block.
# for nc file
# ncdump -h "$INPUT_PATH" | sed 's/^/  /' >> "$OUTFILE"
# cat "$INPUT_PATH" | sed 's/^/  /' >> "$OUTFILE"
# curl -s "$CONNECT_STRING" | sed 's/^/  /' >> "$OUTFILE"
# Manifest file
# dataset: WOA23
# type: zarr
# grid: 0.25
# timescale: monthly
# description: TS metadata for World Ocean Atlas 2023 climatology datasets
# related_api: manuals/WOA23_openapi.json
# tags:
#  - temperature
#  - salinity
#  - monthly
# ./extract_knowledge.sh \
#   --path /home/odbadmin/python/woa23/data/025_degree/monthly/TS/ \
#   --type zarr \
#   --out data_summaries/woa23_zarr_025_degree_monthly_TS_metadata.txt \
#   --manifest /home/odbadmin/backup/odb_know_base/manifest_files/woa23_zarr_025_degree_monthly_TS_manifest.yaml
# Validate yml format: yamllint
# yq eval '.' "$OUTFILE" > /dev/null || echo "⚠️ YAML output is not valid!"

# --- Variables ---
INPUT_PATH=""
TYPE=""
CONNECT_STRING=""
OUTFILE="/dev/stdout"
MANIFEST_FILE=""

# --- Help Function ---
usage() {
  cat <<EOF
Usage: $0 --path <path> --type <type> [--connect <conn_string>] [--out <output_file>] [--manifest manifest.yaml]

Options:
  --path <path>           Path to file, directory, or dataset
  --type <type>           manual, schema, code_snippet, paper, website_doc, data_summary, zarr, nc, api, db
  --connect <string>      DB URI or OpenAPI URL/file
  --out <file>            Output file (default: stdout)
  --manifest <file>       Optional: YAML manifest file to inject metadata
EOF
  exit 1
}

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --path) INPUT_PATH="$2"; shift 2 ;;
    --type) TYPE="$2"; shift 2 ;;
    --connect) CONNECT_STRING="$2"; shift 2 ;;
    --out) OUTFILE="$2"; shift 2 ;;
    --manifest) MANIFEST_FILE="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# --- Validation ---
if [[ -z "$INPUT_PATH" || -z "$TYPE" ]]; then
  echo "[ERROR] --path and --type are required."
  usage
fi

# --- Metadata Prepend Helper ---
prepend_metadata_from_manifest() {
  if [[ -n "$MANIFEST_FILE" && -f "$MANIFEST_FILE" ]]; then
    echo "---" > "$OUTFILE"
    cat "$MANIFEST_FILE" >> "$OUTFILE"
    echo "content: |" >> "$OUTFILE"
  fi
}

# --- Content Extraction Functions ---

extract_text_files() {
  find "$1" -type f \( -iname "*.txt" -o -iname "*.md" -o -iname "*.html" -o -iname "*.py" \) -print0 |
    xargs -0 cat >> "$OUTFILE"
}

extract_zarr_metadata() {
  if ! command -v zarrdump &> /dev/null; then
    echo "[ERROR] zarrdump not installed. Run: pip install zarrdump"
    exit 1
  fi
  prepend_metadata_from_manifest
  zarrdump "$INPUT_PATH" >> "$OUTFILE"
}

extract_netcdf_metadata() {
  if ! command -v ncdump &> /dev/null; then
    echo "[ERROR] ncdump not installed. Run: sudo apt install netcdf-bin"
    exit 1
  fi
  prepend_metadata_from_manifest
  ncdump -h "$INPUT_PATH" >> "$OUTFILE"
}

extract_api_schema() {
  prepend_metadata_from_manifest
  if [[ "$CONNECT_STRING" =~ ^https?:// ]]; then
    curl -s "$CONNECT_STRING" >> "$OUTFILE"
  elif [[ -f "$CONNECT_STRING" ]]; then
    cat "$CONNECT_STRING" >> "$OUTFILE"
  else
    echo "[ERROR] Invalid API source: $CONNECT_STRING"
    exit 1
  fi
}

extract_postgres_schema() {
  prepend_metadata_from_manifest
  local dsn="$1"
  psql "$dsn" -Atc "
    SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
  " | while read -r table; do
    echo "### Table: $table" >> "$OUTFILE"
    psql "$dsn" -Atc "
      SELECT column_name, data_type FROM information_schema.columns
      WHERE table_name = '$table';
    " >> "$OUTFILE"
    echo "" >> "$OUTFILE"
  done
}

extract_mysql_schema() {
  prepend_metadata_from_manifest
  # Requires parsing and external script for MySQL; placeholder for now
  echo "[ERROR] MySQL schema extraction not yet implemented." >> "$OUTFILE"
}

extract_db_schema() {
  if [[ "$CONNECT_STRING" =~ ^postgresql:// ]]; then
    extract_postgres_schema "$CONNECT_STRING"
  elif [[ "$CONNECT_STRING" =~ ^mysql:// ]]; then
    extract_mysql_schema "$CONNECT_STRING"
  else
    echo "[ERROR] Unsupported DB URI format: $CONNECT_STRING"
    exit 1
  fi
}

# --- Main Logic ---
case "$TYPE" in
  manual|schema|code_snippet|paper|website_doc|data_summary)
    prepend_metadata_from_manifest
    if [[ -f "$INPUT_PATH" ]]; then
      cat "$INPUT_PATH" >> "$OUTFILE"
    elif [[ -d "$INPUT_PATH" ]]; then
      extract_text_files "$INPUT_PATH"
    else
      echo "[ERROR] Invalid path: $INPUT_PATH"
      exit 1
    fi
    ;;
  zarr)
    extract_zarr_metadata
    ;;
  nc)
    extract_netcdf_metadata
    ;;
  api)
    if [[ -z "$CONNECT_STRING" ]]; then
      echo "[ERROR] --connect required for API type"
      exit 1
    fi
    extract_api_schema
    ;;
  db)
    if [[ -z "$CONNECT_STRING" ]]; then
      echo "[ERROR] --connect required for DB type"
      exit 1
    fi
    extract_db_schema
    ;;
  *)
    echo "[ERROR] Unknown type: $TYPE"
    usage
    ;;
esac

echo "[INFO] Extraction complete. Output written to $OUTFILE"
