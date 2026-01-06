if [[ $# -eq 0 ]] ; then
    echo 'Warning: No input port, use default 8000'
    port=8000
else
    port="$1"
fi

OUT_DIR=/home/odbadmin/proj/odbchat/rag/json_out
# mkdir -p "$OUT_DIR"
find /home/odbadmin/proj/omnipipe/tests/rag/legacy_yamls -type f -name '*.yml' -print0 |
while IFS= read -r -d '' f; do
  base=$(basename "$f")
  curl -s -X POST -F "file=@$f" "http://localhost:${port}/v1/ingest" \
    > "$OUT_DIR/${base%.yml}.json"
done

# test only one pdf
for f in /home/odbadmin/proj/omnipipe/tests/rag/raw_pdfs/woa23documentation.pdf; do
  base=$(basename "$f")
  curl -s -X POST -F "file=@$f" "http://localhost:${port}/v1/ingest" -F "pages=8,9,10,11,12" > "$OUT_DIR/${base%.*}.json"
done

# test only one img
for f in /home/odbadmin/proj/omnipipe/tests/rag/raw_images/odb_open_apis01.png; do
  base=$(basename "$f")
  curl -s -X POST -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@$f" "http://localhost:${port}/v1/ingest" -F "pages=8,9,10,11,12" > "$OUT_DIR/${base%.*}.json"
done

