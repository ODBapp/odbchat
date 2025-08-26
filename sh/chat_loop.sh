#!/usr/bin/env bash
# 自動續寫到 ⟪END⟫；支援 -m 4b|12b, -l tw|en, -q "PROMPT", -t MAX_TOKENS, -p PORT
# prompt examples
# USER_PROMPT="請以簡明數點解釋 Ekman 運輸：(1) 螺旋與層厚度量級（數十公尺），(2) 北半球右 90°／南半球左 90°，(3) 沿岸上升／下沉的規則。判斷題：北半球、岸線南北向，沿岸風向赤道→近岸是上升或下沉？解釋原因。最後輸出 ⟪END⟫"
# USER_PROMPT="以條列式、科學準確地解釋聖嬰（El Niño）與反聖嬰（La Niña）。請包含：(1) 定義與判別區（Niño‑3.4 海溫異常正/負），(2) Bjerknes 正回饋（風—海溫—熱躍層傾斜），(3) 熱躍層傾斜在聖嬰與反聖嬰的差異，(4) 赤道下傳 Kelvin 波與 Rossby 波角色，(5) Walker/Hadley 環流與降水帶的位移，(6) 對西北太平洋（含臺灣）颱風活動或雨量的典型影響（方向性即可）。"

set -euo pipefail
DEBUG="${DEBUG:-0}"

# -------- defaults --------
PORT=8001
MODEL_ALIAS="12b"      # 4b | 12b
LANG="tw"              # tw | en
USER_PROMPT=""         # default prompt below if empty
MAX_TOKENS=400
TEMPERATURE=0.25
TOP_P=0.9
TOP_K=40
STOP_TOKEN=$'\n⟪END⟫'
MAX_LOOPS=8            # 安全上限，避免無限迴圈

# -------- parse args --------
usage() {
  cat <<USAGE
Usage: $0 [-m 4b|12b] [-l tw|en] [-q "USER_PROMPT"] [-t MAX_TOKENS] [-p PORT]
  -m  模型別名：4b 或 12b（預設 12b）
  -l  語言：tw=繁體中文（預設） / en=English
  -q  自訂問題（未提供則用內建簡短題）
  -t  max_tokens（預設 400）
  -p  服務埠（預設 8001）
環境變數：DEBUG=1 可列印 payload 與 RAW 回應
USAGE
  exit 1
}

while getopts ":m:l:q:t:p:h" opt; do
  case "${opt}" in
    m) MODEL_ALIAS="${OPTARG}" ;;
    l) LANG="${OPTARG}" ;;
    q) USER_PROMPT="${OPTARG}" ;;
    t) MAX_TOKENS="${OPTARG}" ;;
    p) PORT="${OPTARG}" ;;
    h|*) usage ;;
  esac
done 2>/dev/null || true

# 驗證 PORT 為數字
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "❌ PORT 必須是數字，但收到: '$PORT' （你是不是把 -m 寫成 -p 了？用 -m 4b / -m 12b）" >&2
  exit 2
fi

# -------- model mapping --------
# 4b:  google_gemma-3-4b-it-Q4_K_M.gguf
# 12b: google_gemma-3-12b-it-Q8_0.gguf
case "$MODEL_ALIAS" in
  4b) MODEL="google_gemma-3-4b-it-Q4_K_M.gguf" ;;
  12b) MODEL="google_gemma-3-12b-it-Q8_0.gguf" ;;
  *) echo "Unknown -m '$MODEL_ALIAS' (use 4b or 12b)"; exit 2 ;;
esac

# 檢查檔案存在
if [[ ! -f "$HOME/proj/odbchat/models/$MODEL" && ! -f "./models/$MODEL" ]]; then
  echo "⚠️ 找不到模型檔：$MODEL（請確認在 ~/models 或 ./models）" >&2
fi

# -------- language system text --------
if [[ "$LANG" == "tw" ]]; then
  SYSTEM_TEXT="僅用繁體中文作答；若要求條列式回覆，每點最多 2–3 句。務求科學準確；最後獨立一行輸出 ⟪END⟫"
  DEFAULT_PROMPT="請以條列 4 點，簡述黑潮的起源、路徑、氣候/生態影響與黑潮延伸流。"
elif [[ "$LANG" == "en" ]]; then
  SYSTEM_TEXT="Answer only in English; If bulleted response is required, keep each bullet to 2–3 sentences. Be scientifically accurate; end with ⟪END⟫ on its own line."
  DEFAULT_PROMPT="List 4 concise bullets on the Kuroshio Current: origin, path, climate/ecology impacts, and Kuroshio Extension."
else
  echo "Unknown -l '$LANG' (use tw or en)"; exit 2
fi

PROMPT_CONTENT="${USER_PROMPT:-$DEFAULT_PROMPT}"

# -------- init messages --------
MESSAGES=$(jq -n \
  --arg s "$SYSTEM_TEXT" \
  --arg u "$PROMPT_CONTENT" \
  '{messages:[{"role":"system","content":$s},{"role":"user","content":$u}]}')

echo "Model use: $MODEL"
if [[ "$DEBUG" == "1" ]]; then
  echo "Q(messages):"
  echo "$MESSAGES" | jq .
fi

BASE_URL="http://127.0.0.1:${PORT}/v1/chat/completions"
RESPONSE=""
FINISH="length"
LOOP=0

# -------- loop until stop token or non-length finish --------
while [[ "$FINISH" == "length" && "$RESPONSE" != *"⟪END⟫"* && $LOOP -lt $MAX_LOOPS ]]; do
  LOOP=$((LOOP+1))

  PAYLOAD=$(jq -n \
    --arg model "$MODEL" \
    --argjson messages "$(echo "$MESSAGES" | jq '.messages')" \
    --argjson max_tokens "$MAX_TOKENS" \
    --argjson top_k "$TOP_K" \
    --argjson temperature "$TEMPERATURE" \
    --argjson top_p "$TOP_P" \
    --arg stop "$STOP_TOKEN" \
    '{
      model: $model,
      messages: $messages,
      temperature: $temperature,
      top_p: $top_p,
      top_k: $top_k,
      max_tokens: $max_tokens,
      stop: [$stop]
    }')

  if [[ "$DEBUG" == "1" ]]; then
    echo "PAYLOAD:"; echo "$PAYLOAD" | jq .
  fi

  RAW=$(curl --fail -S --max-time 300 -s "$BASE_URL" \
           -H "Content-Type: application/json" -d "$PAYLOAD") || {
    echo "❌ curl 失敗（API 未回應或連線/參數錯誤）" >&2
    exit 3
  }

  if [[ "$DEBUG" == "1" ]]; then
    echo "RAW:"; echo "$RAW" | jq .
  fi

  OUT=$(echo "$RAW" | jq -r '.choices[0].message.content')
  FINISH=$(echo "$RAW" | jq -r '.choices[0].finish_reason // "unknown"')

  echo -ne "$OUT"
  RESPONSE+="$OUT"

  MESSAGES=$(echo "$MESSAGES" | jq --arg content "$OUT" '.messages += [{"role":"assistant","content":$content}]')
done

# 結尾補換行（避免貼齊）
echo
if [[ "$RESPONSE" != *"⟪END⟫"* ]]; then
  echo "⚠️ 未偵測到 ⟪END⟫（finish_reason=$FINISH, loops=$LOOP）" >&2
fi

