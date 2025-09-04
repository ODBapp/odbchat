# gemma3:4b
wget -O google_gemma-3-4b-it-Q4_K_M.gguf https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf

# gemma3:12b
wget -O google_gemma-3-12b-it-Q8_0.gguf https://huggingface.co/bartowski/google_gemma-3-12b-it-GGUF/resolve/main/google_gemma-3-12b-it-Q8_0.gguf

# huggingface_hub cli cannot work
# python -m pip install -U "huggingface_hub[cli]"
# python -m huggingface_hub download bartowski/google_gemma-3-4b-it-GGUF --include "google_gemma-3-4b-it-Q4_K_M.gguf" --local-dir ./models

# Build llama.cpp for specific cuda
# check your current GPU
nvidia-smi
nvcc --version && which nvcc
sudo apt update
sudo apt-get install -y build-essential git ninja-build python3-pip
python3 -m pip install --user cmake
https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
sudo apt-get install -y gcc-9 g++-9
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDAToolkit_ROOT=/usr/local/cuda-11.8 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8/bin -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCMAKE_C_COMPILER=/usr/bin/gcc-9 -DCMAKE_CXX_COMPILER=/usr/bin/g++-9
cmake --build build -j

# execution for /home/odbadmin/backup/src/llama.cpp/build/bin/llama-server
# we had put this export PATH="$HOME/.local/bin:$PATH" in .bashrc
sudo ln -s /home/odbadmin/backup/src/llama.cpp/build/bin/llama-server ~/.local/bin
sudo ln -s /home/odbadmin/backup/src/llama.cpp/build/bin/llama-cli ~/.local/bin/

# run 12b model
llama-server -m /home/odbadmin/proj/odbchat/models/google_gemma-3-12b-it-Q8_0.gguf -c 4096 -ngl 20 --port 8001

# run 4b model
llama-server -m /home/odbadmin/proj/odbchat/models/google_gemma-3-4b-it-Q4_K_M.gguf -c 4096 -ngl 24 --port 8001

# curl to ask with prompts
curl -s http://127.0.0.1:8001/completion   -H "Content-Type: application/json"   -d '{
  "prompt": "以條列式、科學準確地解釋聖嬰（El Niño）與反聖嬰（La Niña）。請包含：(1) 定義與判別區（Niño‑3.4 海溫異常正/負），(2) Bjerknes 正回饋（風—海溫—熱躍層傾斜），(3) 熱躍層傾斜在聖嬰與反聖嬰的差異，(4) 赤道下傳 Kelvin 波與 Rossby 波角色，(5) Walker/Hadley 環流與降水帶的位移，(6) 對西北太平洋（含臺灣）颱風活動或雨量的典型影響（方向性即可）。請用繁體中文，**每個小點最多 1–2 句、避免贅述**，最後僅輸出【END】作結。",
  "temperature": 0.25,
  "top_p": 0.9,
  "top_k": 40,
  "repeat_penalty": 1.05,
  "n_predict": 420,
  "stop": ["【END】"]
}'

curl -s http://127.0.0.1:8001/completion -d '{
  "prompt": "Tell me about the Kuroshio Current.",
  "n_predict": 128
}'

curl -s http://127.0.0.1:8001/completion -d '{
  "prompt": "以正確海洋學術語回答：請用繁體中文介紹黑潮（Kuroshio Current）。請避免「黃流」等錯誤稱呼，說明其起源、路徑、海氣影響與生態意義。",
  "temperature": 0.3,
  "n_predict": 180
}'

# Using chat_loop.sh (It can automatically continue the asking untile answer finished even using shorter token)
./chat_loop.sh -m 4b -q "Tell me about the Kuroshio Current" -l en -t 128

./chat_loop.sh -m 4b -q "以條列式、科學準確地解釋聖嬰（El Niño）與反聖嬰（La Niña）。請包含：(1) 定義與判別區（Niño‑3.4 海溫異常正/負），(2) Bjerknes 正回饋（風—海溫—熱躍層傾斜），(3) 熱躍層傾斜在聖嬰與反聖嬰的差異，(4) 赤道下傳 Kelvin 波與 Rossby 波角色，(5) Walker/Hadley 環流與降水帶的位移，(6) 對西北太平洋（含臺灣）颱風活動或雨量的典型影響（方向性即可）。" -t 480

# Ingest RAG
python dev/ingest_mhw.py --root rag --dry-run
python dev/ingest_mhw.py --root rag --collection odb_mhw_knowledge_v1

