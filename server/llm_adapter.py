# server/llm_adapter.py
from __future__ import annotations
import os, json, logging, requests
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger("odbchat.llm")
DEBUG = os.getenv("ONEPASS_DEBUG", "").lower() in {"1","true","yes"}

def _parse_json_env(name: str, default):
    try:
        val = os.getenv(name, "")
        if not val:
            return default
        obj = json.loads(val)
        return obj if isinstance(obj, type(default)) else default
    except Exception:
        return default

class LLMAdapter:
    def __init__(self):
        self.provider = (os.getenv("ODB_LLM_PROVIDER") or "ollama").strip()
        self.model    = (os.getenv("ODB_LLM_MODEL") or "gemma3:4b").strip()
        self.timeout  = float(os.getenv("LLM_TIMEOUT", "120"))
        self.temp     = float(os.getenv("LLM_TEMP", "0.7"))
        self.top_p    = float(os.getenv("LLM_TOP_P", "0.9"))
        self.top_k    = int(os.getenv("LLM_TOP_K", "40"))
        self.min_p    = float(os.getenv("LLM_MIN_P", "0.05"))
        self.repeat_p = float(os.getenv("LLM_REPEAT_PENALTY", "1.05"))
        self.num_ctx  = int(os.getenv("LLM_NUM_CTX", "8192"))
        self.max_tok  = int(os.getenv("LLM_MAX_TOKENS", "1024"))

        # endpoints
        self.ollama_chat = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
        self.ollama_gen  = os.getenv("OLLAMA_GEN_URL",  "http://localhost:11434/api/generate")
        self.llama_chat  = os.getenv("LLAMA_CHAT_URL",  "http://localhost:8201/v1/chat/completions")
        self.llama_comp  = os.getenv("LLAMA_COMP_URL",  "http://localhost:8201/completion")

        # stop tokens（過濾掉會立即截斷 one-pass 區塊的危險字串）
        raw_stops = _parse_json_env("LLM_STOP", [])
        bad = {"<<<", "<<<MODE>>>", "<<<PLAN>>>", "<<<CODE>>>", "<<<ANSWER>>>", "<<<END>>>"}
        self.stops = [s for s in raw_stops if isinstance(s, str) and s and s not in bad]

        def _ollama_base(self) -> str:
            url = self.ollama_chat or "http://localhost:11434/api/chat"
            # 取到 /api 之前的 base
            return url.split("/api/")[0] if "/api/" in url else url.rstrip("/")

        def _llama_base(self) -> str:
            url = self.llama_chat or "http://localhost:8201/v1/chat/completions"
            # 取到 /v1 之前的 base
            return url.split("/v1/")[0] if "/v1/" in url else url.rstrip("/")

        def list_models(self) -> list[str]:
            """
            列出目前 provider 可用的模型清單：
            - Ollama: GET <base>/api/tags
            - llama.cpp: GET <base>/v1/models（OpenAI 相容）
            失敗時回退為 [self.model]
            """
            if self.provider == "ollama":
                try:
                    base = self._ollama_base()
                    r = requests.get(f"{base}/api/tags", timeout=self.timeout)
                    r.raise_for_status()
                    data = r.json() or {}
                    models = data.get("models") or []
                    names: list[str] = []
                    for m in models:
                        name = m.get("name") or m.get("model") or m.get("tag")
                        if name:
                            names.append(str(name))
                    if not names:
                        names = [self.model]
                    if DEBUG:
                        logger.info("[llm] ollama models: %s", names)
                    return names
                except Exception as e:
                    logger.exception("[llm] ollama list_models failed: %s", e)
                    return [self.model]

            if self.provider == "llama-cpp":
                try:
                    base = self._llama_base()
                    r = requests.get(f"{base}/v1/models", timeout=self.timeout)
                    if r.status_code == 200:
                        data = r.json() or {}
                        arr = data.get("data") or []
                        names: list[str] = []
                        for it in arr:
                            mid = it.get("id") or it.get("model")
                            if mid:
                                names.append(str(mid))
                        if names:
                            if DEBUG:
                                logger.info("[llm] llama-cpp models: %s", names)
                            return names
                except Exception as e:
                    logger.exception("[llm] llama-cpp list_models failed: %s", e)
                return [self.model]

            # 其他 provider：先回目前使用的
            return [self.model]

        def set_model(self, model: str) -> bool:
            if not model:
                return False
            self.model = str(model).strip()
            os.environ["ODB_LLM_MODEL"] = self.model  # 讓之後的讀取一致
            if DEBUG:
                logger.info("[llm] set model -> %s:%s", self.provider, self.model)
            return True

        def set_provider(self, provider: str) -> bool:
            p = (provider or "").strip()
            if not p:
                return False
            self.provider = p
            os.environ["ODB_LLM_PROVIDER"] = self.provider
            if DEBUG:
                logger.info("[llm] set provider -> %s", self.provider)
            return True
        
    # -------- public APIs --------
    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        temperature = self._eff_temp(temperature)
        max_tokens  = self._eff_maxtok(max_tokens)

        if DEBUG:
            logger.info("[llm] provider=%s model=%s temp=%.2f max_tokens=%d stops=%s",
                        self.provider, self.model, temperature, max_tokens, self.stops)

        if self.provider == "ollama":
            txt = self._chat_ollama(messages, temperature, max_tokens)
            if txt.strip():
                return txt.strip()
            # fallback to generate
            txt = self._gen_ollama(messages, temperature, max_tokens)
            return txt.strip()

        if self.provider == "llama-cpp":
            txt = self._chat_llamacpp(messages, temperature, max_tokens)
            if txt.strip():
                return txt.strip()
            # fallback to /completion
            txt = self._comp_llamacpp(messages, temperature, max_tokens)
            return txt.strip()

        # 其他 provider 可在此擴充
        logger.error("Unknown LLM provider: %s", self.provider)
        return ""

    def generate(self, prompt: str, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        """單純 prompt→text（優先走各家 completion 端點）。"""
        temperature = self._eff_temp(temperature)
        max_tokens  = self._eff_maxtok(max_tokens)

        if self.provider == "ollama":
            return self._gen_ollama([{"role":"user","content": prompt}], temperature, max_tokens).strip()

        if self.provider == "llama-cpp":
            # 直接打 /completion 較穩
            try:
                payload = {
                    "prompt": prompt,
                    "n_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "min_p": self.min_p,
                    "repeat_penalty": self.repeat_p,
                }
                if self.stops:
                    payload["stop"] = self.stops
                r = requests.post(self.llama_comp, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                txt = (data.get("content")
                       or data.get("choices",[{}])[0].get("text","")
                       or "")
                if DEBUG:
                    logger.info("[llm] llama.cpp /completion ok, len=%d head=%r", len(txt), txt[:120])
                return txt
            except Exception as e:
                logger.exception("[llm] llama.cpp /completion failed: %s", e)
                return ""

        logger.error("Unknown LLM provider for generate(): %s", self.provider)
        return ""

    # -------- providers (private) --------
    def _chat_ollama(self, messages, temperature, max_tokens) -> str:
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": m.get("role","user"), "content": m.get("content","")} for m in messages],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "min_p": self.min_p,
                    "repeat_penalty": self.repeat_p,
                    "num_ctx": self.num_ctx,
                    "num_predict": max_tokens,
                }
            }
            if self.stops:
                payload["options"]["stop"] = self.stops

            r = requests.post(self.ollama_chat, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            # 新版
            txt = (data.get("message",{}).get("content") or
                   data.get("response") or "")
            if DEBUG:
                logger.info("[llm] ollama /api/chat len=%d head=%r", len(txt), txt[:120])
            if txt:
                return txt
        except Exception as e:
            logger.exception("[llm] ollama /api/chat failed: %s", e)

        return ""

    def _gen_ollama(self, messages, temperature, max_tokens) -> str:
        try:
            flat = self._flatten_messages(messages)
            payload = {
                "model": self.model,
                "prompt": flat,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "min_p": self.min_p,
                    "repeat_penalty": self.repeat_p,
                    "num_ctx": self.num_ctx,
                    "num_predict": max_tokens,
                }
            }
            if self.stops:
                payload["options"]["stop"] = self.stops
            r = requests.post(self.ollama_gen, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            txt = (data.get("response")
                   or data.get("message",{}).get("content","")
                   or "")
            if DEBUG:
                logger.info("[llm] ollama /api/generate len=%d head=%r", len(txt), txt[:120])
            return txt
        except Exception as e:
            logger.exception("[llm] ollama /api/generate failed: %s", e)
            return ""

    def _chat_llamacpp(self, messages, temperature, max_tokens) -> str:
        try:
            payload = {
                "model": os.getenv("LLAMA_MODEL", "local"),  # 某些建置必填但忽略
                "messages": [{"role": m.get("role","user"), "content": m.get("content","")} for m in messages],
                "temperature": temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repeat_penalty": self.repeat_p,
                "max_tokens": max_tokens,
            }
            if self.stops:
                payload["stop"] = self.stops

            r = requests.post(self.llama_chat, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            txt = (data.get("choices",[{}])[0].get("message",{}).get("content","")
                   or data.get("content","")
                   or "")
            if DEBUG:
                logger.info("[llm] llama.cpp /v1/chat/completions len=%d head=%r", len(txt), txt[:120])
            return txt
        except Exception as e:
            logger.exception("[llm] llama.cpp /v1/chat/completions failed: %s", e)
            return ""

    def _comp_llamacpp(self, messages, temperature, max_tokens) -> str:
        try:
            flat = self._flatten_messages(messages)
            payload = {
                "prompt": flat,
                "n_predict": max_tokens,
                "temperature": temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repeat_penalty": self.repeat_p,
            }
            if self.stops:
                payload["stop"] = self.stops
            r = requests.post(self.llama_comp, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            txt = (data.get("content")
                   or data.get("choices",[{}])[0].get("text","")
                   or "")
            if DEBUG:
                logger.info("[llm] llama.cpp /completion len=%d head=%r", len(txt), txt[:120])
            return txt
        except Exception as e:
            logger.exception("[llm] llama.cpp /completion failed: %s", e)
            return ""

    # -------- helpers --------
    def _flatten_messages(self, messages: List[Dict[str,str]]) -> str:
        # 簡單安全地展開 messages 給 legacy completion
        parts = []
        for m in messages:
            role = (m.get("role") or "user").upper()
            parts.append(f"{role}:\n{m.get('content','')}")
        return "\n\n".join(parts)

    def _eff_temp(self, t):  return self.temp if t is None else float(t)
    def _eff_maxtok(self, n): return self.max_tok if n is None else int(n)


# 模組級單例（供 onepass_core 直接 import 使用）
LLM = LLMAdapter()
