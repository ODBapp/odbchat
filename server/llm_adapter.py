# server/llm_adapter.py
from __future__ import annotations
import os, json, logging, requests
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import time

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
        bad_tokens = {"<<<", "<<<MODE>>>", "<<<PLAN>>>", "<<<CODE>>>", "<<<ANSWER>>>", "<<<END>>>"}
        stops: list[str] = []
        for token in raw_stops:
            if not isinstance(token, str):
                continue
            cleaned = token.strip()
            if not cleaned or cleaned in bad_tokens:
                continue
            stops.append(cleaned)
        self.stops = stops
        self.last_error: Optional[str] = None
        self.last_success: Optional[float] = None

    def _ollama_base(self) -> str:
        url = self.ollama_chat or "http://localhost:11434/api/chat"
        return url.split("/api/")[0] if "/api/" in url else url.rstrip("/")

    def _llama_base(self) -> str:
        url = self.llama_chat or "http://localhost:8201/v1/chat/completions"
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
                self._mark_success()
                return names
            except Exception as e:
                logger.exception("[llm] ollama list_models failed: %s", e)
                self.last_error = f"list_models: {type(e).__name__}: {e}"
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
                        self._mark_success()
                        return names
            except Exception as e:
                logger.exception("[llm] llama-cpp list_models failed: %s", e)
                self.last_error = f"list_models: {type(e).__name__}: {e}"
            return [self.model]

        return [self.model]

    def set_model(self, model: str) -> bool:
        if not model:
            return False
        self.model = str(model).strip()
        os.environ["ODB_LLM_MODEL"] = self.model
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

    def _mark_success(self) -> None:
        self.last_error = None
        self.last_success = time.time()

    def _mark_failure(self, exc: Exception) -> None:
        self.last_error = f"{type(exc).__name__}: {exc}"

    def _ping_backend(self) -> tuple[bool, Optional[str]]:
        try:
            timeout = min(self.timeout, 5.0)
            if self.provider == "ollama":
                base = self._ollama_base()
                resp = requests.get(f"{base}/api/version", timeout=timeout)
                resp.raise_for_status()
            elif self.provider == "llama-cpp":
                base = self._llama_base()
                resp = requests.get(f"{base}/v1/models", timeout=timeout)
                resp.raise_for_status()
            else:
                return True, None
            return True, None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def get_status(self) -> dict:
        reachable, ping_error = self._ping_backend()
        return {
            "provider": self.provider,
            "model": self.model,
            "timeout": self.timeout,
            "reachable": reachable,
            "healthy": reachable and not self.last_error,
            "last_error": self.last_error,
            "last_success": self.last_success,
            "ping_error": ping_error,
        }
        
    # -------- public APIs --------
    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        temperature = self._eff_temp(temperature)
        max_tokens  = self._eff_maxtok(max_tokens)

        if DEBUG:
            logger.info("[llm] provider=%s model=%s temp=%.2f max_tokens=%d stops=%s",
                        self.provider, self.model, temperature, max_tokens, self.stops)

        if self.provider == "ollama":
            errors: list[Exception] = []
            try:
                txt = self._chat_ollama(messages, temperature, max_tokens)
                if txt.strip():
                    return txt.strip()
            except Exception as exc:
                errors.append(exc)
            try:
                txt = self._gen_ollama(messages, temperature, max_tokens)
                if txt.strip():
                    return txt.strip()
            except Exception as exc:
                errors.append(exc)
            msg = "Ollama request failed"
            if errors:
                msg = f"{msg}: {errors[-1]}"
            raise RuntimeError(msg)

        if self.provider == "llama-cpp":
            errors: list[Exception] = []
            try:
                txt = self._chat_llamacpp(messages, temperature, max_tokens)
                if txt.strip():
                    return txt.strip()
            except Exception as exc:
                errors.append(exc)
            try:
                txt = self._comp_llamacpp(messages, temperature, max_tokens)
                if txt.strip():
                    return txt.strip()
            except Exception as exc:
                errors.append(exc)
            msg = "llama.cpp request failed"
            if errors:
                msg = f"{msg}: {errors[-1]}"
            raise RuntimeError(msg)

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
                txt = (data.get("content") or data.get("choices",[{}])[0].get("text","") or "")
                if DEBUG:
                    logger.info("[llm] llama.cpp /completion ok, len=%d head=%r", len(txt), txt[:120])
                if not txt:
                    raise RuntimeError("llama.cpp completion returned empty response")
                self._mark_success()
                return txt
            except Exception as e:
                self._mark_failure(e)
                logger.exception("[llm] llama.cpp /completion failed: %s", e)
                raise

        logger.error("Unknown LLM provider for generate(): %s", self.provider)
        return ""

    # -------- providers (private) --------
    def _chat_ollama(self, messages, temperature, max_tokens) -> str:
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

        try:
            r = requests.post(self.ollama_chat, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            txt = (data.get("message",{}).get("content") or data.get("response") or "")
            if DEBUG:
                logger.info("[llm] ollama /api/chat len=%d head=%r", len(txt), txt[:120])
            if not txt:
                raise RuntimeError("Ollama chat returned empty response")
            self._mark_success()
            return txt
        except Exception as e:
            self._mark_failure(e)
            logger.exception("[llm] ollama /api/chat failed: %s", e)
            raise

    def _gen_ollama(self, messages, temperature, max_tokens) -> str:
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
        try:
            r = requests.post(self.ollama_gen, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            txt = (data.get("response") or data.get("message",{}).get("content","") or "")
            if DEBUG:
                logger.info("[llm] ollama /api/generate len=%d head=%r", len(txt), txt[:120])
            if not txt:
                raise RuntimeError("Ollama generate returned empty response")
            self._mark_success()
            return txt
        except Exception as e:
            self._mark_failure(e)
            logger.exception("[llm] ollama /api/generate failed: %s", e)
            raise

    def _chat_llamacpp(self, messages, temperature, max_tokens) -> str:
        payload = {
            "model": os.getenv("LLAMA_MODEL", "local"),
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
        try:
            r = requests.post(self.llama_chat, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            txt = (data.get("choices",[{}])[0].get("message",{}).get("content","")
                   or data.get("content","")
                   or "")
            if DEBUG:
                logger.info("[llm] llama.cpp /v1/chat/completions len=%d head=%r", len(txt), txt[:120])
            if not txt:
                raise RuntimeError("llama.cpp chat returned empty response")
            self._mark_success()
            return txt
        except Exception as e:
            self._mark_failure(e)
            logger.exception("[llm] llama.cpp /v1/chat/completions failed: %s", e)
            raise

    def _comp_llamacpp(self, messages, temperature, max_tokens) -> str:
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
        try:
            r = requests.post(self.llama_comp, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            txt = (data.get("content") or data.get("choices",[{}])[0].get("text","") or "")
            if DEBUG:
                logger.info("[llm] llama.cpp /completion len=%d head=%r", len(txt), txt[:120])
            if not txt:
                raise RuntimeError("llama.cpp completion returned empty response")
            self._mark_success()
            return txt
        except Exception as e:
            self._mark_failure(e)
            logger.exception("[llm] llama.cpp /completion failed: %s", e)
            raise

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
