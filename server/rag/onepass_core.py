from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from server.rag.onepass_prompts import build_continue_prompt, build_main_prompt
from server.llm_adapter import LLM

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer
from functools import lru_cache

_OAS_CACHE: dict[frozenset[str], Dict[str, Any]] = {}

try:
    from ollama import Client as OllamaClient
except ImportError:  # pragma: no cover
    OllamaClient = None

LOG_LEVEL = os.environ.get("ONEPASS_LOG_LEVEL", "INFO").upper()
LOGGER = logging.getLogger("onepass")
DEBUG = os.getenv("ONEPASS_DEBUG", "").strip() not in {"", "0", "false", "False"}
if not logging.getLogger().handlers:
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("odbchat.onepass")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "thenlper/gte-small")
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("ONEPASS_MODEL", os.environ.get("OLLAMA_MODEL", "gemma3:12b"))
DEFAULT_COLLECTIONS = [c.strip() for c in os.environ.get("ODB_ACTIVE_COLLECTIONS", "mhw").split(",") if c.strip()]
if not DEFAULT_COLLECTIONS:
    DEFAULT_COLLECTIONS = ["mhw"]

NOTES_MAX_CHARS = int(os.environ.get("ONEPASS_NOTES_MAX_CHARS", "1600"))
NOTES_SNIPPET_CHARS = int(os.environ.get("ONEPASS_SNIPPET_CHARS", "400"))
DEBUG_MODE = os.environ.get("ONEPASS_DEBUG", "0") == "1"
PROMPT_DIR = Path(__file__).resolve().parents[2] / "specs" / "prompts"
SYSTEM_PROMPT_FILE = PROMPT_DIR / "onepass.system.txt"
USER_PROMPT_FILE = PROMPT_DIR / "onepass.user_template.txt"
BLOCK_TAGS = ("MODE", "PLAN", "CODE", "ANSWER")

_embedder: Optional[SentenceTransformer] = None
_embedding_dim: Optional[int] = None
_qdrant: Optional[QdrantClient] = None
_ollama: Optional[OllamaClient] = None
_system_prompt: Optional[str] = None
_user_prompt_template: Optional[str] = None

# --- Hit-safe accessors (support Hit or dict) ---
def _hit_payload(h) -> dict:
    if isinstance(h, dict):
        return h.get("payload", h) or {}
    return getattr(h, "payload", {}) or {}

def _hit_doc_type(h) -> str:
    if isinstance(h, dict):
        p = h.get("payload", h) or {}
        return str(p.get("doc_type") or h.get("doc_type") or "")
    return str(getattr(h, "doc_type", "") or _hit_payload(h).get("doc_type", ""))

def _hit_title(h) -> str:
    if isinstance(h, dict):
        p = h.get("payload", h) or {}
        return str(h.get("title") or p.get("title") or p.get("doc_id") or p.get("source_file") or "(untitled)")
    p = _hit_payload(h)
    return str(getattr(h, "title", "") or p.get("title") or p.get("doc_id") or p.get("source_file") or "(untitled)")

def _hit_source_file(h) -> str:
    if isinstance(h, dict):
        p = h.get("payload", h) or {}
        return str(h.get("source_file") or p.get("source_file") or "")
    return str(getattr(h, "source_file", "") or _hit_payload(h).get("source_file", ""))

def _hit_score(h) -> float:
    if isinstance(h, dict):
        s = h.get("score")
        if s is None:
            s = (h.get("payload") or {}).get("score")
        try:
            return float(s or 0.0)
        except Exception:
            return 0.0
    try:
        return float(getattr(h, "score", 0.0) or 0.0)
    except Exception:
        return 0.0

def _payload_of(h):
    """Return payload dict if available, else {}."""
    if h is None:
        return {}
    if isinstance(h, dict):
        # prefer explicit payload if present
        p = h.get("payload")
        if isinstance(p, dict):
            return p
        return h
    # Qdrant point or dataclass
    p = getattr(h, "payload", None)
    return p if isinstance(p, dict) else {}

def _val(h, key, default=None):
    """Get value from hit or its payload/attributes."""
    if isinstance(h, dict):
        if key in h:
            return h.get(key, default)
        p = h.get("payload", {})
        if isinstance(p, dict):
            return p.get(key, default)
        return default
    # object path
    if hasattr(h, key):
        return getattr(h, key)
    p = _payload_of(h)
    return p.get(key, default)

def _text_of(h) -> str:
    """Unified text/snippet getter."""
    # direct text on dict
    if isinstance(h, dict):
        t = h.get("text") or (h.get("payload") or {}).get("text")
        if t:
            return str(t)
    # object
    t = getattr(h, "text", None)
    if isinstance(t, str) and t:
        return t
    p = _payload_of(h)
    return str(p.get("text") or p.get("snippet") or "")

def _score_of(h) -> float:
    s = _val(h, "score", None)
    try:
        return float(s) if s is not None else 0.0
    except Exception:
        return 0.0

def _norm_hit(h) -> dict:
    """Normalize to a dict we can work with downstream."""
    p = _payload_of(h)
    return {
        "text": _text_of(h),
        "score": _score_of(h),
        "payload": {
            **p,
            # ensure these are present at top-level of payload
            "doc_type": p.get("doc_type") or _val(h, "doc_type", ""),
            "title":    p.get("title")    or _val(h, "title", ""),
            "source_file": p.get("source_file") or _val(h, "source_file", ""),
            "chunk_id": int(p.get("chunk_id") if p.get("chunk_id") is not None else _val(h, "chunk_id", 0) or 0),
            "doc_id":   p.get("doc_id") or _val(h, "doc_id", None),
        }
    }

def _get_attr(obj, name, default=None):
    return getattr(obj, name, obj.get(name, default)) if isinstance(obj, dict) else getattr(obj, name, default)

def _compress_notes(hits, max_chars=3000, max_docs=6) -> str:
    """
    Balanced notes:
      - 先挑 1 個 api_spec + 1 個 code/guide
      - 再用分數高的補滿，其間全程以「文件鍵」去重
      - 不再用 set(Hit)（Hit 不可雜湊），改用 key set
    """
    def _key(h):
        p = getattr(h, "payload", {}) or {}
        # 以 doc_id 為主，fallback 到 (title, source_file)
        return p.get("doc_id") or (getattr(h, "title", ""), getattr(h, "source_file", ""), getattr(h, "chunk_id", 0))

    def _mk(h):
        p = getattr(h, "payload", {}) or {}
        title = getattr(h, "title", "") or p.get("title") or "(untitled)"
        src = getattr(h, "source_file", "") or p.get("source_file") or ""
        cid = getattr(h, "chunk_id", 0) if getattr(h, "chunk_id", None) is not None else p.get("chunk_id", 0)
        text = (getattr(h, "text", "") or p.get("text") or p.get("snippet") or "").strip()
        return f"[{getattr(h,'doc_type','')}] {title} ({src}#{cid})\n{text}"

    # 類別分組
    api   = [h for h in hits if getattr(h, "doc_type", "") == "api_spec"]
    codey = [h for h in hits if getattr(h, "doc_type", "") in {"code_snippet", "cli_tool_guide"}]

    # 用 key set 取代 set(Hit)
    seen_keys = set(_key(h) for h in (api + codey))
    rest = [h for h in hits if _key(h) not in seen_keys]

    # 組合順序：api_spec → code/guide → 其餘（保持原先排序/分數排序由外層決定）
    ordered = []
    if api:
        ordered.append(api[0])
    if codey:
        ordered.append(codey[0])
    for h in hits:
        k = _key(h)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        ordered.append(h)

    # 逐篇累加到上限
    parts, total, used = [], 0, 0
    for h in ordered:
        if used >= max_docs:
            break
        blob = _mk(h)
        if not blob:
            continue
        if total + len(blob) > max_chars:
            remain = max_chars - total
            if remain > 200:
                parts.append(blob[:remain])
                used += 1
            break
        parts.append(blob)
        total += len(blob)
        used += 1

    return "\n\n---\n\n".join(parts)

def _collections_key(cols: Optional[Sequence[str]]) -> frozenset[str]:
    cols = list(cols) if cols else DEFAULT_COLLECTIONS
    return frozenset(cols or DEFAULT_COLLECTIONS)

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.debug("Loading embedding model %s", EMBED_MODEL)
        _embedder = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    return _embedder

def embedding_dim() -> int:
    global _embedding_dim
    if _embedding_dim is not None:
        return _embedding_dim
    model = get_embedder()
    try:
        _embedding_dim = int(model.get_sentence_embedding_dimension())
    except Exception:  # pragma: no cover
        sample = model.encode(["dimension"], normalize_embeddings=True)
        _embedding_dim = len(sample[0])
    return _embedding_dim


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    return _qdrant


def get_ollama() -> Optional[OllamaClient]:  # pragma: no cover - simple accessor
    global _ollama
    if OllamaClient is None:
        return None
    if _ollama is None:
        _ollama = OllamaClient(host=OLLAMA_URL)
    return _ollama


def load_system_prompt() -> str:
    global _system_prompt
    if _system_prompt is None:
        try:
            _system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
        except FileNotFoundError:  # pragma: no cover
            logger.warning("System prompt %s not found; using fallback", SYSTEM_PROMPT_FILE)
            _system_prompt = "You are an expert ODB assistant."
    return _system_prompt


def load_user_prompt_template() -> str:
    global _user_prompt_template
    if _user_prompt_template is None:
        try:
            _user_prompt_template = USER_PROMPT_FILE.read_text(encoding="utf-8")
        except FileNotFoundError:  # pragma: no cover
            logger.warning("User prompt template %s not found; using fallback", USER_PROMPT_FILE)
            _user_prompt_template = "Question: {query}\nTopK: {top_k}\nNotes:\n{notes}\nOAS:\n{whitelist}"
    return _user_prompt_template


def encode_query(text: str) -> List[float]:
    vec = get_embedder().encode(text, normalize_embeddings=True)
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def get_payload_text(payload: Dict[str, Any]) -> str:
    if not payload:
        return ""
    if isinstance(payload.get("text"), str):
        return payload["text"]
    if isinstance(payload.get("content"), str):
        return payload["content"]
    return json.dumps({k: v for k, v in payload.items() if k not in {"vector"}}, ensure_ascii=False)


def get_title_url(payload: Dict[str, Any]) -> tuple[str, str]:
    title = payload.get("title") or payload.get("doc_id") or payload.get("source_file") or "(untitled)"
    url = payload.get("canonical_url") or payload.get("source_file") or ""
    return str(title), str(url)


def _norm_doc_type(doc_type: Optional[str]) -> str:
    if not isinstance(doc_type, str):
        return "n/a"
    s = doc_type.strip().lower().replace("-", "_")
    if s in {"oas", "openapi", "swagger", "api", "api_specification"}:
        return "api_spec"
    return s


def _force_include_api_spec(all_hits: Sequence[Hit], selected: List[Hit]) -> List[Hit]:
    if any(_norm_doc_type(h.doc_type) == "api_spec" for h in selected):
        return selected
    for cand in all_hits:
        if _norm_doc_type(cand.doc_type) == "api_spec":
            seen: set[str] = set()
            out: List[Hit] = [cand]
            seen.add(cand.id)
            for h in selected:
                if h.id in seen:
                    continue
                seen.add(h.id)
                out.append(h)
            return out
    return selected


def collect_rag_notes(hits: Sequence[Hit], max_chars: int = NOTES_MAX_CHARS) -> str:
    pieces: List[str] = []
    used = 0
    for hit in hits:
        text = (hit.text or "").strip()
        if not text:
            continue
        snippet = text[:NOTES_SNIPPET_CHARS]
        title = hit.title or "Untitled"
        block = f"### {title}\n{snippet}"
        pieces.append(block.strip())
        used += len(block)
        if used >= max_chars:
            break
    notes = "\n\n".join(pieces)
    return notes[:max_chars]


def _norm_append_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip().lower() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip().lower() for v in value]
    return [str(value).strip().lower()]

# ---- Hit helpers: keep Hit objects ----
CODE_LIKE = {"api_spec", "code_snippet", "cli_tool_guide"}

def dedupe_hits_hits(hits: List[Hit]) -> List[Hit]:
    seen: set[tuple] = set()
    out: List[Hit] = []
    for h in sorted(hits, key=lambda x: float(getattr(x, "score", 0.0) or 0.0), reverse=True):
        p = getattr(h, "payload", {}) or {}
        key = p.get("doc_id") or (getattr(h, "title", ""), getattr(h, "source_file", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out

def prefer_code_api_first_hits(hits: List[Hit]) -> List[Hit]:
    codey = [h for h in hits if (getattr(h, "doc_type", "") in CODE_LIKE)]
    other = [h for h in hits if (getattr(h, "doc_type", "") not in CODE_LIKE)]
    return codey + other

def rerank_diversify_hits(hits: List[Hit], k: int = 6) -> List[Hit]:
    if not hits:
        return []
    hits_sorted = sorted(hits, key=lambda x: float(getattr(x, "score", 0.0) or 0.0), reverse=True)
    api_like = [h for h in hits_sorted if getattr(h, "doc_type", "") in CODE_LIKE]
    rest     = [h for h in hits_sorted if getattr(h, "doc_type", "") not in CODE_LIKE]

    picked: List[Hit] = []
    if api_like:
        picked.append(api_like[0])

    groups: dict[tuple[str, str], Hit] = {}
    for h in rest:
        key = (getattr(h, "doc_type", "") or "", getattr(h, "source_file", "") or "")
        if key not in groups:
            groups[key] = h
    picked.extend(groups.values())

    if len(api_like) > 1:
        picked.extend(api_like[1:])

    picked.sort(key=lambda x: float(getattr(x, "score", 0.0) or 0.0), reverse=True)
    return picked[: max(k, 1)]

def _collect_api_specs_from_hits(hits: Sequence[Hit]) -> List[str]:
    texts: List[str] = []
    for hit in hits:
        payload = getattr(hit, "payload", None)
        if not isinstance(payload, dict):
            payload = getattr(hit, "__dict__", {}).get("payload", {}) or {}
        if _norm_doc_type(payload.get("doc_type")) != "api_spec":
            continue
        text = get_payload_text(payload)
        if text:
            texts.append(text)
    return texts

def _collect_api_specs_full_scan(limit: int = 512, collections: Optional[Sequence[str]] = None) -> List[str]:
    client = get_qdrant()
    texts: List[str] = []
    scroll_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="api_spec"))])

    collections = list(collections) if collections else DEFAULT_COLLECTIONS
    if not collections:
        collections = DEFAULT_COLLECTIONS

    for collection in collections:
        next_page = None
        while len(texts) < limit:
            res = client.scroll(
                collection_name=collection,
                scroll_filter=scroll_filter,
                limit=min(64, limit - len(texts)),
                with_payload=True,
                with_vectors=False,
                offset=next_page,
            )
            points, next_page = res
            if not points:
                break
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                text = get_payload_text(payload)
                if text:
                    texts.append(text)
                    if len(texts) >= limit:
                        break
            if next_page is None:
                break
        if len(texts) >= limit:
            break
    return texts


def _json_pointer(obj: Any, pointer: str) -> Any:
    if not pointer.startswith("#/"):
        return None
    current = obj
    for segment in pointer[2:].split("/"):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return None
    return current

_ALLOWED_DESC_RE = re.compile(r"Allowed fields\s*:\s*'([^']+)'(?:\s*,\s*'([^']+)')?(?:\s*,\s*'([^']+)')?(?:\s*,\s*'([^']+)')?", re.I)

def _parse_oas_text(raw: str) -> Dict[str, Any]:
    params: set[str] = set()
    paths: set[str] = set()
    append_allowed: set[str] = set()
    enums: Dict[str, List[str]] = {}

    if not raw:
        return {"params": [], "paths": [], "append_allowed": [], "param_enums": {}}

    try:
        data = yaml.safe_load(raw)
    except Exception:  # pragma: no cover
        data = None

    if isinstance(data, dict):
        for path_key in (data.get("paths") or {}):
            paths.add(path_key)

        comp_params = (data.get("components") or {}).get("parameters") or {}
        for name, entry in comp_params.items():
            entry = entry or {}
            pname = str(entry.get("name") or name)
            params.add(pname)
            if entry.get("schema") and isinstance(entry["schema"], dict):
                enum_vals = entry["schema"].get("enum") or []
                if enum_vals:
                    enums[pname] = [str(v) for v in enum_vals]

        for ref in (data.get("parameters") or []):
            if isinstance(ref, dict):
                if "$ref" in ref:
                    target = _json_pointer(data, ref["$ref"])
                    if isinstance(target, dict):
                        params.add(str(target.get("name") or ""))
                elif ref.get("name"):
                    params.add(str(ref.get("name")))

        # append allowed values from descriptions
        for _, entry in (data.get("paths") or {}).items():
            if not isinstance(entry, dict):
                continue
            for method in entry.values():
                if not isinstance(method, dict):
                    continue
                for param in method.get("parameters", []) or []:
                    if not isinstance(param, dict):
                        continue
                    pname = str(param.get("name") or "")
                    if pname:
                        params.add(pname)
                    desc = str(param.get("description") or "")
                    match = _ALLOWED_DESC_RE.search(desc)
                    if match:
                        for group in match.groups():
                            if group:
                                append_allowed.add(group)

    return {
        "params": sorted(params),
        "paths": sorted(paths),
        "append_allowed": sorted(append_allowed),
        "param_enums": enums,
    }

FENCED_RE = re.compile(r"```(?:python)?\n([\s\S]*?)```", re.I)
BLOCK_RE  = re.compile(r"<<<(MODE|PLAN|CODE|ANSWER)>>>([\s\S]*?)<<<END>>>", re.I)

def _extract_block(txt: str, name: str) -> str:
    m = re.search(rf"<<<{name}>>>([\s\S]*?)<<<END>>>", txt, re.I)
    return (m.group(1).strip() if m else "")

def _extract_first_fenced_code(txt: str) -> str:
    m = FENCED_RE.search(txt or "")
    return (m.group(1).strip() if m else "")

def _recover_plan_from_text_or_code(txt: str) -> Optional[Dict[str, Any]]:
    # 極簡啟發：找 {"endpoint": "...", "params": {...}} 片段
    try:
        cand = re.search(r"\{[^{}]*\"endpoint\"[^{}]*\{[\s\S]*?\}", txt)
        if cand:
            jtxt = cand.group(0)
            plan = json.loads(jtxt)
            if isinstance(plan, dict) and "endpoint" in plan and "params" in plan:
                return plan
    except Exception:
        pass
    return None

def _parse_onepass_output(raw: str) -> Dict[str, Any]:
    out = {
        "mode":   _extract_block(raw, "MODE").strip().lower(),
        "plan":   _extract_block(raw, "PLAN").strip(),
        "code":   _extract_block(raw, "CODE"),
        "answer": _extract_block(raw, "ANSWER"),
        "raw":    raw,
    }
    # 標準化 mode
    if out["mode"] not in ("code", "explain"):
        out["mode"] = "code" if out["code"] else ("explain" if out["answer"] else "")

    # mode=code 但無 code → 從 fenced code 補救
    if out["mode"] == "code" and not out["code"]:
        fenced = _extract_first_fenced_code(raw)
        if fenced:
            out["code"] = fenced

    # PLAN 缺 → 從 raw 嘗試恢復
    if out["mode"] == "code" and not out["plan"]:
        rec = _recover_plan_from_text_or_code(raw)
        if rec:
            out["plan"] = json.dumps(rec, ensure_ascii=False)

    # 仍無 code → 退回 explain
    if out["mode"] == "code" and not out["code"]:
        tmp = re.sub(r"```[\s\S]*?```", " ", raw)
        tmp = re.sub(r"~~~[\s\S]*?~~~", " ", tmp)
        tmp = re.sub(r"<<<[^>]+>>>", " ", tmp)
        ans = re.sub(r"\s+", " ", tmp).strip()
        if ans:
            out["mode"] = "explain"
            out["answer"] = out["answer"] or ans
    return out

@dataclass
class Hit:
    id: str
    score: float
    title: str
    doc_type: str
    source_file: str
    chunk_id: int
    text: str
    payload: Dict[str, Any]

@dataclass
class Citation:
    title: str
    source: str
    chunk_id: int

@dataclass
class OnePassResult:
    mode: str  # 'explain' | 'code'
    text: Optional[str] = None
    code: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    citations: List["Citation"] = None
    debug: Optional[Dict[str, Any]] = None  # <--- NEW

def _to_result(parsed: Dict[str, Any]) -> OnePassResult:
    """Convert a loose dict (from decide_and_generate) into OnePassResult."""
    mode = parsed.get("mode")
    if not mode:
        mode = "code" if parsed.get("code") else "explain"

    res = OnePassResult(mode=mode, citations=[])
    # text/answer
    res.text = parsed.get("text") or parsed.get("answer") or None
    # code
    res.code = parsed.get("code") or None
    # plan: allow str JSON or dict
    plan = parsed.get("plan")
    if isinstance(plan, str):
        try:
            res.plan = json.loads(plan)
        except Exception:
            res.plan = {"_raw": plan}
    elif isinstance(plan, dict):
        res.plan = plan
    else:
        res.plan = None

    # 把 debug/raw 也掛上，供後續續寫或 debug 用
    setattr(res, "debug", parsed.get("debug"))
    setattr(res, "raw", parsed.get("raw"))
    return res

def search_qdrant(query: str, k: int = 6, collections: Optional[List[str]] = None) -> List[Hit]:
    """
    Two-stage retrieval (general + code/api-biased) across collections,
    then dedupe + light diversify, finally return top-K Hit objects.
    """
    if not (query or "").strip():
        return []

    collections = collections or DEFAULT_COLLECTIONS
    vector = encode_query(query)
    client = get_qdrant()

    CODE_LIKE = {"api_spec", "code_snippet", "cli_tool_guide"}

    # --- Stage 1: general search (wider net) ---
    all_hits: List[Hit] = []
    limit_general = max(20, (k or 6) * 4)

    for coll in collections:
        try:
            resp = client.query_points(
                collection_name=coll,
                query=vector,
                limit=limit_general,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Qdrant general query failed for %s: %s", coll, exc)
            continue

        points = getattr(resp, "points", resp)
        for point in points or []:
            payload = getattr(point, "payload", {}) or {}
            text = get_payload_text(payload)
            score = float(getattr(point, "score", payload.get("score", 0.0)) or 0.0)
            chunk = payload.get("chunk_id")
            try:
                chunk_id = int(chunk) if chunk is not None else 0
            except Exception:
                chunk_id = 0
            title = str(payload.get("title") or payload.get("doc_id") or payload.get("source_file") or "(untitled)")
            hit = Hit(
                id=str(getattr(point, "id", payload.get("doc_id") or uuid.uuid4())),
                score=score,
                title=title,
                doc_type=str(payload.get("doc_type") or "n/a"),
                source_file=str(payload.get("source_file") or ""),
                chunk_id=chunk_id,
                text=text,
                payload=payload,
            )
            all_hits.append(hit)

    # --- Stage 2: biased search (prefer code/api/guide), best-effort ---
    for coll in collections:
        try:
            # Try to build models.Filter (qdrant_client.http.models). If not available, skip stage 2.
            try:
                from qdrant_client.http import models  # type: ignore
                code_filter = models.Filter(
                    should=[
                        models.FieldCondition(key="doc_type", match=models.MatchValue(value="api_spec")),
                        models.FieldCondition(key="doc_type", match=models.MatchValue(value="code_snippet")),
                        models.FieldCondition(key="doc_type", match=models.MatchValue(value="cli_tool_guide")),
                        models.FieldCondition(key="source_type", match=models.MatchText(text="code")),
                    ]
                )
                resp2 = client.query_points(
                    collection_name=coll,
                    query=vector,
                    limit=40,
                    with_payload=True,
                    with_vectors=False,
                    query_filter=code_filter,
                )
            except Exception:
                # If models API not available, skip biased pass for this collection.
                resp2 = None

            if not resp2:
                continue

            points2 = getattr(resp2, "points", resp2)
            for point in points2 or []:
                payload = getattr(point, "payload", {}) or {}
                text = get_payload_text(payload)
                score = float(getattr(point, "score", payload.get("score", 0.0)) or 0.0)
                chunk = payload.get("chunk_id")
                try:
                    chunk_id = int(chunk) if chunk is not None else 0
                except Exception:
                    chunk_id = 0
                title = str(payload.get("title") or payload.get("doc_id") or payload.get("source_file") or "(untitled)")
                hit = Hit(
                    id=str(getattr(point, "id", payload.get("doc_id") or uuid.uuid4())),
                    score=score,
                    title=title,
                    doc_type=str(payload.get("doc_type") or "n/a"),
                    source_file=str(payload.get("source_file") or ""),
                    chunk_id=chunk_id,
                    text=text,
                    payload=payload,
                )
                all_hits.append(hit)

        except Exception as exc:  # pragma: no cover
            logger.error("Qdrant biased query failed for %s: %s", coll, exc)

    # --- Dedupe (by doc_id or (title, source_file)) ---
    all_hits.sort(key=lambda h: h.score, reverse=True)
    deduped: List[Hit] = []
    seen: set[tuple[Any, Any]] = set()
    for h in all_hits:
        doc_id = (getattr(h, "payload", {}) or {}).get("doc_id")
        key = (doc_id or h.title, h.source_file)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)

    # --- Light diversify: ensure ≥1 code/api-like, then pick reps by (doc_type, source_file) ---
    if not deduped:
        logger.debug("search_qdrant retrieved 0 hits")
        return []

    # ensure at least one code/api-like
    api_like = [h for h in deduped if (h.doc_type in CODE_LIKE)]
    rest     = [h for h in deduped if (h.doc_type not in CODE_LIKE)]

    picked: List[Hit] = []
    if api_like:
        picked.append(api_like[0])  # highest-scored since deduped is sorted

    # group rest by (doc_type, source_file) and pick top-1 per group
    groups: Dict[tuple[str, str], Hit] = {}
    for h in rest:
        key = (h.doc_type or "", h.source_file or "")
        if key not in groups:
            groups[key] = h  # already sorted by score
    picked.extend(groups.values())

    # include remaining api_like (beyond the first) to improve OAS presence
    if len(api_like) > 1:
        picked.extend(api_like[1:])

    # final sort by score and cut to k
    picked.sort(key=lambda h: h.score, reverse=True)
    final_hits = picked[: max(k, 1)]

    logger.debug("search_qdrant retrieved %d hits (dedup %d → final %d)",
                 len(all_hits), len(deduped), len(final_hits))
    return final_hits

def harvest_oas_whitelist(hits: List[Hit]) -> Dict[str, Any]:
    texts = _collect_api_specs_from_hits(hits)

    def merge_into(dst, src):
        for name in src.get("params", []) or []: dst["params"].add(str(name))
        for p in src.get("paths", []) or []:     dst["paths"].add(str(p))
        for v in src.get("append_allowed", []) or []: dst["append_allowed"].add(str(v))
        for k, vals in (src.get("param_enums", {}) or {}).items():
            dst["param_enums"].setdefault(k, [])
            for v in vals:
                if v not in dst["param_enums"][k]:
                    dst["param_enums"][k].append(v)

    acc = {"params": set(), "paths": set(), "append_allowed": set(), "param_enums": {}}
    for raw in texts:
        merge_into(acc, _parse_oas_text(raw))

    if (len(acc["params"]) <= 1 or not acc["paths"]):
        cols = {h.payload.get("collection") for h in hits if h.payload}
        key = _collections_key([c for c in cols if isinstance(c, str) and c] or None)
        cached = _OAS_CACHE.get(key)
        if not cached:
            merged = {"params": set(), "paths": set(), "append_allowed": set(), "param_enums": {}}
            for raw in _collect_api_specs_full_scan(limit=256, collections=list(key)):
                merge_into(merged, _parse_oas_text(raw))
            cached = {
                "params": sorted(merged["params"]),
                "paths": sorted(merged["paths"]),
                "append_allowed": sorted(merged["append_allowed"]),
                "param_enums": merged["param_enums"],
            }
            _OAS_CACHE[key] = cached
        return cached

    return {
        "params": sorted(acc["params"]),
        "paths": sorted(acc["paths"]),
        "append_allowed": sorted(acc["append_allowed"]),
        "param_enums": acc["param_enums"],
    }

def decide_and_generate(
    *,
    query: str,
    notes: str,
    whitelist: Dict[str, Any],
    llm_id: str,
    max_continue: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    prompt = build_main_prompt(query=query, notes=notes or "", whitelist=whitelist, top_k=6)

    # 只在這裡印一次 prompt 尺寸（可選）
    if DEBUG_MODE or debug:
        logger.info("[onepass] llm_provider=%s llm_model=%s", LLM.provider, LLM.model)
        logger.info("[onepass] prompt_len=%d", len(prompt))

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user",   "content": "Follow STRICT OUTPUT FORMAT and answer now."},
    ]

    raw = LLM.chat(messages, temperature=0.2, max_tokens=1024) or ""
    if DEBUG_MODE or debug:
        logger.info("[onepass] raw_len=%d head=%r", len(raw), raw[:64])
    if not raw.strip():
        msg = "LLM returned empty response"
        if debug or DEBUG_MODE:
            raise RuntimeError(msg)
        logger.warning(f"{msg}; using explain fallback")
        return {
            "mode": "explain",
            "answer": "無法取得模型回覆，請確認 LLM 服務是否啟動。",
            "debug": {"mode": "explain", "raw": raw, "continued": False},
        }

    parsed = _parse_onepass_output(raw); parsed["raw"] = raw

    def _invalid(p: Dict[str, Any]) -> bool:
        m = (p.get("mode") or "").strip().lower()
        only_token = (raw.strip().lower() in {"code", "explain"})
        no_blocks = not ((p.get("code") or "").strip() or (p.get("answer") or "").strip())
        return only_token or (m in {"code","explain"} and no_blocks)

    if _invalid(parsed):
        fix_user = (
            "Your previous response was INVALID. "
            "Immediately output the full tagged blocks per STRICT OUTPUT FORMAT. "
            "Begin with '<<<MODE>>>', then (if MODE=code) include non-empty <<<PLAN>>> and <<<CODE>>>."
        )
        raw2 = LLM.chat([{"role":"system","content":prompt},{"role":"user","content":fix_user}], temperature=0.2, max_tokens=1024) or ""
        if DEBUG_MODE or debug:
            logger.info("[onepass] retry_raw_len=%d head=%r", len(raw2), raw2[:64])
        parsed2 = _parse_onepass_output(raw2); parsed2["raw"] = raw2
        if (parsed2.get("code") or parsed2.get("answer")):
            parsed = parsed2

    # 如果 explain 但沒有 answer，就塞 raw（避免空白）
    if (parsed.get("mode") == "explain") and not (parsed.get("answer") or "").strip():
        parsed["answer"] = (parsed.get("answer") or parsed.get("text") or parsed.get("raw") or "").strip()

    # 單次 continue（若 code 太短）
    code_text = parsed.get("code") or ""
    if max_continue and parsed.get("mode") == "code" and len(code_text) < 128:
        try:
            cont_prompt = build_continue_prompt(prev_raw=parsed.get("raw",""), prev_code=code_text, query=query, whitelist=whitelist)
            cont_msgs = [
                {"role": "system", "content": cont_prompt},
                {"role": "user",   "content": code_text},
            ]
            more = LLM.chat(cont_msgs, temperature=0.2, max_tokens=1024) or ""
            extra = _extract_block(more, "CODE") or _extract_first_fenced_code(more) or more.strip()
            if extra and len(extra) > len(code_text):
                parsed["code"] = extra
                parsed.setdefault("debug", {})["continued"] = True
        except Exception:
            pass

    return parsed

def enforce_static_guards(code: str) -> None:
    if not code or not isinstance(code, str):
        raise ValueError("Empty code")
    if "requests.get(" in code and "params=" not in code:
        raise ValueError("requests.get must use params=")
    if re.search(r"requests\.get\([^)]*\?[^)]*\)", code):
        raise ValueError("Avoid handcrafted querystrings in requests.get")
    m = re.search(r'url\s*=\s*[\'"]([^\'"]+)[\'"]', code)
    if m and m.group(1).startswith("/"):
        raise ValueError("Use absolute API URL (no relative '/api/...').")
    if re.search(r"api[_-]?key|token|authorization|bearer", code, re.I):
        raise ValueError("Code must not reference secrets/tokens")

def validate_plan(plan: Dict[str, Any], whitelist: Dict[str, Any]) -> None:
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a dict")
    endpoint = plan.get("endpoint")
    params = plan.get("params")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("Plan missing endpoint")
    if not isinstance(params, dict):
        raise ValueError("Plan params must be an object")
    allowed_paths = set(whitelist.get("paths") or [])
    if not allowed_paths:
        raise ValueError("No OAS paths available; cannot validate endpoint against spec.")
    if endpoint not in allowed_paths:
        raise ValueError(f"Endpoint '{endpoint}' not allowed by OAS")
    allowed_params = set(whitelist.get("params") or [])
    for key in params:
        if allowed_params and key not in allowed_params:
            raise ValueError(f"Parameter '{key}' not allowed")
    enums = whitelist.get("param_enums") or {}
    for key, value in params.items():
        if key in enums and enums[key]:
            allowed_values = {str(v) for v in enums[key]}
            if str(value) not in allowed_values:
                raise ValueError(f"Parameter '{key}' value '{value}' not allowed")
    append_allowed = {str(v).lower() for v in (whitelist.get("append_allowed") or [])}
    if append_allowed and "append" in params:
        forbidden = [v for v in _norm_append_values(params.get("append")) if v not in append_allowed]
        if forbidden:
            raise ValueError(f"Append contains values not allowed: {forbidden}")

    for key in ("start", "end"):
        value = params.get(key)
        if isinstance(value, str) and re.match(r"^\d{4}-\d{2}$", value):
            params[key] = f"{value}-01"


def format_citations(hits: List[Hit]) -> List[Citation]:
    citations: List[Citation] = []
    seen: set[tuple[str, str, int]] = set()
    for hit in hits:
        payload = hit.payload or {}
        title, url = get_title_url(payload)
        source = url or hit.source_file or payload.get("source_file", "")
        key = (title, source, hit.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(title=title, source=source, chunk_id=hit.chunk_id))
    return citations

def _looks_incomplete(code: str) -> bool:
    if not code: return True
    opens = sum(code.count(x) for x in ("(", "[", "{"))
    closes = sum(code.count(x) for x in (")", "]", "}"))
    if opens != closes: return True
    if code.strip().endswith(("#", "\\", ",")): return True
    if re.search(r"\.\.\.$|TODO|continue code", code, re.I): return True
    return False

def _continue_llm_code(prev_raw: str, prev_code: str, query: str, whitelist: dict) -> str:
    prompt = build_continue_prompt(prev_raw=prev_raw, prev_code=prev_code, query=query, whitelist=whitelist)
    raw2 = LLM.generate(prompt, temperature=0.2)
    code2 = _extract_block(raw2, "CODE") or _extract_first_fenced_code(raw2) or raw2.strip()
    return code2

def run_onepass(
    query: str,
    k: int = 6,
    collections: Optional[List[str]] = None,
    temperature: float = 0.2,
    debug: bool = False,
) -> OnePassResult:
    t0 = time.perf_counter()
    debug_enabled = DEBUG or DEBUG_MODE or debug

    hits_raw = search_qdrant(query, k=max(k, 1) * 3, collections=collections)

    # 去重（檔案層）→ code/api 優先 → 多樣化取前 k
    hits_stage = prefer_code_api_first_hits(dedupe_hits_hits(hits_raw))
    hits       = rerank_diversify_hits(hits_stage, k=k)
    t1 = time.perf_counter()

    whitelist = harvest_oas_whitelist(hits)
    t2 = time.perf_counter()

    # NEW: 由檢索結果組 notes
    notes = _compress_notes(hits, max_chars=3000, max_docs=6)
    if debug_enabled:
        nlen = len(notes or "")
        nhead = "\n".join((notes or "").splitlines()[:6])
        logger.info("[onepass] notes_len=%d preview:\n%s", nlen, nhead)

    llm_id = f"{LLM.provider}:{LLM.model}"
    result = decide_and_generate(
        query=query,
        notes=notes,
        whitelist=whitelist,
        llm_id=llm_id,
        max_continue=True,
        debug=debug_enabled,
    )
    if isinstance(result, dict):
        result = _to_result(result)
    t3 = time.perf_counter()

    raw_prev = getattr(result, "raw", "")
    continued = False
    if result.mode == "code" and _looks_incomplete(result.code or ""):
        try:
            more = _continue_llm_code(raw_prev or (result.code or ""), result.code or "", query, whitelist)
            if more and len(more) > len(result.code or ""):
                result.code = more
                continued = True
        except Exception:
            pass
    if debug_enabled:
        result.debug = (result.debug or {})
        result.debug["continued"] = continued

    plan_ok, guard_ok = None, None
    if result.plan and result.code:
        try:
            validate_plan(result.plan, whitelist); plan_ok = True
        except Exception as e:
            plan_ok = False
            if debug_enabled:
                LOGGER.info("[onepass] plan validation error: %s", e)
            raise
        try:
            enforce_static_guards(result.code); guard_ok = True
        except Exception as e:
            guard_ok = False
            if debug_enabled:
                LOGGER.info("[onepass] static guard error: %s", e)
            raise

    t4 = time.perf_counter()

    # citations (keep your existing implementation)
    result.citations = format_citations(hits)

    # Build debug payload
    if debug_enabled:
        wl = whitelist or {}
        wl_summary = {
            "paths_count": len(wl.get("paths") or []),
            "params_count": len(wl.get("params") or []),
            "append_allowed": list(wl.get("append_allowed") or []),
            "sample_paths": (wl.get("paths") or [])[:5],
            "sample_params": (wl.get("params") or [])[:8],
        }
        parsed_mode = (getattr(result, "mode", "") or "").strip().lower()
        code_len = len(result.code or "")
        debug_obj = result.debug or {}
        debug_obj.update(
            {
                "mode": parsed_mode,
                "code_len": code_len,
                "plan": result.plan,
                "whitelist": wl_summary,
                "continued": continued,
                "durations_ms": {
                    "search": int(1000 * (t1 - t0)),
                    "whitelist": int(1000 * (t2 - t1)),
                    "llm": int(1000 * (t3 - t2)),
                    "validate_guard": int(1000 * (t4 - t3)),
                    "total": int(1000 * (t4 - t0)),
                },
            }
        )
        result.debug = debug_obj

        LOGGER.info(
            "[onepass] mode=%s code_len=%s plan_ok=%s guard_ok=%s",
            parsed_mode,
            code_len,
            plan_ok,
            guard_ok,
        )
        LOGGER.info("[onepass] whitelist: %s", json.dumps(wl_summary, ensure_ascii=False))
        if result.plan:
            LOGGER.info("[onepass] plan: %s", json.dumps(result.plan, ensure_ascii=False))

    return result
