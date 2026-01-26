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
from functools import lru_cache

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

from server.rag.onepass_prompts import build_continue_prompt, build_main_prompt
from server.llm_adapter import LLM
from server.time_utils import now_iso_in_tz, resolve_tz_name, today_in_tz
from server.keyword_sets import (
    CODE_KEYWORDS,
    CODE_REGEX,
    FOLLOWUP_KEYWORDS,
    GHRSST_GEO_REGEX,
    GHRSST_REGEX,
    NO_CODE_REGEX,
    PLOT_TERM_REGEX,
    TIME_HINT_REGEX,
    TIDE_REGEX,
    CLI_TOOL_REGEX,
)

_OAS_CACHE: dict[frozenset[str], Dict[str, Any]] = {}
LAST_PLAN: Optional[Dict[str, Any]] = None

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
def get_active_collections() -> List[str]:
    source = os.environ.get("ODB_ONEPASS_SOURCE", "yaml").lower()
    if source == "omnipipe":
        collections = [
            c.strip() for c in os.environ.get("OMNIPIPE_COLLECTIONS", "omnipipe_default").split(",") if c.strip()
        ]
        return collections or ["omnipipe_default"]
    collections = [c.strip() for c in os.environ.get("ODB_ACTIVE_COLLECTIONS", "mhw").split(",") if c.strip()]
    return collections or ["mhw"]



NOTES_MAX_CHARS = int(os.environ.get("ONEPASS_NOTES_MAX_CHARS", "1600"))
NOTES_MAX_DOCS = int(os.environ.get("ONEPASS_NOTES_MAX_DOCS", "4"))
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
COORD_PAIR_RE = re.compile(r"\(\s*-?\d{1,3}(?:\.\d+)?,\s*-?\d{1,2}(?:\.\d+)?\s*\)")
BBOX_RE = re.compile(
    r"\[\s*-?\d{1,3}(?:\.\d+)?(?:\s*,\s*-?\d{1,3}(?:\.\d+)?){3,}\s*\]"
)

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


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _query_is_meaningful(query: str) -> bool:
    if not query or not query.strip():
        return False
    meaningful = sum(1 for ch in query if ch.isalnum() or _is_cjk(ch))
    if meaningful < 2:
        return False
    condensed = "".join(ch for ch in query if not ch.isspace())
    if condensed and len(set(condensed)) <= 1:
        return False
    return True


def _wants_code(query: str) -> bool:
    if not query:
        return False
    lowered = query.lower()
    if NO_CODE_REGEX.search(lowered):
        return False
    if CLI_TOOL_REGEX.search(lowered):
        return False
    return bool(CODE_REGEX.search(lowered))


def _tokenize_for_overlap(text: str) -> set[str]:
    if not text:
        return set()
    tokens = set(re.findall(r"[a-z0-9_]{2,}", text.lower()))
    tokens.update(ch for ch in text if _is_cjk(ch))
    return tokens


def _score_bonus_from_payload(payload: Dict[str, Any], query: str) -> float:
    if not payload or not query:
        return 0.0
    strong_tokens = set(re.findall(r"[a-z0-9_]{2,}", query.lower()))
    tokens = strong_tokens or _tokenize_for_overlap(query)
    if not tokens:
        return 0.0
    fields = []
    for key in ("title", "dataset_name", "doc_type", "source_file", "caption", "table_label"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            fields.append(val)
    tags = payload.get("tags")
    if isinstance(tags, list):
        fields.extend(str(t) for t in tags if t)
    text = " ".join(fields)
    overlap = len(tokens & _tokenize_for_overlap(text))
    return 0.15 * overlap


def _lexical_filter_hits(query: str, hits: Sequence["Hit"]) -> List["Hit"]:
    strong_tokens = set(re.findall(r"[a-z0-9_]{2,}", query.lower()))
    tokens = strong_tokens or _tokenize_for_overlap(query)
    if not tokens:
        return list(hits)
    matched: List[Hit] = []
    for hit in hits:
        payload = getattr(hit, "payload", {}) or {}
        fields = [
            str(payload.get("title") or ""),
            str(payload.get("dataset_name") or ""),
            str(payload.get("doc_type") or ""),
            str(payload.get("source_file") or ""),
            str(payload.get("caption") or ""),
            str(payload.get("table_label") or ""),
        ]
        tags = payload.get("tags")
        if isinstance(tags, list):
            fields.extend(str(t) for t in tags if t)
        hay = " ".join(f for f in fields if f).lower()
        if not hay:
            continue
        if any(tok in hay for tok in tokens):
            matched.append(hit)
    return matched or list(hits)


def _looks_like_table_request(query: str) -> bool:
    if not query:
        return False
    lowered = query.lower()
    if "table" in lowered:
        return True
    if any(token in query for token in ("表格", "表 4", "表4", "表 3", "表3")):
        return True
    if ("depth" in lowered or "深度" in query) and ("variable" in lowered or "變數" in query):
        return True
    return False


def _pick_best_table_hit(query: str, hits: Sequence["Hit"]) -> Optional["Hit"]:
    if not hits:
        return None
    q = (query or "").lower()
    wants_table4 = any(token in q for token in ("table 4", "表4", "表 4"))
    candidates = [h for h in hits if getattr(h, "doc_type", "") == "table"]
    if not candidates:
        return None

    tokens = _table_query_tokens(query)

    def score(hit: "Hit") -> int:
        payload = getattr(hit, "payload", {}) or {}
        caption = str(payload.get("caption") or "").lower()
        text = str(payload.get("text") or "").lower()
        markdown = str(payload.get("markdown_content") or "").lower()
        hay = " ".join([caption, text, markdown])
        s = 0
        if wants_table4 and "table 4" in caption:
            s += 50
        if "depth" in q and "depth" in caption:
            s += 20
        if "variable" in q and "variable" in caption:
            s += 20
        if "oceanographic" in q and "oceanographic" in caption:
            s += 10
        if "table 4" in text:
            s += 5
        if tokens:
            overlap = len(tokens & _tokenize_for_overlap(hay))
            s += overlap
        return s

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def _table_query_tokens(query: str) -> set[str]:
    tokens = _tokenize_for_overlap(query)
    if not tokens:
        return set()
    expansions = {
        "深度": "depth",
        "變數": "variable",
        "範圍": "range",
        "海洋學": "oceanographic",
        "表格": "table",
    }
    expanded = set(tokens)
    for key, value in expansions.items():
        if key in query:
            expanded.update(_tokenize_for_overlap(value))
    return expanded


def _fetch_table_candidates(query: str, collections: Sequence[str], limit: int = 20) -> List["Hit"]:
    client = get_qdrant()
    vector = encode_query(query)
    hits: List[Hit] = []
    for coll in collections:
        try:
            from qdrant_client.http import models  # type: ignore
            table_filter = models.Filter(
                must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="table"))]
            )
            resp = client.query_points(
                collection_name=coll,
                query=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                query_filter=table_filter,
            )
        except Exception:
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
            hits.append(
                Hit(
                    id=str(getattr(point, "id", payload.get("doc_id") or uuid.uuid4())),
                    score=score,
                    title=title,
                    doc_type=str(payload.get("doc_type") or "n/a"),
                    source_file=str(payload.get("source_file") or ""),
                    chunk_id=chunk_id,
                    text=text,
                    payload=payload,
                )
            )
    return hits



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

def _compress_notes(hits, max_chars=NOTES_MAX_CHARS, max_docs=NOTES_MAX_DOCS, query: str = "") -> str:
    """
    Balanced notes:
      - code 類問題：先挑 1 個 api_spec + 1 個 code/guide
      - 非 code 類問題：以排序後結果為主，不強制特定 doc_type
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
        if not text:
            return ""
        doc_type = getattr(h, "doc_type", "")
        max_snippet = NOTES_SNIPPET_CHARS
        if doc_type in {"table", "image_description"}:
            max_snippet = min(len(text), max(NOTES_SNIPPET_CHARS, 1400))
        snippet = text[:max_snippet]
        return f"[{getattr(h,'doc_type','')}] {title} ({src}#{cid})\n{snippet}"

    want_code = _wants_code(query or "")

    api = [h for h in hits if getattr(h, "doc_type", "") == "api_spec"]
    codey = [h for h in hits if getattr(h, "doc_type", "") in {"code_snippet", "cli_tool_guide"}]

    if want_code:
        # 用 key set 取代 set(Hit)
        seen_keys = set(_key(h) for h in (api + codey))
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
    else:
        ordered = []
        seen_keys = set()
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
    cols = list(cols) if cols else get_active_collections()
    return frozenset(cols or get_active_collections())

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


def _is_remote_source(value: str) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def get_title_url(payload: Dict[str, Any]) -> tuple[str, str]:
    title = payload.get("title") or payload.get("doc_id") or "(untitled)"
    url = payload.get("canonical_url") or ""
    issuer = payload.get("issuer") or ""
    source_file = payload.get("source_file") or ""

    if not url and _is_remote_source(source_file):
        url = source_file

    label = ""
    if issuer and url:
        label = f"{issuer} — {url}"
    elif issuer:
        label = str(issuer)
    elif url:
        label = str(url)

    return str(title), str(label)


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


def _select_citation_hits(hits: Sequence[Hit], answer_text: str, limit: int) -> List[Hit]:
    if not hits:
        return []
    limit = max(1, limit or 1)
    tokens = _tokenize_for_overlap(answer_text)
    if not tokens:
        return list(hits[:limit])
    scored: List[tuple[float, int, Hit]] = []
    for idx, hit in enumerate(hits):
        snippet_tokens = _tokenize_for_overlap(_text_of(hit))
        overlap = len(tokens & snippet_tokens)
        if overlap <= 0:
            continue
        score = overlap + 0.1 * max(0.0, _score_of(hit))
        scored.append((score, idx, hit))
    if not scored:
        return list(hits[:limit])
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [hit for _, _, hit in scored[:limit]]


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


def _omnipipe_source_enabled() -> bool:
    return os.environ.get("ODB_ONEPASS_SOURCE", "yaml").lower() == "omnipipe"


def _link_expansion_enabled() -> bool:
    flag = os.environ.get("OMNIPIPE_LINK_EXPANSION", "").strip().lower()
    return flag not in {"", "0", "false", "no"}


def expand_hits_via_links(hits: Sequence["Hit"], client: QdrantClient, limit: int = 6) -> List["Hit"]:
    if not hits or limit <= 0:
        return []
    allowed = {"references", "related_to", "parent"}
    seen_artifacts: set[str] = set()
    expanded: List[Hit] = []

    for hit in hits:
        payload = getattr(hit, "payload", {}) or {}
        artifact_id = payload.get("artifact_id") or payload.get("doc_id")
        if artifact_id:
            seen_artifacts.add(str(artifact_id))

    for hit in hits:
        if len(expanded) >= limit:
            break
        payload = getattr(hit, "payload", {}) or {}
        links = payload.get("links") or []
        collection = payload.get("collection")
        if not collection or not isinstance(links, list):
            continue
        for link in links:
            if len(expanded) >= limit:
                break
            if not isinstance(link, dict):
                continue
            link_type = link.get("link_type")
            if link_type not in allowed:
                continue
            target_id = link.get("target_id")
            if not target_id:
                continue
            target_id = str(target_id)
            if target_id in seen_artifacts:
                continue
            try:
                points, _ = client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(must=[FieldCondition(key="artifact_id", match=MatchValue(value=target_id))]),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:  # pragma: no cover
                continue
            if not points:
                continue
            point = points[0]
            payload2 = getattr(point, "payload", {}) or {}
            text = get_payload_text(payload2)
            chunk = payload2.get("chunk_id")
            try:
                chunk_id = int(chunk) if chunk is not None else 0
            except Exception:
                chunk_id = 0
            title = str(payload2.get("title") or payload2.get("doc_id") or payload2.get("source_file") or "(untitled)")
            base_score = float(getattr(hit, "score", 0.0) or 0.0)
            confidence = 1.0
            metadata = link.get("metadata") if isinstance(link.get("metadata"), dict) else {}
            if metadata and "confidence" in metadata:
                try:
                    confidence = float(metadata.get("confidence", 1.0))
                except Exception:
                    confidence = 1.0
            score = base_score * confidence
            expanded.append(
                Hit(
                    id=str(getattr(point, "id", payload2.get("doc_id") or uuid.uuid4())),
                    score=score,
                    title=title,
                    doc_type=str(payload2.get("doc_type") or "n/a"),
                    source_file=str(payload2.get("source_file") or ""),
                    chunk_id=chunk_id,
                    text=text,
                    payload=payload2,
                )
            )
            seen_artifacts.add(target_id)
    return expanded

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


def _collect_api_endpoints_from_hits(hits: Sequence[Hit]) -> Dict[str, Any]:
    paths: set[str] = set()
    params: set[str] = set()
    append_allowed: set[str] = set()
    param_enums: Dict[str, List[str]] = {}
    servers: set[str] = set()
    for hit in hits:
        payload = getattr(hit, "payload", None)
        if not isinstance(payload, dict):
            payload = getattr(hit, "__dict__", {}).get("payload", {}) or {}
        doc_type = _norm_doc_type(payload.get("doc_type"))
        if doc_type not in {"api_endpoint"} and not payload.get("path"):
            continue
        path = payload.get("path")
        if isinstance(path, str) and path.strip():
            paths.add(path.strip())
        parameters = payload.get("parameters") or []
        if isinstance(parameters, list):
            for param in parameters:
                if not isinstance(param, dict):
                    continue
                name = param.get("name") or param.get("param") or param.get("id")
                if isinstance(name, str) and name.strip():
                    name = name.strip()
                    params.add(name)
                    schema = param.get("schema") if isinstance(param.get("schema"), dict) else {}
                    enum_vals = schema.get("enum") if isinstance(schema, dict) else None
                    if isinstance(enum_vals, list) and enum_vals:
                        param_enums.setdefault(name, [])
                        for v in enum_vals:
                            sv = str(v)
                            if sv not in param_enums[name]:
                                param_enums[name].append(sv)
                        if name == "append":
                            append_allowed.update(str(v) for v in enum_vals if v is not None)
                    if name == "append":
                        for desc in (param.get("description"), schema.get("description")):
                            for value in _extract_allowed_fields_from_text(desc):
                                append_allowed.add(value)
    return {
        "paths": sorted(paths),
        "params": sorted(params),
        "append_allowed": sorted(append_allowed),
        "param_enums": param_enums,
        "servers": sorted(servers),
    }

def _collect_api_specs_full_scan(limit: int = 512, collections: Optional[Sequence[str]] = None) -> List[str]:
    client = get_qdrant()
    texts: List[str] = []
    scroll_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="api_spec"))])

    collections = list(collections) if collections else get_active_collections()
    if not collections:
        collections = get_active_collections()

    for collection in collections:
        next_page = None
        while len(texts) < limit:
            try:
                res = client.scroll(
                    collection_name=collection,
                    scroll_filter=scroll_filter,
                    limit=min(64, limit - len(texts)),
                    with_payload=True,
                    with_vectors=False,
                    offset=next_page,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("scroll api_spec failed for %s: %s", collection, exc)
                break
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


def _collect_api_endpoints_full_scan(limit: int = 512, collections: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    client = get_qdrant()
    scroll_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="api_endpoint"))])
    collections = list(collections) if collections else get_active_collections()
    if not collections:
        collections = get_active_collections()
    acc = {"paths": set(), "params": set(), "append_allowed": set(), "param_enums": {}, "servers": set()}

    for collection in collections:
        next_page = None
        while len(acc["paths"]) < limit:
            try:
                res = client.scroll(
                    collection_name=collection,
                    scroll_filter=scroll_filter,
                    limit=128,
                    with_payload=True,
                    with_vectors=False,
                    offset=next_page,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("scroll api_endpoint failed for %s: %s", collection, exc)
                break
            points, next_page = res
            if not points:
                break
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                if not isinstance(payload, dict):
                    continue
                path = payload.get("path")
                if isinstance(path, str) and path.strip():
                    acc["paths"].add(path.strip())
                parameters = payload.get("parameters") or []
                if isinstance(parameters, list):
                    for param in parameters:
                        if not isinstance(param, dict):
                            continue
                        name = param.get("name") or param.get("param") or param.get("id")
                        if isinstance(name, str) and name.strip():
                            name = name.strip()
                            acc["params"].add(name)
                            schema = param.get("schema") if isinstance(param.get("schema"), dict) else {}
                            enum_vals = schema.get("enum") if isinstance(schema, dict) else None
                            if isinstance(enum_vals, list) and enum_vals:
                                acc["param_enums"].setdefault(name, [])
                                for v in enum_vals:
                                    sv = str(v)
                                    if sv not in acc["param_enums"][name]:
                                        acc["param_enums"][name].append(sv)
                                if name == "append":
                                    acc["append_allowed"].update(str(v) for v in enum_vals if v is not None)
                            if name == "append":
                                for desc in (param.get("description"), schema.get("description")):
                                    for value in _extract_allowed_fields_from_text(desc):
                                        acc["append_allowed"].add(value)
            if next_page is None:
                break

    return {
        "paths": sorted(acc["paths"]),
        "params": sorted(acc["params"]),
        "append_allowed": sorted(acc["append_allowed"]),
        "param_enums": acc["param_enums"],
        "servers": sorted(acc["servers"]),
    }


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

_ALLOWED_DESC_RE = re.compile(
    r"Allowed fields\s*:\s*'([^']+)'(?:\s*,\s*'([^']+)')?(?:\s*,\s*'([^']+)')?(?:\s*,\s*'([^']+)')?",
    re.I,
)

def _extract_allowed_fields_from_text(text: str) -> List[str]:
    if not text:
        return []
    match = _ALLOWED_DESC_RE.search(str(text))
    if not match:
        return []
    return [group for group in match.groups() if group]

def _parse_oas_text(raw: str) -> Dict[str, Any]:
    params: set[str] = set()
    paths: set[str] = set()
    append_allowed: set[str] = set()
    servers: set[str] = set()
    enums: Dict[str, List[str]] = {}

    if not raw:
        return {"params": [], "paths": [], "append_allowed": [], "param_enums": {}, "servers": []}

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

        for srv in data.get("servers", []) or []:
            if isinstance(srv, dict):
                url = srv.get("url")
                if isinstance(url, str) and url.strip():
                    servers.add(url.strip())

    return {
        "params": sorted(params),
        "paths": sorted(paths),
        "append_allowed": sorted(append_allowed),
        "param_enums": enums,
        "servers": sorted(servers),
    }

FENCED_RE = re.compile(r"```(?:python)?\n([\s\S]*?)```", re.I)
BLOCK_RE  = re.compile(r"<<<(MODE|PLAN|CODE|ANSWER|MCP)>>>([\s\S]*?)<<<END>>>", re.I)
TAG_FIX_RE = re.compile(r"<<<(MODE|PLAN|CODE|ANSWER|MCP)>>(?!>)", re.I)
TAG_FIX_ONE_RE = re.compile(r"<<<(MODE|PLAN|CODE|ANSWER|MCP)>(?!>)", re.I)

def _extract_block(txt: str, name: str) -> str:
    m = re.search(rf"<<<{name}>>>([\s\S]*?)<<<END>>>", txt, re.I)
    return (m.group(1).strip() if m else "")

def _extract_first_fenced_code(txt: str) -> str:
    m = FENCED_RE.search(txt or "")
    return (m.group(1).strip() if m else "")

def _normalize_inline_code(code: str) -> str:
    if not code:
        return code
    if code.count("\n") > 2:
        return code
    normalized = code.strip()
    patterns = [
        r"import ", r"from ", r"BASE_URL", r"endpoint", r"params\s*=", r"response\s*=",
        r"r\s*=", r"df\s*=", r"data\s*=", r"plt\.", r"pd\.", r"np\.", r"print\(",
        r"return ", r"for ", r"while ", r"if ", r"plt\.figure", r"plt\.plot", r"plt\.pcolormesh",
    ]
    for pat in patterns:
        normalized = re.sub(r"\s*(?=" + pat + ")", "\n", normalized)
    replacements = {
        ') r.': ')\nr.',
        ') df': ')\ndf',
        ') data': ')\ndata',
        ') print': ')\nprint',
        'df =\npd.': 'df = pd.',
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    normalized = re.sub(r"requests\.get\s*\(\s*BASE_URL\s*\+\s*endpoint,", "requests.get(BASE_URL + endpoint,", normalized)
    normalized = normalized.replace("requests.get(BASE_URL + endpoint,\nparams=", "requests.get(BASE_URL + endpoint, params=")
    normalized = normalized.replace("pd.read_csv(\nio.StringIO", "pd.read_csv(io.StringIO")
    normalized = normalized.replace("print(df)\nelse:", "print(df)\nelse:")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    cleaned = []
    indent = 0
    for line in lines:
        if line.startswith("else"):
            indent = max(indent - 1, 0)
        cleaned.append("    " * indent + line)
        if line.startswith("if ") and line.endswith(":"):
            indent += 1
        if line.startswith("else:"):
            indent += 1
        if line.startswith("return"):
            indent = max(indent - 1, 0)
    normalized = "\n".join(cleaned)
    normalized = re.sub(r"\n{2,}", "\n", normalized)
    return normalized.strip()

def _grab_block_flexible(raw_text: str, tag: str) -> str:
    if not raw_text:
        return ""
    pattern = re.compile(rf"<<<{tag}>>>", re.I)
    m = pattern.search(raw_text)
    if not m:
        return ""
    start = m.end()
    remainder = raw_text[start:]
    m_end = re.search(r"<<<END>>>", remainder, re.I)
    if m_end:
        return remainder[: m_end.start()].strip()
    m_next = re.search(r"<<<[A-Z_]+>>>", remainder, re.I)
    if m_next:
        return remainder[: m_next.start()].strip()
    return remainder.strip()

def _recover_plan_from_text_or_code(txt: str) -> Optional[tuple[Dict[str, Any], int, int]]:
    decoder = json.JSONDecoder()
    idx = txt.find('"endpoint"')
    while idx != -1:
        brace = txt.rfind('{', 0, idx)
        if brace == -1:
            break
        fragment = txt[brace:]
        try:
            obj, consumed = decoder.raw_decode(fragment)
            if isinstance(obj, dict) and obj.get("endpoint") and obj.get("params"):
                return obj, brace, brace + consumed
        except Exception:
            pass
        idx = txt.find('"endpoint"', idx + 10)
    return None

def _parse_onepass_output(raw: str) -> Dict[str, Any]:
    txt = raw or ""
    # strip any Markdown fences or language labels like ```json / ```python that bracket the blocks
    txt = re.sub(r"```[a-zA-Z0-9_+\-]*", "", txt)
    # fix common mistakes where model emits <<<TAG>> or <<<TAG>
    txt = TAG_FIX_RE.sub(lambda m: f"<<<{m.group(1).upper()}>>>", txt)
    txt = TAG_FIX_ONE_RE.sub(lambda m: f"<<<{m.group(1).upper()}>>>", txt)
    raw = txt
    out = {
        "mode":   _extract_block(raw, "MODE").strip().lower(),
        "plan":   _extract_block(raw, "PLAN").strip(),
        "code":   _extract_block(raw, "CODE"),
        "answer": _extract_block(raw, "ANSWER"),
        "mcp":    _extract_block(raw, "MCP").strip(),
        "raw":    raw,
    }
    # 標準化 mode
    allowed_modes = {"code", "explain", "fallback", "mcp_tools"}
    if out["mode"] not in allowed_modes:
        if out["code"]:
            out["mode"] = "code"
        elif out["answer"]:
            out["mode"] = "explain"
        elif out["mcp"]:
            out["mode"] = "mcp_tools"
        else:
            out["mode"] = ""

    # mode=code 但無 code → 從 fenced code 補救
    if out["mode"] == "code" and not out["code"]:
        fenced = _extract_first_fenced_code(raw)
        if fenced:
            out["code"] = fenced

    plan_span = None
    if out["mode"] == "code" and not out["plan"]:
        soft = _grab_block_flexible(raw, "PLAN")
        if soft:
            out["plan"] = soft.strip()

    if out["mode"] == "code" and not out["code"]:
        soft_code = _grab_block_flexible(raw, "CODE")
        if soft_code:
            out["code"] = soft_code.strip()

    # PLAN 缺 → 從 raw 嘗試恢復
    if out["mode"] == "code" and not out["plan"]:
        rec = _recover_plan_from_text_or_code(raw)
        if rec:
            plan_obj, span_start, span_end = rec
            out["plan"] = json.dumps(plan_obj, ensure_ascii=False)
            plan_span = (span_start, span_end)

    if out.get("mode") != "code":
        m = re.search(r"\bcode\s*(\{)", raw, re.I)
        idx = m.start(1) if m else raw.find("{")
        if idx != -1 and ("import " in raw or "requests.get" in raw or "pd.read" in raw or "plt." in raw):
            try:
                plan_obj, end_idx = json.JSONDecoder().raw_decode(raw[idx:])
                code_text = raw[idx + end_idx :].strip()
                if isinstance(plan_obj, dict) and plan_obj.get("endpoint"):
                    out["mode"] = "code"
                    out["plan"] = json.dumps(plan_obj, ensure_ascii=False)
                    out["code"] = code_text
                    plan_span = (idx, idx + end_idx)
            except Exception:
                pass

    if plan_span and (not out.get("code")):
        _, span_end = plan_span
        remainder = raw[span_end:].strip()
        if remainder:
            out["code"] = remainder

    if out.get("mode") == "code" and not out.get("code"):
        for marker in ("import ", "from ", "BASE_URL", "endpoint =", "params =", "requests.get"):
            pos = raw.find(marker)
            if pos != -1:
                out["code"] = raw[pos:].strip()
                break

    if out.get("mode") == "code" and out.get("code"):
        if out["code"].startswith("code "):
            brace = out["code"].find("}")
            if brace != -1:
                out["code"] = out["code"][brace + 1 :].strip()
        cleaned = _normalize_inline_code(out["code"])
        cleaned = cleaned.replace("<<<END>>>", "").strip()
        out["code"] = cleaned

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
    mcp: Optional[Dict[str, Any]] = None

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

    mcp = parsed.get("mcp")
    mcp_obj: Optional[Dict[str, Any]] = None
    if isinstance(mcp, str):
        stripped = mcp.strip()
        if stripped:
            try:
                loaded = json.loads(stripped)
                if isinstance(loaded, dict):
                    mcp_obj = loaded
                else:
                    mcp_obj = {"_raw": stripped}
            except Exception:
                mcp_obj = {"_raw": stripped}
    elif isinstance(mcp, dict):
        mcp_obj = mcp
    res.mcp = mcp_obj

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

    collections = collections or get_active_collections()
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

    if _looks_like_table_request(query):
        for coll in collections:
            try:
                from qdrant_client.http import models  # type: ignore
                table_filter = models.Filter(
                    must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value="table"))]
                )
                resp3 = client.query_points(
                    collection_name=coll,
                    query=vector,
                    limit=10,
                    with_payload=True,
                    with_vectors=False,
                    query_filter=table_filter,
                )
            except Exception:
                resp3 = None
            if not resp3:
                continue
            points3 = getattr(resp3, "points", resp3)
            for point in points3 or []:
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

    if all_hits:
        for hit in all_hits:
            payload = getattr(hit, "payload", {}) or {}
            bonus = _score_bonus_from_payload(payload, query)
            if bonus:
                hit.score = float(hit.score or 0.0) + bonus

    if _omnipipe_source_enabled() and _link_expansion_enabled():
        link_start = time.perf_counter()
        expanded = expand_hits_via_links(all_hits, client, limit=max(k, 1))
        link_ms = (time.perf_counter() - link_start) * 1000.0
        if expanded:
            all_hits.extend(expanded)
        logger.debug("[onepass] link_expansion_ms=%.2f expanded=%d", link_ms, len(expanded or []))

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
    endpoint_whitelist = _collect_api_endpoints_from_hits(hits)

    def merge_into(dst, src):
        for name in src.get("params", []) or []: dst["params"].add(str(name))
        for p in src.get("paths", []) or []:     dst["paths"].add(str(p))
        for v in src.get("append_allowed", []) or []: dst["append_allowed"].add(str(v))
        for url in src.get("servers", []) or []: dst["servers"].add(str(url))
        for k, vals in (src.get("param_enums", {}) or {}).items():
            dst["param_enums"].setdefault(k, [])
            for v in vals:
                if v not in dst["param_enums"][k]:
                    dst["param_enums"][k].append(v)

    acc = {"params": set(), "paths": set(), "append_allowed": set(), "servers": set(), "param_enums": {}}
    for raw in texts:
        merge_into(acc, _parse_oas_text(raw))

    if endpoint_whitelist.get("paths"):
        merge_into(acc, endpoint_whitelist)

    if (len(acc["params"]) <= 1 or not acc["paths"]):
        cols = {h.payload.get("collection") for h in hits if h.payload}
        key = _collections_key([c for c in cols if isinstance(c, str) and c] or None)
        cached = _OAS_CACHE.get(key)
        if not cached:
            merged = {"params": set(), "paths": set(), "append_allowed": set(), "servers": set(), "param_enums": {}}
            for raw in _collect_api_specs_full_scan(limit=256, collections=list(key)):
                merge_into(merged, _parse_oas_text(raw))
            if not merged["paths"]:
                endpoint_full = _collect_api_endpoints_full_scan(limit=512, collections=list(key))
                merge_into(merged, endpoint_full)
            cached = {
                "params": sorted(merged["params"]),
                "paths": sorted(merged["paths"]),
                "append_allowed": sorted(merged["append_allowed"]),
                "servers": sorted(merged["servers"]),
                "param_enums": merged["param_enums"],
            }
            _OAS_CACHE[key] = cached
        return cached

    return {
        "params": sorted(acc["params"]),
        "paths": sorted(acc["paths"]),
        "append_allowed": sorted(acc["append_allowed"]),
        "servers": sorted(acc["servers"]),
        "param_enums": acc["param_enums"],
    }

def decide_and_generate(
    *,
    query: str,
    notes: str,
    whitelist: Dict[str, Any],
    llm_id: str,
    prev_plan: Optional[Dict[str, Any]] = None,
    followup_hint: bool = False,
    max_continue: bool = True,
    debug: bool = False,
    today: str,
    ghrsst_hint: bool = False,
    tz: str = "Asia/Taipei",
    query_time: str = "",
    tide_hint: bool = False,
) -> Dict[str, Any]:
    prompt = build_main_prompt(
        query=query,
        notes=notes or "",
        whitelist=whitelist,
        top_k=6,
        prev_plan=prev_plan,
        followup_hint=followup_hint,
        today=today,
        ghrsst_hint=ghrsst_hint,
        tz=tz,
        query_time=query_time,
        tide_hint=tide_hint,
    )

    # 只在這裡印一次 prompt 尺寸（可選）
    if DEBUG_MODE or debug:
        logger.info("[onepass] llm_provider=%s llm_model=%s", LLM.provider, LLM.model)
        logger.info("[onepass] prompt_len=%d", len(prompt))

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user",   "content": "Follow STRICT OUTPUT FORMAT and answer now."},
    ]

    try:
        raw = LLM.chat(messages, temperature=0.2, max_tokens=1024)
    except Exception as exc:
        logger.error("[onepass] LLM chat error: %s", exc)
        if DEBUG_MODE or debug:
            raise
        return {
            "mode": "explain",
            "answer": "LLM 呼叫失敗或逾時，請稍後再試，並可執行 /llm status 確認狀態。",
            "debug": {"error": str(exc), "mode": "explain"},
        }
    raw = raw or ""
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
        allowed = {"code", "explain", "fallback", "mcp_tools"}
        if not m or m not in allowed:
            return True
        only_token = raw.strip().lower() in allowed
        answer_empty = not (p.get("answer") or p.get("text") or "").strip()
        code_empty = not (p.get("code") or "").strip()
        if m == "code" and (code_empty or answer_empty and not p.get("plan")):
            return True
        if m in {"explain", "fallback"} and answer_empty:
            return True
        if m == "mcp_tools" and not (p.get("mcp") or "").strip():
            return True
        return only_token

    if _invalid(parsed):
        fix_user = (
            "Your previous response was INVALID. "
            "Immediately output the full tagged blocks per STRICT OUTPUT FORMAT with no markdown fences. "
            "Begin with '<<<MODE>>>', then (if MODE=code) include non-empty <<<PLAN>>> JSON and <<<CODE>>> python. "
            "Do NOT wrap them in ``` or prepend language labels." 
        )
        raw2 = LLM.chat([{"role":"system","content":prompt},{"role":"user","content":fix_user}], temperature=0.2, max_tokens=1024) or ""
        if DEBUG_MODE or debug:
            logger.info("[onepass] retry_raw_len=%d head=%r", len(raw2), raw2[:64])
        parsed2 = _parse_onepass_output(raw2); parsed2["raw"] = raw2
        if (parsed2.get("code") or parsed2.get("answer")):
            parsed = parsed2

    # 如果 explain 但沒有 answer，就塞 raw（避免空白）
    if (parsed.get("mode") in {"explain", "fallback"}) and not (parsed.get("answer") or "").strip():
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
    # if re.search(r"api[_-]?key|token|authorization|bearer", code, re.I):
    #     raise ValueError("Code must not reference secrets/tokens")

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


def format_citations(hits: List[Hit], answer_text: Optional[str] = None, limit: int = 2) -> List[Citation]:
    if not hits or limit == 0:
        return []
    selected_hits = hits
    if limit:
        selected_hits = selected_hits[:limit]
    if answer_text:
        selected_hits = _select_citation_hits(hits, answer_text, limit or len(hits))

    citations: List[Citation] = []
    seen: set[tuple[str, str, int]] = set()
    for hit in selected_hits:
        payload = hit.payload or {}
        title, url = get_title_url(payload)
        source = url or ""
        key = (title, source, hit.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(title=title, source=source, chunk_id=hit.chunk_id))
    return citations


def _looks_like_followup(query: str, prev_plan: Optional[Dict[str, Any]]) -> bool:
    if not prev_plan or not query:
        return False
    lowered = query.lower()
    if any(keyword in lowered for keyword in FOLLOWUP_KEYWORDS):
        return True
    params = (prev_plan.get("params") or {}) if isinstance(prev_plan, dict) else {}
    changes = sum(1 for key in params if key and str(key) in query)
    has_plot_terms = bool(PLOT_TERM_REGEX.search(lowered))
    return (changes >= 2) or (changes >= 1 and has_plot_terms)


def _looks_like_ghrsst_request(query: str) -> bool:
    if not query:
        return False
    lowered = query.lower()
    if not GHRSST_REGEX.search(lowered):
        return False
    coord_like = bool(COORD_PAIR_RE.search(query) or BBOX_RE.search(query))
    if coord_like:
        return True
    if GHRSST_GEO_REGEX.search(lowered):
        return True
    digits = sum(ch.isdigit() for ch in query)
    if TIME_HINT_REGEX.search(lowered) and digits >= 2:
        return True
    return False


def _looks_like_tide_request(query: str) -> bool:
    if not query:
        return False
    lowered = query.lower()
    return bool(TIDE_REGEX.search(lowered))


def _looks_like_unclear_answer(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    phrases = (
        "請再具體說明",
        "請重新描述",
        "無法理解",
        "不太確定",
        "i cannot understand",
        "please clarify",
        "not sure what you mean",
    )
    return any(p in lowered for p in phrases)

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
    try:
        raw2 = LLM.generate(prompt, temperature=0.2)
    except Exception as exc:
        logger.error("[onepass] continue code failed: %s", exc)
        return prev_code
    code2 = _extract_block(raw2, "CODE") or _extract_first_fenced_code(raw2) or raw2.strip()
    return code2

def run_onepass(
    query: str,
    k: int = 6,
    collections: Optional[List[str]] = None,
    temperature: float = 0.2,
    debug: bool = False,
    today: Optional[str] = None,
    tz: Optional[str] = None,
    query_time: Optional[str] = None,
) -> OnePassResult:
    global LAST_PLAN
    t0 = time.perf_counter()
    debug_enabled = DEBUG or DEBUG_MODE or debug

    if not _query_is_meaningful(query):
        msg = "問題內容過短或不清楚，請再描述一次。"
        fallback = OnePassResult(mode="explain", text=msg, citations=[])
        fallback.debug = {"error": "query_too_short"}
        return fallback

    tz_value = resolve_tz_name(tz)
    query_time_value = query_time or now_iso_in_tz(tz_value)
    today_value = today or today_in_tz(tz_value)

    prev_plan = LAST_PLAN.copy() if isinstance(LAST_PLAN, dict) else None
    followup_hint = _looks_like_followup(query, prev_plan)
    ghrsst_hint = _looks_like_ghrsst_request(query)
    tide_hint = _looks_like_tide_request(query)

    hits_raw = search_qdrant(query, k=max(k, 1) * 3, collections=collections)
    lex_start = time.perf_counter()
    hits_raw = _lexical_filter_hits(query, hits_raw)
    lex_ms = (time.perf_counter() - lex_start) * 1000.0
    if debug_enabled:
        logger.info("[onepass] lexical_filter_ms=%.2f hits=%d", lex_ms, len(hits_raw))
    if debug_enabled:
        active_cols = collections or get_active_collections()
        logger.info("[onepass] active_collections=%s", active_cols)
        preview = []
        for h in hits_raw[:5]:
            preview.append(
                {
                    "doc_type": getattr(h, "doc_type", ""),
                    "source_file": getattr(h, "source_file", ""),
                    "title": getattr(h, "title", ""),
                    "score": float(getattr(h, "score", 0.0) or 0.0),
                }
            )
        logger.info("[onepass] hits_preview=%s", json.dumps(preview, ensure_ascii=False))

    # 去重（檔案層）→ code/api 優先 → 多樣化取前 k
    hits_stage = prefer_code_api_first_hits(dedupe_hits_hits(hits_raw))
    hits       = rerank_diversify_hits(hits_stage, k=k)
    if _looks_like_table_request(query):
        table_candidates = _fetch_table_candidates(query, collections or get_active_collections(), limit=20)
        table_hit = _pick_best_table_hit(query, table_candidates or hits_raw)
        if table_hit:
            hits = [table_hit] + [h for h in hits if h is not table_hit][: max(0, k - 1)]

    t1 = time.perf_counter()

    whitelist = harvest_oas_whitelist(hits)
    t2 = time.perf_counter()

    # NEW: 由檢索結果組 notes
    notes = _compress_notes(hits, max_chars=NOTES_MAX_CHARS, max_docs=NOTES_MAX_DOCS, query=query)
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
        prev_plan=prev_plan,
        followup_hint=followup_hint,
        max_continue=True,
        debug=debug_enabled,
        today=today_value,
        tz=tz_value,
        query_time=query_time_value,
        ghrsst_hint=ghrsst_hint,
        tide_hint=tide_hint,
    )
    if isinstance(result, dict):
        result = _to_result(result)
    t3 = time.perf_counter()

    if getattr(result, "mode", "").lower() != "code":
        result.plan = None

    plan_candidate = getattr(result, "plan", None)
    inherited = False
    if isinstance(plan_candidate, dict) and set(plan_candidate.keys()) == {"_raw"}:
        raw_val = str(plan_candidate.get("_raw") or "").strip()
        if not raw_val:
            plan_candidate = None
    if plan_candidate is None and prev_plan and followup_hint and (result.mode or "").lower() == "code":
        result.plan = prev_plan.copy()
        inherited = True
    elif plan_candidate is None:
        result.plan = None

    if inherited and debug_enabled:
        result.debug = (result.debug or {})
        result.debug["plan_inherited"] = True

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
        result.debug["active_collections"] = collections or get_active_collections()
        result.debug["hits_preview"] = [
            {
                "doc_type": getattr(h, "doc_type", ""),
                "source_file": getattr(h, "source_file", ""),
                "title": getattr(h, "title", ""),
                "score": float(getattr(h, "score", 0.0) or 0.0),
            }
            for h in hits_raw[:5]
        ]

    plan_ok, guard_ok = None, None
    warnings: List[str] = []
    if result.plan and result.code:
        try:
            validate_plan(result.plan, whitelist)
            plan_ok = True
        except Exception as e:
            plan_ok = False
            msg = f"Plan validation warning: {e}"
            warnings.append(msg)
            if debug_enabled:
                LOGGER.info("[onepass] %s", msg)
        try:
            enforce_static_guards(result.code)
            guard_ok = True
        except Exception as e:
            guard_ok = False
            msg = f"Static guard warning: {e}"
            warnings.append(msg)
            if debug_enabled:
                LOGGER.info("[onepass] %s", msg)

        params = result.plan.get("params") if isinstance(result.plan, dict) else {}
        code_text = result.code or ""
        code_lower = code_text.lower()
        if isinstance(params, dict):
            for key in params:
                key_token = f"['{key}']"
                key_token_alt = f'"{key}"'
                if key_token not in code_text and key_token_alt not in code_text:
                    warnings.append(f"Code may not reference plan param '{key}' explicitly; verify request uses it.")

        chunk_rule = str(result.plan.get("chunk_rule") or "").strip().lower()
        if chunk_rule == "yearly" and "year" not in code_lower:
            warnings.append("chunk_rule='yearly' but code lacks a yearly loop.")
        if chunk_rule == "yearly" and "pd.concat" not in code_lower:
            warnings.append("chunk_rule='yearly' but code does not concat chunk results.")
        if chunk_rule == "monthly" and "month" not in code_lower:
            warnings.append("chunk_rule='monthly' but code lacks a monthly loop.")
        if chunk_rule == "monthly" and "pd.concat" not in code_lower:
            warnings.append("chunk_rule='monthly' but code does not concat chunk results.")
        if chunk_rule == "decade" and "decade" not in code_lower and "10" not in code_lower:
            warnings.append("chunk_rule='decade' but code lacks decade segmentation.")

        plot_rule = str(result.plan.get("plot_rule") or "").strip().lower()
        if plot_rule == "timeseries" and "plt.plot" not in code_lower:
            warnings.append("plot_rule='timeseries' but code does not call plt.plot().")
        if plot_rule == "map" and "pcolormesh" not in code_lower:
            warnings.append("plot_rule='map' but code does not use pcolormesh().")
        if plot_rule == "map" and "meshgrid" not in code_lower and "pivot" not in code_lower:
            warnings.append("plot_rule='map' but code does not build a lon/lat grid via pivot/meshgrid.")

    allow_code_request = _wants_code(query) or followup_hint
    auto_reason = None
    if (result.mode or "").lower() == "code":
        # if not allow_code_request:
            # auto_reason = "question did not request code"
        if plan_ok is False or guard_ok is False:
            auto_reason = "plan_or_guard_failed"
    if auto_reason:
        result.mode = "explain"
        result.code = None
        result.plan = None
        if not (result.text and result.text.strip()):
            result.text = "請再具體說明需求。"
        warnings.append(f"Auto downgraded to explain: {auto_reason}")
        dbg = result.debug or {}
        dbg["auto_downgraded"] = auto_reason
        result.debug = dbg
        auto_downgraded = True
    else:
        auto_downgraded = False

    fallback_mode = (result.mode or "").lower() == "fallback"
    if not fallback_mode and _looks_like_unclear_answer(result.text or ""):
        fallback_mode = True
        result.mode = "fallback"
    if fallback_mode:
        result.plan = None
        result.code = None
        if not (result.text and result.text.strip()):
            result.text = "無法辨識需求，請再描述一次。"

    t4 = time.perf_counter()

    parsed_mode = (result.mode or "").strip().lower()
    answer_text = (result.text or "").strip()
    debug_obj = getattr(result, "debug", None) or {}
    suppress_citations = (
        bool(debug_obj.get("error"))
        or auto_downgraded
        or fallback_mode
        or parsed_mode == "mcp_tools"
    )
    if suppress_citations:
        result.citations = []
    else:
        result.citations = format_citations(hits, answer_text=answer_text, limit=2)

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
                "warnings": warnings,
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

    if warnings and result.code:
        warning_block = "\n\n# WARNINGS (needs manual review)\n" + "\n".join(f"# {w}" for w in warnings)
        result.code = f"{result.code}{warning_block}"

    if result.plan and isinstance(result.plan, dict):
        LAST_PLAN = result.plan.copy()

    return result
