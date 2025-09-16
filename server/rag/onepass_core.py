from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

try:
    from ollama import Client as OllamaClient
except ImportError:  # pragma: no cover
    OllamaClient = None


LOG_LEVEL = os.environ.get("ONEPASS_LOG_LEVEL", "INFO").upper()
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

_embedder: Optional[SentenceTransformer] = None
_embedding_dim: Optional[int] = None
_qdrant: Optional[QdrantClient] = None
_ollama: Optional[OllamaClient] = None
_system_prompt: Optional[str] = None
_user_prompt_template: Optional[str] = None


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


def _collect_api_specs_from_hits(hits: Sequence[Hit]) -> List[str]:
    texts: List[str] = []
    for hit in hits:
        payload = hit.payload or {}
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


def _extract_block(text: str, tag: str) -> str:
    pat = re.compile(rf"<<<{tag}>>>\s*([\s\S]*?)\s*<<<END>>>", flags=re.IGNORECASE)
    match = pat.search(text)
    return (match.group(1) if match else "").strip()


def _parse_onepass_output(raw: str) -> Dict[str, str]:
    parsed = {
        "mode": _extract_block(raw, "MODE").lower(),
        "plan": _extract_block(raw, "PLAN"),
        "code": _extract_block(raw, "CODE"),
        "answer": _extract_block(raw, "ANSWER"),
    }
    if not parsed["mode"]:
        if parsed["code"]:
            parsed["mode"] = "code"
        elif parsed["answer"]:
            parsed["mode"] = "explain"
    if parsed["mode"] == "code" and not parsed["code"]:
        fence = re.search(r"```[a-zA-Z0-9]*\s*([\s\S]*?)```", raw)
        if fence and fence.group(1).strip():
            parsed["code"] = fence.group(1).strip()
    if parsed["mode"] == "explain" and not parsed["answer"]:
        stripped = re.sub(r"```[\s\S]*?```", " ", raw)
        stripped = re.sub(r"<<<[^>]+>>>", " ", stripped)
        parsed["answer"] = re.sub(r"\s+", " ", stripped).strip()
    return parsed

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
    citations: List[Citation] = None


def search_qdrant(query: str, k: int = 6, collections: Optional[List[str]] = None) -> List[Hit]:
    if not query.strip():
        return []
    collections = collections or DEFAULT_COLLECTIONS
    vector = encode_query(query)
    client = get_qdrant()
    all_hits: List[Hit] = []

    limit = max(20, (k or 6) * 4)
    for coll in collections:
        try:
            resp = client.query_points(
                collection_name=coll,
                query=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Qdrant query failed for %s: %s", coll, exc)
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

    all_hits.sort(key=lambda h: h.score, reverse=True)
    deduped: List[Hit] = []
    seen_keys: set[tuple[Any, Any, Any]] = set()
    for hit in all_hits:
        key = (hit.payload.get("doc_id"), hit.chunk_id, hit.source_file)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(hit)

    top_hits = deduped[: max(k, 1)]
    top_hits = _force_include_api_spec(deduped, top_hits)
    final_hits = top_hits[: max(k, 1)]
    logger.debug("search_qdrant retrieved %d hits (returning %d)", len(deduped), len(final_hits))
    return final_hits

def harvest_oas_whitelist(hits: List[Hit]) -> Dict[str, Any]:
    texts = _collect_api_specs_from_hits(hits)
    params: set[str] = set()
    paths: set[str] = set()
    append_allowed: set[str] = set()
    enums: Dict[str, List[str]] = {}

    def merge(meta: Dict[str, Any]) -> None:
        for name in meta.get("params", []) or []:
            params.add(str(name))
        for path in meta.get("paths", []) or []:
            paths.add(str(path))
        for val in meta.get("append_allowed", []) or []:
            append_allowed.add(str(val))
        for key, values in (meta.get("param_enums", {}) or {}).items():
            merged = enums.setdefault(key, [])
            for value in values:
                if value not in merged:
                    merged.append(value)

    for raw in texts:
        merge(_parse_oas_text(raw))

    if (len(params) <= 1 or not paths):
        collections = {hit.payload.get("collection") for hit in hits if hit.payload}
        col_list = [c for c in collections if isinstance(c, str) and c]
        for raw in _collect_api_specs_full_scan(limit=256, collections=col_list):
            merge(_parse_oas_text(raw))

    whitelist = {
        "params": sorted(params),
        "paths": sorted(paths),
        "append_allowed": sorted(append_allowed),
        "param_enums": enums,
    }
    logger.debug(
        "harvest_oas_whitelist params=%d paths=%d append=%d", len(params), len(paths), len(append_allowed)
    )
    return whitelist

def decide_and_generate(
    query: str,
    notes: str,
    whitelist: Dict[str, Any],
    temperature: float = 0.2,
    top_k: int = 6,
) -> OnePassResult:
    system_prompt = load_system_prompt()
    user_template = load_user_prompt_template()
    whitelist_blob = json.dumps(whitelist or {}, ensure_ascii=False, indent=2)
    user_prompt = user_template.format(
        query=query,
        notes=notes or "(none)",
        whitelist=whitelist_blob,
        top_k=top_k,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    client = get_ollama()
    if client is None:
        raise RuntimeError("Ollama client not available for one-pass generation")

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"temperature": temperature},
    )
    if DEBUG_MODE:
        logger.debug("One-pass prompt:\n%s", user_prompt)
        logger.debug("LLM raw response: %r", response)

    output = ""
    if isinstance(response, dict):
        output = (response.get("message") or {}).get("content", "") or response.get("response", "")
    else:
        output = str(response or "")
    if not output:
        msg = "LLM returned empty response"
        if DEBUG_MODE:
            raise RuntimeError(msg)
        logger.warning("%s; using fallback explain mode", msg)
        return OnePassResult(mode="explain", text="無法取得模型回覆，請確認 LLM 服務是否啟動。", citations=[])

    parsed = _parse_onepass_output(output)
    mode = parsed.get("mode") or "explain"
    plan = None
    if parsed.get("plan"):
        try:
            plan = json.loads(parsed["plan"])
        except Exception:
            plan = None
    result = OnePassResult(mode=mode, citations=[])
    if mode == "code":
        result.code = parsed.get("code") or ""
        result.plan = plan
    else:
        result.text = parsed.get("answer") or parsed.get("text") or parsed.get("code") or output
    return result

def validate_plan(plan: Dict[str, Any], whitelist: Dict[str, Any]) -> None:
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a dict")
    endpoint = plan.get("endpoint")
    params = plan.get("params")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("Plan missing endpoint")
    if not isinstance(params, dict):
        raise ValueError("Plan params must be an object")

    allowed_paths = set(whitelist.get("paths", []) or [])
    if allowed_paths and endpoint not in allowed_paths:
        raise ValueError(f"Endpoint '{endpoint}' not allowed")

    allowed_params = set(whitelist.get("params", []) or [])
    for key in params:
        if allowed_params and key not in allowed_params:
            raise ValueError(f"Parameter '{key}' not allowed")

    enums = whitelist.get("param_enums", {}) or {}
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

def enforce_static_guards(code: str) -> None:
    if not code or not isinstance(code, str):
        raise ValueError("Code block is empty")

    if "requests.get(" in code and "params=" not in code:
        raise ValueError("requests.get must use params=")
    if re.search(r"requests\.get\([^)]*\?[^)]*\)", code):
        raise ValueError("Avoid concatenating query strings in requests.get")
    if re.search(r"pd\.read_json\s*\(", code):
        raise ValueError("Use response.json() instead of pandas.read_json")
    if re.search(r"io\.StringIO\s*\(\s*r\.text\s*\)", code, flags=re.I):
        raise ValueError("Do not wrap response.text with io.StringIO for JSON")
    if re.search(r"pd\.read_csv\s*\(\s*io\.StringIO\s*\(\s*r\.text\s*\)\s*\)", code, flags=re.I):
        raise ValueError("Do not parse JSON payloads with pandas.read_csv")
    if re.search(r"api[_-]?key|token|authorization|bearer", code, re.I):
        raise ValueError("Code must not reference API keys or tokens")
    if "```python" in code and "```" not in code.strip().split("```python", 1)[-1]:
        raise ValueError("Code fence appears truncated")

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

def run_onepass(query: str, k: int = 6, collections: Optional[List[str]] = None, temperature: float = 0.2) -> OnePassResult:
    hits = search_qdrant(query, k=k, collections=collections)
    whitelist = harvest_oas_whitelist(hits)
    notes = collect_rag_notes(hits)
    result = decide_and_generate(query, notes, whitelist, temperature=temperature, top_k=len(hits))
    if result.plan and result.code:
        validate_plan(result.plan, whitelist)
    if result.code:
        enforce_static_guards(result.code)
    result.citations = format_citations(hits)
    return result
