"""RAG layer over course_data.json.

Builds (or loads) a FAISS index of Seoul travel courses, embedded with
Google's text-embedding-004. Each course becomes one Document; the full
course dict is stashed in metadata so retrieval hands the planner
ready-to-use POI sequences.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent
COURSE_DATA_PATH = _BASE_DIR / "course_data.json"
VECTORSTORE_DIR = _BASE_DIR / "vectorstore"

EMBEDDING_MODEL = "models/gemini-embedding-2"

# Gemini free tier allows ~100 embed requests / minute / model. Stay well
# below that so a single chunk never trips the limit even if other calls
# are in flight.
EMBED_CHUNK_SIZE = 50
EMBED_CHUNK_SLEEP_SECONDS = 60
EMBED_MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------

def _course_to_text(course: dict[str, Any]) -> str:
    """Flatten a course into a single searchable string."""
    title = course.get("course_title", "")
    themes = ", ".join(course.get("theme_category", []) or [])
    source = course.get("source", "")

    poi_lines = []
    for poi in course.get("sequence", []) or []:
        name = poi.get("poi_name", "")
        ptype = poi.get("poi_type", "")
        addr = poi.get("address_en") or poi.get("address_ko", "")
        poi_lines.append(f"- ({ptype}) {name} — {addr}")
    pois = "\n".join(poi_lines)

    return (
        f"Title: {title}\n"
        f"Source: {source}\n"
        f"Themes: {themes}\n"
        f"POIs:\n{pois}"
    )


def _course_to_document(course: dict[str, Any]) -> Document:
    sequence = course.get("sequence", []) or []
    try:
        total_min = sum(int(p.get("estimated_stay_time", 0) or 0) for p in sequence)
    except (TypeError, ValueError):
        total_min = 0

    metadata = {
        "course_id": course.get("course_id"),
        "source": course.get("source"),
        "source_url": course.get("source_url"),
        "course_title": course.get("course_title"),
        "themes": course.get("theme_category", []) or [],
        "poi_count": len(sequence),
        "total_estimated_minutes": total_min,
        # Full course payload so the planner can read POIs straight from
        # the retrieved Document without a second lookup.
        "course": course,
    }
    return Document(page_content=_course_to_text(course), metadata=metadata)


def _load_courses(path: Path = COURSE_DATA_PATH) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Vector store build / load
# ---------------------------------------------------------------------------

_vectorstore: FAISS | None = None
_vectorstore_api_key: str | None = None


def _get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)


_RETRY_DELAY_RE = re.compile(r"retry[_ ]delay[^0-9]*(\d+)", re.IGNORECASE)


def _parse_retry_seconds(err: Exception, default: int = EMBED_CHUNK_SLEEP_SECONDS) -> int:
    """Extract the server-suggested retry_delay from a 429 error string."""
    m = _RETRY_DELAY_RE.search(str(err))
    if m:
        try:
            return int(m.group(1)) + 1  # +1s of slack
        except ValueError:
            pass
    return default


def _embed_with_retry(
    embeddings: GoogleGenerativeAIEmbeddings,
    texts: list[str],
) -> list[list[float]]:
    """embed_documents with exponential-ish backoff on 429s."""
    for attempt in range(EMBED_MAX_RETRIES):
        try:
            return embeddings.embed_documents(texts)
        except Exception as e:
            msg = str(e)
            is_quota = "429" in msg or "quota" in msg.lower() or "rate" in msg.lower()
            if not is_quota or attempt == EMBED_MAX_RETRIES - 1:
                raise
            wait = _parse_retry_seconds(e)
            print(
                f"   ⚠️  rate-limited (attempt {attempt + 1}/{EMBED_MAX_RETRIES}), "
                f"sleeping {wait}s and retrying..."
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")  # for type-checkers


def _embed_documents_chunked(
    docs: list[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
    chunk_size: int = EMBED_CHUNK_SIZE,
    chunk_sleep: int = EMBED_CHUNK_SLEEP_SECONDS,
) -> list[tuple[str, list[float]]]:
    """Embed docs in rate-limit-friendly chunks. Returns (text, vector) pairs."""
    texts = [d.page_content for d in docs]
    pairs: list[tuple[str, list[float]]] = []
    total = len(texts)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = texts[start:end]
        print(f"   embedding {start + 1}–{end} of {total}...")
        vectors = _embed_with_retry(embeddings, chunk)
        pairs.extend(zip(chunk, vectors))
        # Sleep only if there's another chunk coming.
        if end < total:
            print(f"   sleeping {chunk_sleep}s to respect rate limit...")
            time.sleep(chunk_sleep)
    return pairs


def build_or_load_vectorstore(
    api_key: str,
    persist_dir: Path = VECTORSTORE_DIR,
    rebuild: bool = False,
) -> FAISS:
    """Return a FAISS index over course_data.json.

    Persists the index to ``persist_dir`` so embedding cost is paid once
    per machine. Pass ``rebuild=True`` to force re-embedding.

    Embedding runs in chunks of ``EMBED_CHUNK_SIZE`` with
    ``EMBED_CHUNK_SLEEP_SECONDS`` between chunks to stay under Gemini's
    free-tier per-minute quota; 429 errors are retried with the server's
    suggested delay.
    """
    global _vectorstore, _vectorstore_api_key

    # Reuse in-process cache if the key hasn't changed.
    if _vectorstore is not None and _vectorstore_api_key == api_key and not rebuild:
        return _vectorstore

    embeddings = _get_embeddings(api_key)
    index_file = persist_dir / "index.faiss"

    if index_file.exists() and not rebuild:
        store = FAISS.load_local(
            str(persist_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        courses = _load_courses()
        docs = [_course_to_document(c) for c in courses]
        metadatas = [d.metadata for d in docs]

        text_embeddings = _embed_documents_chunked(docs, embeddings)
        store = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=embeddings,
            metadatas=metadatas,
        )
        persist_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(persist_dir))

    _vectorstore = store
    _vectorstore_api_key = api_key
    return store


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def build_query(
    purpose: str | None,
    dietary: str | None,
    location: str | None,
    duration: str | None = None,
) -> str:
    """Compose a natural-language query from the user's confirmed fields."""
    parts = []
    if purpose:
        parts.append(f"Travel purpose: {purpose}.")
    if location:
        parts.append(f"Area or neighborhood of interest: {location}.")
    if dietary and dietary.lower() not in {"none", "no", "n/a", "없음"}:
        parts.append(f"Dietary preference: {dietary}.")
    if duration:
        parts.append(f"Trip length: {duration}.")
    if not parts:
        return "Seoul travel itinerary"
    return " ".join(parts)


def retrieve_courses(
    api_key: str,
    query: str,
    k: int = 10,
) -> list[dict[str, Any]]:
    """Return the top-k course dicts (full payloads) for a query."""
    store = build_or_load_vectorstore(api_key)
    docs = store.similarity_search(query, k=k)
    return [d.metadata.get("course", {}) for d in docs if d.metadata.get("course")]
