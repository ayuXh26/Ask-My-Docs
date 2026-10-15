"""Source extraction: turn retrieved chunks into numbered, de-duplicated sources.

Two rules drive everything else in the citation package:
  1. The same chunk text is only ever kept once (conversation_chunks grows every turn).
  2. Chunks from the same (file, page) share ONE citation number, so the user sees
     one "p. 132" source instead of three.
"""
import hashlib
import os
from typing import TypedDict


class Source(TypedDict):
    id: int
    source: str
    page: str | None
    snippet: str


def dedupe_chunks(documents):
    """Drop chunks whose text we've already seen, keeping first-seen order."""
    seen = set()
    unique = []
    for doc in documents:
        key = hashlib.sha1(doc.page_content.strip().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _source_name(metadata):
    source = metadata.get("source")
    return os.path.basename(source) if source else "Unknown"


def _page_label(metadata):
    label = metadata.get("page_label")
    if label not in (None, ""):
        return str(label)
    page = metadata.get("page")  # 0-indexed in Pinecone
    return str(int(page) + 1) if page is not None else None


def _snippet(text, limit=200):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def group_chunks(documents):
    """Return [(Source, [chunks on that page])], numbered 1..N by first appearance."""
    groups = {}
    for doc in dedupe_chunks(documents):
        metadata = doc.metadata or {}
        key = (_source_name(metadata), _page_label(metadata))
        groups.setdefault(key, []).append(doc)

    result = []
    for i, ((name, page), docs) in enumerate(groups.items(), start=1):
        source: Source = {
            "id": i,
            "source": name,
            "page": page,
            "snippet": _snippet(docs[0].page_content),
        }
        result.append((source, docs))
    return result


def extract_sources(documents):
    return [source for source, _ in group_chunks(documents)]

