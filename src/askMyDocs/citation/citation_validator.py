import re

# [1]   [1, 2]   [1,2,3]  (optional leading spaces are swallowed when a marker is removed)
_MARKER = re.compile(r"([ \t]*)\[(\d+(?:\s*,\s*\d+)*)\]")
# 【1】  【1†L1-L5】  【1, 2】 style markers some models emit -> normalised to [1]
_FANCY = re.compile(r"【(\d+(?:\s*,\s*\d+)*)(?:†[^】]*)?】")
# "[1] ." -> "[1]."
_PUNCT = re.compile(r"(\])[ \t]+([.,;:!?])")
# [1][1] -> [1]
_DUPLICATE = re.compile(r"(\[\d+\])(?:\1)+")
# Never touch ``` code blocks: arr[1] is not a citation
_CODE_FENCE = re.compile(r"(```.*?```)", re.DOTALL)


def normalize_markers(text):
    return _FANCY.sub(r"[\1]", text)


def validate_citations(answer, sources):
    """
    Validate the model's [n] markers against the sources it was given.

    - markers pointing to non-existent sources are removed
    - remaining markers are renumbered 1..N by first appearance in the answer
    - returns the cleaned answer plus bookkeeping; `citation_map` is {old_id: new_id}
    """
    valid_ids = {source["id"] for source in sources}
    id_map = {}
    invalid = []

    def _replace(match):
        lead, raw_ids = match.group(1), match.group(2)
        kept = []
        for raw in raw_ids.split(","):
            old = int(raw)
            if old not in valid_ids:
                if old not in invalid:
                    invalid.append(old)
                continue
            new = id_map.setdefault(old, len(id_map) + 1)
            if new not in kept:
                kept.append(new)
        if not kept:
            return ""
        return lead + "".join(f"[{n}]" for n in sorted(kept))

    parts = _CODE_FENCE.split(normalize_markers(answer))
    cleaned = "".join(
        part if part.startswith("```") else _PUNCT.sub(r"\1\2", _MARKER.sub(_replace, part))
        for part in parts
    )
    cleaned = _DUPLICATE.sub(r"\1", cleaned).strip()

    return {
        "answer": cleaned,
        "used_citations": list(id_map),         # original ids, in order of appearance
        "citation_map": id_map,                 # original id -> final id
        "invalid_citations": sorted(invalid),
        "has_citations": bool(id_map),
        "is_valid": not invalid,
    }


def strip_citations(text):
    """Remove all [n] markers, e.g. before saving an answer into chat history."""
    parts = _CODE_FENCE.split(normalize_markers(text))
    return "".join(
        part if part.startswith("```") else _MARKER.sub("", part) for part in parts
    ).strip()