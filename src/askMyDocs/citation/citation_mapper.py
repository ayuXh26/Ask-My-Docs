def map_citations(sources, validation):
    """Return only the sources the answer actually cites, numbered as in the final answer."""
    by_id = {source["id"]: source for source in sources}
    used = [
        {**by_id[old], "id": new}
        for old, new in validation["citation_map"].items()
    ]
    return sorted(used, key=lambda source: source["id"])


def format_sources(used_sources):
    """Plain-text source list for the terminal (swap for UI rendering later)."""
    if not used_sources:
        return ""
    lines = ["Sources:"]
    for source in used_sources:
        page = f", p. {source['page']}" if source["page"] else ""
        lines.append(f"  [{source['id']}] {source['source']}{page}")
    return "\n".join(lines)