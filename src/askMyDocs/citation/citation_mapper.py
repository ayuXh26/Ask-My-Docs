def map_citations(citations, used_citations):
    citation_map = {
        citation["id"]: citation
        for citation in citations
    }

    used_sources = []

    for citation_id in used_citations:
        if citation_id in citation_map:
            used_sources.append(citation_map[citation_id])

    return used_sources