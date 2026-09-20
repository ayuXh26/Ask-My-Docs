import re

def validate_citations(answer, citations):
    citation_ids = {citation["id"] for citation in citations}

    used_citations = {
        int(match)
        for match in re.findall(r"\[(\d+)\]", answer)
    }

    invalid_citations = used_citations - citation_ids

    return {
        "used_citations": sorted(used_citations),
        "invalid_citations": sorted(invalid_citations),
        "is_valid": len(invalid_citations) == 0
    }