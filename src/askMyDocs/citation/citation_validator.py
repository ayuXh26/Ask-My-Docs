import re


def validate_citations(answer, citations):
    citation_ids = {
        citation["id"]
        for citation in citations
    }

    # Supports both:
    # [1]
    # 【1†L1-L5】
    citation_matches = re.findall(
        r"(?:\[(\d+)\]|【(\d+)†L\d+-L\d+】)",
        answer
    )

    used_citations = {
        int(match)
        for group in citation_matches
        for match in group
        if match
    }

    invalid_citations = used_citations - citation_ids

    return {
        "used_citations": sorted(used_citations),
        "invalid_citations": sorted(invalid_citations),
        "is_valid": (
            len(used_citations) > 0
            and len(invalid_citations) == 0
        )
    }