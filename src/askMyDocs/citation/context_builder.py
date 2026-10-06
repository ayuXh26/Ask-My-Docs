from src.askMyDocs.citation.citations import group_chunks


def build_context(documents):
    """Return (context_string_for_prompt, sources). Numbers in the context match sources."""
    context_parts = []
    sources = []

    for source, docs in group_chunks(documents):
        sources.append(source)
        content = "\n\n".join(doc.page_content.strip() for doc in docs)
        context_parts.append(
            f"[{source['id']}]\n"
            f"Source: {source['source']}\n"
            f"Page: {source['page']}\n"
            f"Content:\n{content}"
        )

    return "\n\n".join(context_parts), sources