from src.askMyDocs.generation.citations import extract_sources

def build_context(documents):

    sources = extract_sources(documents)

    context_parts = []

    for document, source in zip(documents, sources):

        context_parts.append(
            f"[{source['id']}]\n"
            f"Source: {source['source']}\n"
            f"Page: {source['page']}\n"
            f"Content:\n{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    return context, sources

