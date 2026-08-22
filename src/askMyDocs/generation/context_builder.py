def build_context(documents):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return context

