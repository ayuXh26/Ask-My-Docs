from src.askMyDocs.retrieval.retriever import create_retriever

def build_context(documents):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return context

