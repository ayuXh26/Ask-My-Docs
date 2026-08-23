import os


def extract_sources(documents):

    sources = []

    for document in documents:
        source = document.metadata.get("source")
        page = document.metadata.get("page_label")

        sources.append({
            "source": os.path.basename(source) if source else "Unknown",
            "page": page
        })

    return sources