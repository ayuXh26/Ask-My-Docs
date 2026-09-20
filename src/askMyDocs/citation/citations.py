import os

def extract_sources(documents):

    sources = []

    for i, document in enumerate(documents, start=1):
        source = document.metadata.get("source")
        page = document.metadata.get("page_label")

        sources.append({
            "id": i,
            "source": os.path.basename(source) if source else "Unknown",
            "page": page
        })

    return sources