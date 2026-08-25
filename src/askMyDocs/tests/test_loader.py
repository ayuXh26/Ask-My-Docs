from src.askMyDocs.ingestion.loader import load_documents

documents = load_documents()

print("Total pages loaded:", len(documents))

for document in documents[:3]:
    print("\nCONTENT:")
    print(document.page_content[:200])

    print("\nMETADATA:")
    print(document.metadata)