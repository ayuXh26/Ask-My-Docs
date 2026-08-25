from src.askMyDocs.retrieval.retriever import create_retriever


retriever = create_retriever()

query = "What is a Bloom filter and why is it useful in a URL shortener?"

results = retriever.invoke(query)

for document in results:
    print("\nCONTENT:")
    print(document.page_content[:200])

    print("\nMETADATA:")
    print(document.metadata)