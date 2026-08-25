from src.askMyDocs.ingestion.pinecone_vector_store import load_vector_store

def create_retriever():

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {
            "k" : 3
        }
    )

    return retriever