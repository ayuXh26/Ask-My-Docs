from langchain_community.retrievers import BM25Retriever
from src.askMyDocs.ingestion.chunker import chunk_documents

def create_bm25_retriever():
    chunks = chunk_documents()

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 3

    return retriever


