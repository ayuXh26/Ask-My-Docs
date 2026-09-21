from src.askMyDocs.ingestion.loader import load_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents():
    files = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 75
    )   

    chunks = text_splitter.split_documents(files)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i


    return chunks