from src.askMyDocs.ingestion.loader import load_documents
from langchain_classic import text_splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents():
    files = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 75
    )   

    chunks = text_splitter.split_documents(files)

    return chunks


chunks = chunk_documents()

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:10]):
    print(f"\n--- Chunk {i} ---")
    print("Metadata:", chunk.metadata)
    print("Text:", chunk.page_content[:150])