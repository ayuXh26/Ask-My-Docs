import os
from src.askMyDocs.ingestion.chunker import chunk_documents
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding_model = OpenAIEmbeddings(model = "text-embedding-3-large")

def embedding_docs():
    chunks = chunk_documents()

    
    embeddings = embedding_model.embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    return chunks, embeddings #Need both for Pinecone

'''
if __name__ == "__main__":
    chunks, embeddings = embedding_docs()

    print("Number of chunks:", len(chunks))
    print("Number of embeddings:", len(embeddings))
    print("Embedding dimension:", len(embeddings[0]))
    print("First chunk metadata:", chunks[0].metadata)
'''
