import os
import time
from dotenv import load_dotenv
from litellm import embedding
from openai import embeddings
from src.askMyDocs.ingestion.embeddings import embedding_docs, embedding_model
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "ask-my-docs"
dimension  = 3072
metric     = "cosine"
cloud      = "aws"
region     = "us-east-1"


def create_vector_store():
    try:
        pc.create_index(
            name = index_name,
            dimension = dimension,
            metric = metric,
            spec = ServerlessSpec(cloud=cloud, region=region)
        )

        while True:
            status = pc.describe_index(index_name).status
            if status.get("ready"):
                print(f"Index: {index_name} created successfully and is ready to use")
                break
            else:
                print("Waiting for index to become ready")
                time.sleep(2)

    except Exception as e:
        print(f"Failed to create index: '{index_name}' : {e}")


    index = pc.Index(index_name)


    vectors = []
    chunks, embeddings = embedding_docs()

    #Creating the records
    for i, (chunk,embedding) in enumerate(zip(chunks, embeddings)):   # -> zip(chunks, embeddings) => pairs chunk <-> embedding
        vectors.append({
            "id" : f"chunk - {i}",
            "values": embedding,
            "metadata": {
                **chunk.metadata, # -> unpacking the dictionary and putting everything inside the metadata
                "text": chunk.page_content
            }
        })

    batch_size = 100

    for i in range(0, len(vectors), batch_size):

        batch = vectors[i:i + batch_size]

        index.upsert(vectors=batch)

        print(f"Uploaded {i + len(batch)} / {len(vectors)} vectors")


    print(index.describe_index_stats())

def load_vector_store():
    try:
        index = pc.Index(index_name)

        vector_store = PineconeVectorStore(
            index = index,
            embedding = embedding_model
        )

        print(f"Index: {index_name} loaded successfully")

        return vector_store
    except Exception as e:
        print(f"Failed to load index: '{index_name}' : {e}")
        
        