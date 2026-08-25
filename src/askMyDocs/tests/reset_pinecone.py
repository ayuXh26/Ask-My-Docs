from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "ask-my-docs"

if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    print(f"Deleted index: {index_name}")
else:
    print(f"Index '{index_name}' does not exist")