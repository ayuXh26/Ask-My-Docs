"""
export_chunks.py

Exports every vector's id + metadata (page, text) from your Pinecone index
into a single JSON file, in the shape resolve_chunks.py expects:

    [
      {"chunk_id": "chunk - 0", "page": 1, "text": "..."},
      {"chunk_id": "chunk - 1", "page": 2, "text": "..."},
      ...
    ]

Usage
-----
1. Install the client if you don't already have it:
     pip install pinecone --break-system-packages

2. Set your API key as an environment variable (don't hardcode it):
     export PINECONE_API_KEY="your-key-here"

3. Fill in INDEX_NAME (and NAMESPACE if you used one — leave "" for default).

4. Run it:
     python export_chunks.py

This writes indexed_chunks.json in the current directory. Upload that file
back to Claude (or run resolve_chunks.py yourself) to get precise
chunk-level expected_chunks in eval_dataset.json.
"""

import json
import os
from dotenv import load_dotenv
from pinecone import Pinecone
load_dotenv()
# ---- fill these in ----
INDEX_NAME = "ask-my-docs"   # e.g. "ask-my-docs"
NAMESPACE = "__default__"                    # "" if you didn't use namespaces
OUTPUT_PATH = "indexed_chunks.json"
# ------------------------

API_KEY = os.environ.get("PINECONE_API_KEY")
if not API_KEY:
    raise SystemExit(
        "PINECONE_API_KEY is not set. Run:\n"
        '  export PINECONE_API_KEY="your-key-here"\n'
        "then re-run this script."
    )

pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

print(f"Connected to index '{INDEX_NAME}'. Listing vector ids...")

all_ids = []
# index.list() paginates internally and yields batches of ids
for id_batch in index.list(namespace=NAMESPACE or None):
    all_ids.extend(id_batch)

print(f"Found {len(all_ids)} vector ids. Fetching metadata in batches...")

chunks = []
BATCH_SIZE = 100  # Pinecone's fetch() accepts up to 100 ids per call

for i in range(0, len(all_ids), BATCH_SIZE):
    batch_ids = all_ids[i : i + BATCH_SIZE]
    result = index.fetch(ids=batch_ids, namespace=NAMESPACE or None)

    # result.vectors is a dict keyed by id -> Vector(id=..., metadata=..., values=...)
    for vec_id, vec in result.vectors.items():
        meta = vec.metadata or {}
        chunks.append(
            {
                "chunk_id": vec_id,
                "page": meta.get("page"),
                "text": meta.get("text"),
            }
        )

    print(f"  fetched {min(i + BATCH_SIZE, len(all_ids))}/{len(all_ids)}")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"\nDone. Wrote {len(chunks)} chunks to {OUTPUT_PATH}")
