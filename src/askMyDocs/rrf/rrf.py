def rrf(vector_docs, bm25_docs, k=60):
    scores = {}
    document = {}

    for rank, doc in enumerate(vector_docs, start = 1):
        chunk_id = doc.metadata["chunk_id"]
        document[chunk_id] = doc
        scores[chunk_id] = scores.get(chunk_id, 0) + ( 1 / (k + rank))

    for rank, doc in enumerate(bm25_docs, start = 1):
            chunk_id = doc.metadata["chunk_id"]
            document[chunk_id] = doc
            scores[chunk_id] = scores.get(chunk_id, 0) + ( 1 / (k + rank))

    # Sorting chunks by RRF Score
    ranked_ids = sorted(scores, key = scores.get, reverse=True)

    return [document[chunk_id] for chunk_id in ranked_ids]
    

