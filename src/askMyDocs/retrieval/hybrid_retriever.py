from langchain_core.runnables import RunnableLambda

from src.askMyDocs.retrieval.retriever import create_retriever
from src.askMyDocs.retrieval.bm25_retriever import create_bm25_retriever
from src.askMyDocs.rrf.rrf import rrf
from src.askMyDocs.reranker.reranker import create_reranker


def create_hybrid_retriever():

    vector_retriever = create_retriever()
    bm25_retriever = create_bm25_retriever()
    reranker = create_reranker()

    def retrieve_and_rerank(query):

        # Step 1: Retrieve from both systems
        vector_docs = vector_retriever.invoke(query)
        bm25_docs = bm25_retriever.invoke(query)

        # Step 2: Combine using RRF
        hybrid_docs = rrf(
            vector_docs,
            bm25_docs
        )

        # Step 3: Rerank the hybrid candidates
        reranked_docs = reranker.compress_documents(
            hybrid_docs,
            query
        )

        return reranked_docs

    return RunnableLambda(retrieve_and_rerank)