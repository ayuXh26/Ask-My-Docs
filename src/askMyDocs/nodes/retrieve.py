from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.retrieval.hybrid_retriever import create_hybrid_retriever
from src.askMyDocs.citation.citations import dedupe_chunks

retriever = create_hybrid_retriever()


def retrieve_node(state: graphState):
    question = state["standalone_question"]
    chunks = retriever.invoke(question)
    previous_chunks = state.get("conversation_chunks", [])

    return {
        "retrieved_chunks": chunks,
        # de-duplicated so repeat retrievals don't bloat the context
        "conversation_chunks": dedupe_chunks(previous_chunks + chunks),
    }