from src.askMyDocs.generation import citations
from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.retrieval.retriever import create_retriever
from src.askMyDocs.generation.citations import extract_sources

retriever = create_retriever()
def retrieve_node(state: graphState):
    question = state["standalone_question"]
    chunks = retriever.invoke(question)
    previous_chunks = state.get("conversation_chunks", [])
    conversation_chunks = previous_chunks + chunks
    citations = extract_sources(chunks)
    return {
        "retrieved_chunks": chunks,
        "conversation_chunks": conversation_chunks,
        "citations": citations
    }