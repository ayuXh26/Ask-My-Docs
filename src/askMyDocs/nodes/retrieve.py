from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.retrieval.retriever import create_retriever

retriever = create_retriever()
def retrieve_node(state: graphState):
    question = state["standalone_question"]
    chunks = retriever.invoke(question)
    return {"retrieved_chunks": chunks}