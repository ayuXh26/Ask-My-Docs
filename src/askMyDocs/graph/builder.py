from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.askMyDocs.nodes.generate import generate_node
from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.nodes.rewrite import rewrite_node
from src.askMyDocs.nodes.retrieve import retrieve_node
from src.askMyDocs.nodes.generate import generate_node

def route_after_rewrite(state: graphState):

    if state["need_retrieval"]:
        return "retrieve"

    return "generate"

def build_graph():

    graph = StateGraph(graphState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "rewrite")
    graph.add_conditional_edges("rewrite", route_after_rewrite)
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    memory = MemorySaver()

    return graph.compile(checkpointer=memory)