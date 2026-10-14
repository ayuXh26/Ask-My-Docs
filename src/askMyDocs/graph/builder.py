from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.nodes.rewrite import rewrite_node
from src.askMyDocs.nodes.retrieve import retrieve_node
from src.askMyDocs.nodes.generate import generate_node
from src.askMyDocs.nodes.citation_veriffication import citation_verification_node
from src.askMyDocs.nodes.router import router_node
from src.askMyDocs.nodes.out_of_scope import out_of_scope_node
from src.askMyDocs.nodes.web_search import web_search_node
from src.askMyDocs.nodes.web_generate import web_generate_node
from src.askMyDocs.nodes.direct_response import direct_response_node

def route_after_router(state: graphState):
    return state["route"]


def build_graph():
    graph = StateGraph(graphState)

    # Register nodes
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("citation_verification", citation_verification_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("web_generate", web_generate_node)
    graph.add_node("direct_response", direct_response_node)

    # Main workflow
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "router")

    # Route based on the router's decision
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve": "retrieve",
            "direct": "direct_response",
            "web_search": "web_search",
            "out_of_scope": "out_of_scope",
        },
    )

    # Retrieval and answer generation
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "citation_verification")
    graph.add_edge("citation_verification", END)

    # Routes that return their own messages
    graph.add_edge("direct_response", END)
    graph.add_edge("out_of_scope", END)
    graph.add_edge("web_search", "web_generate")
    graph.add_edge("web_generate", END) 

    memory = MemorySaver()

    return graph.compile(checkpointer=memory)
