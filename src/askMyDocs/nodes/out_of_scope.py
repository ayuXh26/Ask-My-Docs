from src.askMyDocs.state.graph_state import graphState


def out_of_scope_node(state: graphState) -> dict:
    return {
        "answer": (
            "I'm AskMyDocs, an assistant focused on system design "
            "and the topics covered by System Design Interview: "
            "An Insider's Guide by Alex Xu. "
            "Your question appears to be outside my scope. "
            "Please ask me something related to system design."
        ),
        "need_retrieval": False,
    }
