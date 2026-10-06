from langchain_core.messages import AIMessage
from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.citation.citation_validator import validate_citations, strip_citations
from src.askMyDocs.citation.citation_mapper import map_citations

def citation_verification_node(state: graphState):
    raw_answer = state["answer"]
    sources = state["citations"]

    validation = validate_citations(raw_answer, sources)
    used_sources = map_citations(sources, validation)

    return {
        # cleaned + renumbered: markers match `used_sources`
        "answer": validation["answer"], #validation is a dictionary and answer is stored inside that
        # history gets the answer WITHOUT [n] markers (numbering restarts every turn)
        "messages": [AIMessage(content=strip_citations(validation["answer"]))],
        "citation_validation": {key: value for key, value in validation.items() if key != "answer"},
        # only what the answer cites (numbering = final answer)
        "used_sources": used_sources,
    }