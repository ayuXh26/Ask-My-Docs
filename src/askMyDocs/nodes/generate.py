from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.LLM.llm import create_llm
from src.askMyDocs.prompts.prompt import generate_prompt
from src.askMyDocs.generation.context_builder import build_context
from src.askMyDocs.generation.citation_validator import validate_citations
from src.askMyDocs.generation.citation_mapper import map_citations

llm = create_llm()
generate_chain = generate_prompt | llm


def generate_node(state: graphState):

    question = state["standalone_question"]
    chunks = state["conversation_chunks"]
    messages = state["messages"]

    context, citations = build_context(chunks)

    response = generate_chain.invoke({
        "context": context,
        "question": question,
        "messages": messages
    })

    citation_validation = validate_citations(
        response.content,
        citations
    )

    used_sources = map_citations(
        citations,
        citation_validation["used_citations"]
    )
    
    return {
        "answer": response.content,
        "citations": citations,
        "citation_validation": citation_validation,
        "used_sources": used_sources
    }