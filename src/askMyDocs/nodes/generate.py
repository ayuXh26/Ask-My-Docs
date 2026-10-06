from langchain_core.messages import AIMessage
from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.LLM.llm import create_llm
from src.askMyDocs.prompts.prompt import generate_prompt
from src.askMyDocs.citation.context_builder import build_context
from src.askMyDocs.citation.citation_validator import validate_citations, strip_citations
from src.askMyDocs.citation.citation_mapper import map_citations
 
llm = create_llm()
generate_chain = generate_prompt | llm
 
 
def generate_node(state: graphState):
 
    question = state["standalone_question"]
    chunks = state["conversation_chunks"]
    messages = state["messages"]
 
    context, sources = build_context(chunks)
 
    response = generate_chain.invoke({
        "context": context,
        "question": question,
        "messages": messages
    })
 
    return {
        "answer": response.content,
        "citations": sources,
    }
 
