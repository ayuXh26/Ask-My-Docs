from src.askMyDocs.LLM.llm import create_llm
from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.prompts.prompt import web_prompt
from src.askMyDocs.citation.citation_validator import validate_citations
from src.askMyDocs.citation.citation_mapper import map_citations

llm = create_llm()

web_chain = web_prompt | llm


def web_generate_node(state: graphState) -> dict:
    question = (state["standalone_question"]).strip()
    results = state.get("web_search_results", [])
 
    if not results:
        return {
            "answer": "I couldn't find relevant web results to answer this question.",
            "web_citations": [],
        }
 
    # Number every result and remember its URL
    all_citations = [
        {"id": i, "title": item.get("title", ""), "url": item.get("url", "")}
        for i, item in enumerate(results, start=1)
    ]
 
    context = "\n\n".join(
        f"[{i}]\n"
        f"Title: {item.get('title', '')}\n"
        f"URL: {item.get('url', '')}\n"
        f"Content: {item.get('content', '')}"
        for i, item in enumerate(results, start=1)
    )
 
    response = web_chain.invoke({
        "question": question,
        "context": context,
    })
 
    # Keep only the results the answer actually cited
    validation = validate_citations(response.content, all_citations)
    used_ids = {int(i) for i in validation["used_citations"]}
    web_citations = [c for c in all_citations if c["id"] in used_ids]
 
    return {
        "answer": response.content,
        "web_citations": web_citations,
    }
 
