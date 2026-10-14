import os
from dotenv import load_dotenv
from tavily import TavilyClient
import tavily
from src.askMyDocs.state.graph_state import graphState
load_dotenv()


def web_search_node(state: graphState):
    question = (state["standalone_question"]).strip()

    api_key = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=api_key)
    
    response = client.search(
        query=question,
        search_depth="basic",
        topic="general",
        max_results=5,
        include_answer=False,
    )

    results = []

    for item in response.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": item.get("score", 0.0),
        })

    return {
        "web_search_results": results,
    }


