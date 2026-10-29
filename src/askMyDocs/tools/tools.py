from typing import Annotated
from dotenv import load_dotenv

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.retrieval.hybrid_retriever import create_hybrid_retriever
from src.askMyDocs.citation.citations import dedupe_chunks
from langchain_tavily import TavilySearch
from src.askMyDocs.nodes.web_search import web_search_node

load_dotenv()

retriever = create_hybrid_retriever()



@tool
def search_documents(
    query: str,
    state: Annotated[graphState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """ 
        Search the knowledge base for evidence relevant to a question.
        Use this tool to retrieve relevant passages from the documents
        indexed in the application's vector database. You may call it
        again with a refined query if the initial results are insufficient.
    """

    chunks = retriever.invoke(query)

    previous_chunks = state.get("conversation_chunks", [])

    combined_chunks = dedupe_chunks(previous_chunks + chunks)

    if chunks:
        observations = []

        for index, chunk in enumerate(chunks, start=1):
            text = (
                f"Chunk {index}\n"
                f"Content: {chunk.page_content}\n"
                f"Metadata: {chunk.metadata}"
            )

            observations.append(text)

        observation = "\n\n".join(observations)
    else:
        observation = "No relevant document chunks were found."

    return Command(
        update={
            "retrieved_chunks": chunks,
            "conversation_chunks": combined_chunks,
            "messages": [
                ToolMessage(
                    content=observation,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def search_web(query: str) -> str:
    """Search the public web only as a fallback for book-related questions.

    Use this tool only when:

    1. The user asks about new developments, recent updates, or
       information that may have changed since the book was published.
    2. The indexed book knowledge base does not contain enough
       relevant information to answer the user's question successfully.

    First, use search_document to check the indexed book knowledge
    base for relevant evidence.

    If the book provides sufficient evidence for the question,
    do not use this tool.

    When the book provides partial evidence, use this tool only
    to find the missing information needed to answer the question.

    Keep the web search focused on the original book-related topic
    and the specific information that is missing or outdated.

    Do not use this tool for unrelated topics or as the default
    source of information.
    """

    result = web_search_node({
        "standalone_question": query,
    })

    web_results = result.get("web_search_results", [])

    if not web_results:
        return "No relevant web search results were found."

    return "\n\n".join(
        (
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content: {item.get('content', '')}\n"
            f"Score: {item.get('score', 0.0)}"
        )
        for item in web_results
    )