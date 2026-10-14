from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from src.askMyDocs.citation import citations
from typing import Annotated, TypedDict, Literal, Any

class graphState(TypedDict):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    standalone_question: str
    route: Literal["direct", "retrieve", "web_search", "out_of_scope"]
    retrieved_chunks: list[Document]
    conversation_chunks: list[Document]
    citations: list
    citation_validation: dict
    used_sources: list
    answer: str
    need_retrieval: bool
    web_search_results: list[dict[str, Any]]
    web_citations: list[dict[str, str]]


