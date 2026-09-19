from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document

from src.askMyDocs.generation import citations


class graphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    standalone_question: str
    retrieved_chunks: list[Document]
    conversation_chunks: list[Document]
    citations: list
    citation_validation: dict
    used_sources: list
    answer: str
    need_retrieval: bool
    



