from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class graphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    standalone_question : str
    retrieved_chunks: list[Document]
    answer: str
    need_retrieval: bool
    



