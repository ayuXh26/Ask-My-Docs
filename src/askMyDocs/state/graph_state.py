
from typing import TypedDict, Annotated, Literal, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class graphState(TypedDict, total=False):
    # Question and conversation
    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    standalone_question: str

    # Routing
    route: Literal[
        "direct",
        "retrieve",
        "web_search",
        "out_of_scope",
        "agent",
    ]

    # Retrieval
    retrieved_chunks: list[Document]
    conversation_chunks: list[Document]

    # Citations
    citations: list
    citation_validation: dict
    used_sources: list

    # Answer generation
    answer: str
    draft_answer: str

    # Retrieval decisions
    need_retrieval: bool

    # Web search
    web_search_results: list[dict[str, Any]]
    web_citations: list[dict[str, str]]

    # Answer evaluation and retries (not used yet)
    retry_count: int
    retry_feedback: str
    evaluation_passed: bool
    evaluation_reason: str
    missing_information: str

    next_action: Literal[
        "finish",
        "retry_documents",
        "search_web",
        "retry_combined",
    ]

    # Optional evaluator diagnostics
    evaluation_scores: dict[str, Any]
