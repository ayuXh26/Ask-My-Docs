import logging
from typing import Literal

from pydantic import BaseModel, Field

from src.askMyDocs.prompts.prompt import router_prompt
from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.LLM.llm import create_llm

logger = logging.getLogger(__name__)

llm = create_llm()  # make sure this is configured with temperature=0


class routerDecision(BaseModel):
    reason: str = Field(
        description="One sentence naming the rule that applies and why."
    )
    route: Literal["direct", "retrieve", "web_search", "out_of_scope"] = Field(
        description="The component that should handle the question."
    )


router_chain = router_prompt | llm.with_structured_output(routerDecision)

DEFAULT_ROUTE = "retrieve"  


def router_node(state: graphState) -> dict:
    question = state["standalone_question"].strip()

    if not question:
        return {"route": "direct", "need_retrieval": False}

    try:
        decision = router_chain.invoke({"standalone_question": question})
        route = decision.route
        logger.info("route=%s | reason=%s | q=%r", route, decision.reason, question)
    except Exception:
        
        logger.exception("Router failed, falling back to %s", DEFAULT_ROUTE)
        route = DEFAULT_ROUTE

    return {
        "route": route,
        "need_retrieval": route == "retrieve",
    }