from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.LLM.llm import create_llm
from src.askMyDocs.prompts.prompt import rewrite_prompt
from pydantic import BaseModel, Field

llm = create_llm()

class RewriteResult(BaseModel):
    standalone_question: str = Field(
        description="The latest user question rewritten as a standalone question."
    )

    need_retrieval: bool = Field(
        description="Whether document retrieval is required to answer the question."
    )


rewrite_chain = rewrite_prompt | llm.with_structured_output(
    RewriteResult,
    method="json_schema"
)

def rewrite_node(state: graphState):

    messages = state["messages"]
    question = messages[-1].content
    history = messages[:-1]
    response = rewrite_chain.invoke({
        "history": history,
        "question": question
    })

    return {
        "standalone_question": response.standalone_question,
        "need_retrieval": response.need_retrieval
    }   
 