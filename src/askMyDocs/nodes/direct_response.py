from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.LLM.llm import create_llm
from src.askMyDocs.prompts.prompt import direct_prompt
llm = create_llm()

direct_chain = direct_prompt | llm

def direct_response_node(state: graphState) -> dict:

    question = state["standalone_question"].strip().lower()
    response = direct_chain.invoke({"question": question})

    return {
        "answer": response.content,
        "need_retrieval": False,
    }