from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.LLM.llm import create_llm
from src.askMyDocs.prompts.prompt import generate_prompt

llm = create_llm()
generate_chain = generate_prompt | llm


def generate_node(state: graphState):

    question = state["standalone_question"]
    chunks = state["retrieved_chunks"]
    messages = state["messages"]

    context = "\n\n".join(
        chunk.page_content
        for chunk in chunks
    )

    response = generate_chain.invoke({
        "context": context,
        "question": question,
        "messages": messages
    })

    return {
        "answer": response.content
    }