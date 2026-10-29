from langchain_groq import ChatGroq
from src.askMyDocs.tools.tools import search_documents, search_web
from src.askMyDocs.state.graph_state import graphState
from src.askMyDocs.prompts.prompt import react_agent_prompt
from dotenv import load_dotenv
from langchain_core.prompts import (
   ChatPromptTemplate,
   MessagesPlaceholder
)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

tools = [search_documents, search_web]

llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", react_agent_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

agent_chain = prompt | llm_with_tools


def agent_node(state: graphState):
    """Run the agent and return its response."""

    response = agent_chain.invoke(
        {
            "messages": state["messages"],
        }
    )

    updates = {"messages": [response]}

    if not response.tool_calls and response.content:
        updates["answer"] = response.content

    return updates
