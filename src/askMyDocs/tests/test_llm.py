from src.askMyDocs.LLM.llm import create_llm

llm = create_llm()

response = llm.invoke("What is LangGraph?")

print(response.content)