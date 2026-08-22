from langchain_core.runnables import RunnablePassthrough
from src.askMyDocs.generation.context_builder import build_context
from src.askMyDocs.retrieval.retriever import create_retriever
from src.askMyDocs.prompts.prompt import rag_prompt
from src.askMyDocs.LLM.llm import create_llm

def create_rag_chain():

    retriever = create_retriever()
    prompt = rag_prompt
    llm = create_llm()

    chain = (
        {
            "context" : retriever | build_context,
            "question" : RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return chain