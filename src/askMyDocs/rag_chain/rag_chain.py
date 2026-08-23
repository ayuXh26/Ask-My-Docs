from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough

from src.askMyDocs.generation.context_builder import build_context
from src.askMyDocs.retrieval.retriever import create_retriever
from src.askMyDocs.prompts.prompt import rag_prompt
from src.askMyDocs.LLM.llm import create_llm
from src.askMyDocs.generation.citations import extract_sources


def create_rag_chain():

    retriever = create_retriever()
    prompt = rag_prompt
    llm = create_llm()

    retrieval_chain = (
        {
            "question": RunnablePassthrough(),
            "documents": retriever
        }
        | RunnablePassthrough.assign(
            context=lambda x: build_context(x["documents"])
        )
    )

    answer_chain = (
        {
            "context": itemgetter("context"),
            "question": itemgetter("question")
        }
        | prompt
        | llm
    )

    chain = (
    retrieval_chain
    | RunnablePassthrough.assign(
        answer=answer_chain
    )
    | RunnablePassthrough.assign(
        sources=lambda x: extract_sources(x["documents"])
    )
)

    return chain