from src.askMyDocs.retrieval.retriever import create_retriever
from src.askMyDocs.citation.context_builder import build_context
from src.askMyDocs.prompts.prompt import rag_prompt
from src.askMyDocs.LLM.llm import create_llm


def create_rag_chain():

    retriever = create_retriever()
    llm = create_llm()

    def rag_chain(question):

        documents = retriever.invoke(question)

        context = build_context(documents)

        prompt = rag_prompt.invoke({
            "context": context,
            "question": question
        })

        response = llm.invoke(prompt)

        return response.content

    return rag_chain
