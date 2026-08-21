from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant.

    Answer the user's question using only the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
                                              )