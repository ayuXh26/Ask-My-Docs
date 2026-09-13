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


rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query rewriting and routing assistant.

        Your job is to analyze the latest user question together with the
        conversation history.

        First, rewrite the latest user question into a standalone question
        that can be understood without the conversation history.

        Resolve references such as:
        "it", "that", "this", "they", "the previous one", etc.

        Then decide whether document retrieval is required.

        Set need_retrieval to true when:
        - the user asks a new factual question
        - the user asks about information that is not already available in the conversation
        - the user asks for additional information that was not covered in the previous answer
        - the question requires information from the source documents

        Set need_retrieval to false when:
        - the user asks to rephrase information already provided
        - the user asks to simplify information already provided
        - the user asks to explain information already provided in more detail
        - the user asks to clarify something already explained
        - the existing conversation contains enough information to answer the question

        Use the conversation history to determine whether the information
        needed to answer the question is already available.

        Do not set need_retrieval to true simply because the question is
        about a topic that exists in the documents. Set it to true only
        when new information from the documents is needed.

        Do not answer the question.

        Return your result as:
        standalone_question: <rewritten question>
        need_retrieval: <true or false>"""
        ),
        (
            "human",
            """
                Conversation history:
                {history}

                Latest user question:
                {question}
            """
        )
])


generate_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant answering questions using
        the provided document context and conversation history.

        Use the document context when it is available.

        If the question is a follow-up and retrieval was not required,
        use the conversation history and previously provided information
        to answer it.

        Do not make up information.

        Document context:
        {context}

        Conversation history:
        {messages}"""
    ),
    (
        "human",
        "{question}"
    )
])