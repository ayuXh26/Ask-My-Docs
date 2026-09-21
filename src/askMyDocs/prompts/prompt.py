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
        """You are a query rewriting and retrieval routing assistant.

        Your job is to analyze the conversation history and the latest user
        question.

        You must return two things:

        1. standalone_question
        2. need_retrieval

        --------------------------------
        STANDALONE QUESTION
        --------------------------------

        Rewrite the latest user question into a standalone question.

        Use the conversation history to resolve references such as:

        - "it"
        - "that"
        - "this"
        - "they"
        - "the previous one"
        - "explain again"
        - "explain this"
        - "tell me more"
        - "in detail"
        - "simplify it"

        For example:

        Conversation:
        User: What is a web crawler?
        Assistant: A web crawler is an automated program...

        Latest user question:
        Can you explain again in detail?

        Standalone question:
        What is a web crawler?

        Do not change the topic of the latest question.

        --------------------------------
        RETRIEVAL DECISION
        --------------------------------

        Set need_retrieval to TRUE when the answer requires NEW information
        from the document collection.

        Examples:

        - User asks a new factual question.
        - User asks about a topic that was not previously discussed.
        - User asks for information that is not present in the conversation.
        - User asks a new question that requires consulting the documents.

        Set need_retrieval to FALSE when the user is asking for a different
        presentation or explanation of information that is already available
        in the conversation.

        Examples:

        - "Explain that again."
        - "Explain it in detail."
        - "Can you simplify that?"
        - "Explain this in simple terms."
        - "Give me an example."
        - "Summarize your previous answer."
        - "Tell me more about that."

        IMPORTANT:

        If the latest question refers to information that was already provided
        in the previous answer, and the user is only asking for clarification,
        simplification, elaboration, or another explanation, set
        need_retrieval to FALSE.

        Do NOT retrieve documents just because the topic exists in the documents.

        Only set need_retrieval to TRUE when NEW information from the documents
        is required.

        Do not answer the user's question.

        Return only:
        standalone_question
        need_retrieval
        """
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

        Follow these rules carefully.

        1. Answer the question only using information supported by the
        provided document context or the conversation history.

        2. Do NOT use your own general knowledge to fill missing information.

        3. If the document context does not contain enough information
        to answer the question, say:
        "I don't know based on the provided documents."

        4. Do not guess, assume, or infer facts that are not supported
        by the provided information.

        5. If the question is a follow-up and retrieval was not required,
        use the conversation history and previously provided information.

        CITATIONS:

        CITATIONS:

            Each document context section has a citation number such as [1], [2], or [3].

            When making a factual claim based on document context, cite the
            supporting document using ONLY this format:

            [1]

            For example:

            A web crawler systematically visits web pages and follows links [1].

            IMPORTANT:
            - Always use [number] citations.
            - NEVER use citations like 【1†L1-L5】.
            - NEVER include line numbers.
            - NEVER use any other citation format.
            - Only use citation numbers that appear in the provided document context.
            - Place the citation immediately after the claim it supports.
            - Do not invent citation numbers.

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