from langchain_cohere import CohereRerank


def create_reranker():
    return CohereRerank(
        model = "rerank-v4.0-fast",
        top_n = 5,
    )

print("Cohere reranker initialized successfully")