from src.askMyDocs.rag_chain.rag_chain import create_rag_chain


def test_rag_chain():

    chain = create_rag_chain()

    questions = [
    "What is a Bloom filter used for in a URL shortener?",

    "Why is updating the autocomplete trie after every user query not practical at large scale?",

    "Explain the complete flow of a URL shortener from receiving a long URL to redirecting the user.",

    "What are the major scalability challenges in designing a web crawler, and how does the proposed architecture address them?",

    "Explain the CAP theorem.",

    "Who won 2023 cricket world cup?"
]

    for question in questions:

        print("\n" + "=" * 70)
        print("QUESTION:")
        print(question)

        response = chain.invoke(question)

        print("\nANSWER:")
        print(response["answer"].content)

        print("\nSOURCES:")

        for source in response["sources"]:
            print(
                f"- {source['source']} — Page {source['page']}"
            )


if __name__ == "__main__":
    test_rag_chain()